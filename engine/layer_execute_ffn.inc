static int execute_ffn_delta(const ds4_layer_state *state,
                             const ds4_layer_execute_config *cfg,
                             const ds4_gpu_tensor *ffn_input,
                             ds4_gpu_tensor *ffn_delta,
                             ds4_layer_execute_report *report,
                             char *err,
                             size_t errlen) {
    const bool hash_mode = state->router_kind == DS4_V100_ROUTER_HASH && state->has_hash_router;
    const bool bias_mode = state->router_kind == DS4_V100_ROUTER_BIAS && state->has_bias_router;
    if (!hash_mode && !bias_mode) {
        return exec_error(err, errlen, "layer executor requires hash or bias router metadata");
    }

    const uint32_t hidden = state->hidden_size;
    const uint32_t mid = state->intermediate_size;
    const uint32_t routes = state->routes_per_token;
    if (routes == 0 || routes > 6u) {
        return exec_error(err, errlen, "unsupported route count %u", routes);
    }
    ds4_gpu_source_row_view router_v;
    ds4_gpu_source_row_view shared_gate_v;
    ds4_gpu_source_row_view shared_up_v;
    ds4_gpu_source_row_view shared_down_v;
    if (source_view(&state->router, &router_v, err, errlen) ||
        source_view(&state->shared_gate, &shared_gate_v, err, errlen) ||
        source_view(&state->shared_up, &shared_up_v, err, errlen) ||
        source_view(&state->shared_down, &shared_down_v, err, errlen)) {
        return 1;
    }

    const bool use_single_shared_pair =
        env_flag_enabled("DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE");
    int rc = 1;
    const bool use_scratch =
        cfg->batch_scratch != NULL && single_slot_batch_scratch_enabled();
    ds4_gpu_tensor *router_t = NULL;
    ds4_gpu_tensor *probs_t = NULL;
    ds4_gpu_tensor *selected_t = NULL;
    ds4_gpu_tensor *weights_t = NULL;
    ds4_gpu_tensor *routed_mid_t = NULL;
    ds4_gpu_tensor *accum_a = NULL;
    ds4_gpu_tensor *shared_gate_t = NULL;
    ds4_gpu_tensor *shared_up_t = NULL;
    ds4_gpu_tensor *shared_mid_t = NULL;
    ds4_gpu_tensor *shared_t = NULL;
    bool used_tp2 = false;
    if (use_scratch &&
        ensure_batch_scratch(cfg->batch_scratch,
                             hidden,
                             mid,
                             routes,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    router_t = use_scratch ? cfg->batch_scratch->ffn_router
        : ds4_gpu_tensor_alloc(256u * sizeof(float));
    probs_t = use_scratch ? cfg->batch_scratch->ffn_probs
        : ds4_gpu_tensor_alloc(256u * sizeof(float));
    selected_t = use_scratch ? cfg->batch_scratch->ffn_selected
        : ds4_gpu_tensor_alloc(6u * sizeof(int32_t));
    weights_t = use_scratch ? cfg->batch_scratch->ffn_weights
        : ds4_gpu_tensor_alloc(6u * sizeof(float));
    routed_mid_t = use_scratch ? cfg->batch_scratch->ffn_routed_mid
        : ds4_gpu_tensor_alloc((uint64_t)routes * mid * sizeof(float));
    accum_a = use_scratch ? cfg->batch_scratch->ffn_routed_out_view[0]
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    shared_gate_t = use_single_shared_pair ? NULL
        : (use_scratch ? cfg->batch_scratch->ffn_shared_gate
                       : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float)));
    shared_up_t = use_single_shared_pair ? NULL
        : (use_scratch ? cfg->batch_scratch->ffn_shared_up
                       : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float)));
    shared_mid_t = use_scratch ? cfg->batch_scratch->ffn_shared_mid
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    shared_t = use_scratch ? cfg->batch_scratch->ffn_shared
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));

    int32_t selected[6] = {0};
    float weights[6] = {0};
    const bool readback_routes = !cfg->suppress_router_readback;
    if (!router_t || !probs_t || !selected_t || !weights_t ||
        !routed_mid_t || !accum_a ||
        (!use_single_shared_pair && (!shared_gate_t || !shared_up_t)) ||
        !shared_mid_t || !shared_t) {
        exec_error(err, errlen, "failed to allocate FFN executor tensors");
        goto done;
    }

    if (ds4_gpu_arena_f32_matmul_f32(cfg->arena, &router_v, ffn_input, router_t) != 0) {
        exec_error(err, errlen, "router matmul failed");
        goto done;
    }
    if (!ds4_gpu_router_select_tensor(selected_t,
                                      weights_t,
                                      probs_t,
                                      cfg->model_map,
                                      cfg->model_size,
                                      bias_mode ? model_offset_for_binding(cfg, &state->router_bias) : 0,
                                      hash_mode ? model_offset_for_binding(cfg, &state->router_hash) : 0,
                                      hash_mode ? (uint32_t)state->router_hash.shape[1] : 0,
                                      cfg->router_token,
                                      0,
                                      0,
                                      bias_mode,
                                      hash_mode,
                                      router_t)) {
        exec_error(err, errlen, "router select failed");
        goto done;
    }
    if (readback_routes) {
        if (!ds4_gpu_tensor_read(selected_t, 0, selected, sizeof(selected)) ||
            !ds4_gpu_tensor_read(weights_t, 0, weights, sizeof(weights))) {
            exec_error(err, errlen, "router readback failed");
            goto done;
        }
        for (uint32_t route = 0; route < routes; route++) {
            if (selected[route] < 0 || (uint32_t)selected[route] >= state->routed_experts) {
                exec_error(err, errlen, "router selected invalid expert %d", selected[route]);
                goto done;
            }
        }
    }

    if (state->has_turbomind_routed) {
        const bool use_fused_gate_up =
            state->has_turbomind_fused_gate_up &&
            env_flag_default_enabled("DS4_V100_TURBOMIND_FUSED_GATE_UP");
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        int tm_rc = 1;
        const bool use_tp2 = tp2_routed_enabled(state, cfg, 1, use_fused_gate_up);
        if (use_tp2) {
            used_tp2 = true;
            ds4_layer_execute_report tp2_report;
            memset(&tp2_report, 0, sizeof(tp2_report));
            tm_rc = execute_turbomind_tp2_routed(state,
                                                 cfg,
                                                 selected_t,
                                                 weights_t,
                                                 ffn_input,
                                                 1,
                                                 accum_a,
                                                 &tp2_report,
                                                 err,
                                                 errlen);
            if (tm_rc == 0 && report) {
                report->timing_tp2_copy_in_ms = tp2_report.timing_tp2_copy_in_ms;
                report->timing_tp2_owner_ms = tp2_report.timing_tp2_owner_ms;
                report->timing_tp2_peer_ms = tp2_report.timing_tp2_peer_ms;
                report->timing_tp2_copy_out_ms = tp2_report.timing_tp2_copy_out_ms;
                report->timing_tp2_reduce_ms = tp2_report.timing_tp2_reduce_ms;
                report->timing_tp2_total_ms = tp2_report.timing_tp2_total_ms;
            }
        } else {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(
                  cfg->arena,
                  &state->turbomind_gate_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  ffn_input,
                  1,
                  accum_a)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32(
                  cfg->arena,
                  &state->turbomind_gate_view,
                  &state->turbomind_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  ffn_input,
                  1,
                  accum_a);
        }
        if (tm_rc != 0) {
            if (!err || !errlen || !err[0]) {
                exec_error(err, errlen, "TurboMind routed FFN failed");
            }
            goto done;
        }
    } else {
        ds4_route_matrices route0;
        if (ds4_layer_state_route_matrices(state, 0, &route0, err, errlen)) {
            goto done;
        }
        if (route0.gate.bytes != route0.up.bytes ||
            route0.gate.row_bytes != route0.up.row_bytes ||
            route0.gate.rows != route0.up.rows ||
            route0.gate.cols != route0.up.cols) {
            exec_error(err, errlen, "routed gate/up MXFP4 layouts differ");
            goto done;
        }
        if (ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32(
                cfg->arena,
                state->routed_gate_binding.shard_offset,
                state->routed_gate_binding.byte_length,
                state->routed_up_binding.shard_offset,
                state->routed_up_binding.byte_length,
                state->routed_down_binding.shard_offset,
                state->routed_down_binding.byte_length,
                route0.gate.bytes,
                (uint32_t)route0.gate.row_bytes,
                route0.down.bytes,
                (uint32_t)route0.down.row_bytes,
                hidden,
                mid,
                state->routed_experts,
                selected_t,
                weights_t,
                routes,
                ffn_input,
                routed_mid_t,
                accum_a) != 0) {
            exec_error(err, errlen, "grouped routed FFN failed");
            goto done;
        }
    }

    ds4_gpu_tensor *accum = accum_a;
    const bool use_shared_down_add =
        env_flag_enabled("DS4_V100_F8_SHARED_DOWN_ADD");

    int shared_rc = 0;
    if (use_single_shared_pair) {
        shared_rc =
            ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_f32(cfg->arena,
                                                       &shared_gate_v,
                                                       &shared_up_v,
                                                       ffn_input,
                                                       shared_mid_t,
                                                       10.0f,
                                                       1.0f) != 0;
    } else {
        shared_rc =
            ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                 &shared_gate_v,
                                                 ffn_input,
                                                 shared_gate_t) != 0 ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                 &shared_up_v,
                                                 ffn_input,
                                                 shared_up_t) != 0 ||
            !ds4_gpu_swiglu_tensor(shared_mid_t,
                                  shared_gate_t,
                                  shared_up_t,
                                  mid,
                                  10.0f,
                                  1.0f);
    }
    if (!shared_rc) {
        if (use_shared_down_add) {
            shared_rc =
                ds4_gpu_arena_f8_e4m3_b128_matmul_add_f32(cfg->arena,
                                                          &shared_down_v,
                                                          shared_mid_t,
                                                          accum,
                                                          ffn_delta) != 0;
        } else {
            shared_rc =
                ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfg->arena,
                                                     &shared_down_v,
                                                     shared_mid_t,
                                                     shared_t) != 0 ||
                !ds4_gpu_add_tensor(ffn_delta, accum, shared_t, hidden);
        }
    }
    if (shared_rc) {
        exec_error(err, errlen, "shared FFN failed");
        goto done;
    }

    if (report) {
        const double tp2_copy_in_ms = report->timing_tp2_copy_in_ms;
        const double tp2_owner_ms = report->timing_tp2_owner_ms;
        const double tp2_peer_ms = report->timing_tp2_peer_ms;
        const double tp2_copy_out_ms = report->timing_tp2_copy_out_ms;
        const double tp2_reduce_ms = report->timing_tp2_reduce_ms;
        const double tp2_total_ms = report->timing_tp2_total_ms;
        const double timing_attn_proj_ms = report->timing_attn_proj_ms;
        const double timing_attn_cache_ms = report->timing_attn_cache_ms;
        const double timing_attn_softmax_ms = report->timing_attn_softmax_ms;
        const double timing_attn_inverse_rope_ms = report->timing_attn_inverse_rope_ms;
        const double timing_attn_output_ms = report->timing_attn_output_ms;
        memset(report, 0, sizeof(*report));
        report->routes = routes;
        report->turbomind_routed = state->has_turbomind_routed ? 1u : 0u;
        report->turbomind_tp2_routed = used_tp2 ? 1u : 0u;
        report->timing_tp2_copy_in_ms = tp2_copy_in_ms;
        report->timing_tp2_owner_ms = tp2_owner_ms;
        report->timing_tp2_peer_ms = tp2_peer_ms;
        report->timing_tp2_copy_out_ms = tp2_copy_out_ms;
        report->timing_tp2_reduce_ms = tp2_reduce_ms;
        report->timing_tp2_total_ms = tp2_total_ms;
        report->timing_attn_proj_ms = timing_attn_proj_ms;
        report->timing_attn_cache_ms = timing_attn_cache_ms;
        report->timing_attn_softmax_ms = timing_attn_softmax_ms;
        report->timing_attn_inverse_rope_ms = timing_attn_inverse_rope_ms;
        report->timing_attn_output_ms = timing_attn_output_ms;
        if (readback_routes) {
            for (uint32_t i = 0; i < routes && i < 6u; i++) {
                report->selected_experts[i] = selected[i];
                report->route_weights[i] = weights[i];
            }
        }
    }
    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(shared_t);
        ds4_gpu_tensor_free(shared_mid_t);
        ds4_gpu_tensor_free(shared_up_t);
        ds4_gpu_tensor_free(shared_gate_t);
        ds4_gpu_tensor_free(accum_a);
        ds4_gpu_tensor_free(routed_mid_t);
        ds4_gpu_tensor_free(weights_t);
        ds4_gpu_tensor_free(selected_t);
        ds4_gpu_tensor_free(probs_t);
        ds4_gpu_tensor_free(router_t);
    }
    return rc;
}

static int execute_ffn_delta_batch(const ds4_layer_state *state,
                                   const ds4_layer_execute_config *cfgs,
                                   const ds4_gpu_tensor *const *ffn_inputs,
                                   ds4_gpu_tensor *const *ffn_deltas,
                                   uint32_t n_slots,
                                   ds4_layer_execute_report *reports,
                                   char *err,
                                   size_t errlen) {
    if (!state || !cfgs || !ffn_inputs || !ffn_deltas ||
        n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid FFN batch inputs");
    }
    if (n_slots == 1) {
        return execute_ffn_delta(state,
                                 &cfgs[0],
                                 ffn_inputs[0],
                                 ffn_deltas[0],
                                 reports ? &reports[0] : NULL,
                                 err,
                                 errlen);
    }

    const bool hash_mode = state->router_kind == DS4_V100_ROUTER_HASH && state->has_hash_router;
    const bool bias_mode = state->router_kind == DS4_V100_ROUTER_BIAS && state->has_bias_router;
    if (!hash_mode && !bias_mode) {
        return exec_error(err, errlen, "layer executor requires hash or bias router metadata");
    }

    const uint32_t hidden = state->hidden_size;
    const uint32_t mid = state->intermediate_size;
    const uint32_t routes = state->routes_per_token;
    if (routes == 0 || routes > 6u) {
        return exec_error(err, errlen, "unsupported route count %u", routes);
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ffn_inputs[slot] || !ffn_deltas[slot] ||
            ds4_gpu_tensor_bytes(ffn_inputs[slot]) < (uint64_t)hidden * sizeof(float) ||
            ds4_gpu_tensor_bytes(ffn_deltas[slot]) < (uint64_t)hidden * sizeof(float)) {
            return exec_error(err, errlen, "FFN batch tensor is too small");
        }
        if (cfgs[slot].arena != cfgs[0].arena ||
            cfgs[slot].model_map != cfgs[0].model_map ||
            cfgs[slot].model_size != cfgs[0].model_size ||
            cfgs[slot].batch_scratch != cfgs[0].batch_scratch) {
            return exec_error(err, errlen, "FFN batch cfgs must share arena/model");
        }
    }

    ds4_gpu_source_row_view router_v;
    ds4_gpu_source_row_view shared_gate_v;
    ds4_gpu_source_row_view shared_up_v;
    ds4_gpu_source_row_view shared_down_v;
    if (source_view(&state->router, &router_v, err, errlen) ||
        source_view(&state->shared_gate, &shared_gate_v, err, errlen) ||
        source_view(&state->shared_up, &shared_up_v, err, errlen) ||
        source_view(&state->shared_down, &shared_down_v, err, errlen)) {
        return 1;
    }

    const bool use_scratch = cfgs[0].batch_scratch != NULL;
    if (use_scratch && ensure_batch_scratch(cfgs[0].batch_scratch,
                                            hidden,
                                            mid,
                                            routes,
                                            state->q_lora_rank,
                                            state->q_width,
                                            state->kv_latent_width,
                                            state->attention_output_rank,
                                            err,
                                            errlen)) {
        return 1;
    }
    const bool batch_shared_f8 = env_flag_enabled("DS4_V100_BATCH_SHARED_F8");
    const bool use_single_shared_pair =
        env_flag_enabled("DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE");
    const bool use_shared_down_add =
        env_flag_enabled("DS4_V100_F8_SHARED_DOWN_ADD");
    ds4_gpu_tensor *router_t = use_scratch ? cfgs[0].batch_scratch->ffn_router
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * 256u * sizeof(float));
    ds4_gpu_tensor *probs_t = use_scratch ? cfgs[0].batch_scratch->ffn_probs
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * 256u * sizeof(float));
    ds4_gpu_tensor *selected_t = use_scratch ? cfgs[0].batch_scratch->ffn_selected
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * sizeof(int32_t));
    ds4_gpu_tensor *weights_t = use_scratch ? cfgs[0].batch_scratch->ffn_weights
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * sizeof(float));
    ds4_gpu_tensor *tokens_t = use_scratch ? cfgs[0].batch_scratch->ffn_tokens
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * sizeof(int32_t));
    ds4_gpu_tensor *input_ptrs_t = use_scratch ? cfgs[0].batch_scratch->ffn_input_ptrs
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * sizeof(void *));
    ds4_gpu_tensor *routed_mid_t = use_scratch ? cfgs[0].batch_scratch->ffn_routed_mid
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * routes * mid * sizeof(float));
    ds4_gpu_tensor *routed_out_t = use_scratch ? cfgs[0].batch_scratch->ffn_routed_out
        : ds4_gpu_tensor_alloc((uint64_t)n_slots * hidden * sizeof(float));
    ds4_gpu_tensor *shared_gate_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_gate
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_up_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_up
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_mid_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_mid
        : ds4_gpu_tensor_alloc((uint64_t)mid * sizeof(float));
    ds4_gpu_tensor *shared_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared
        : ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    ds4_gpu_tensor *shared_mid_batch_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_mid_batch
        : (batch_shared_f8 ? ds4_gpu_tensor_alloc((uint64_t)n_slots * mid * sizeof(float)) : NULL);
    ds4_gpu_tensor *shared_batch_t = use_scratch ? cfgs[0].batch_scratch->ffn_shared_batch
        : (batch_shared_f8 ? ds4_gpu_tensor_alloc((uint64_t)n_slots * hidden * sizeof(float)) : NULL);

    int rc = 1;
    int32_t tokens[DS4_V100_LAYER_MAX_BATCH] = {0};
    int32_t selected[DS4_V100_LAYER_MAX_BATCH * 6u] = {0};
    float weights[DS4_V100_LAYER_MAX_BATCH * 6u] = {0};
    bool readback_routes = false;
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!cfgs[slot].suppress_router_readback) {
            readback_routes = true;
            break;
        }
    }
    if (!router_t || !probs_t || !selected_t || !weights_t || !tokens_t || !input_ptrs_t ||
        !routed_mid_t || !routed_out_t ||
        !shared_gate_t || !shared_up_t || !shared_mid_t || !shared_t ||
        (batch_shared_f8 && (!shared_mid_batch_t || !shared_batch_t))) {
        exec_error(err, errlen, "failed to allocate FFN batch tensors");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        tokens[slot] = (int32_t)cfgs[slot].router_token;
        ds4_gpu_tensor *router_view =
            ds4_gpu_tensor_view(router_t,
                                (uint64_t)slot * 256u * sizeof(float),
                                256u * sizeof(float));
        if (!router_view) {
            exec_error(err, errlen, "FFN batch router view failed");
            goto done;
        }
        const int router_rc =
            ds4_gpu_arena_f32_matmul_f32(cfgs[slot].arena,
                                         &router_v,
                                         ffn_inputs[slot],
                                         router_view);
        ds4_gpu_tensor_free(router_view);
        if (router_rc != 0) {
            exec_error(err, errlen, "FFN batch router matmul failed");
            goto done;
        }
    }
    if (!ds4_gpu_tensor_write(tokens_t,
                              0,
                              tokens,
                              (uint64_t)n_slots * sizeof(int32_t))) {
        exec_error(err, errlen, "FFN batch token upload failed");
        goto done;
    }
    if (!ds4_gpu_router_select_batch_tensor(selected_t,
                                            weights_t,
                                            probs_t,
                                            cfgs[0].model_map,
                                            cfgs[0].model_size,
                                            bias_mode ? model_offset_for_binding(&cfgs[0], &state->router_bias) : 0,
                                            hash_mode ? model_offset_for_binding(&cfgs[0], &state->router_hash) : 0,
                                            hash_mode ? (uint32_t)state->router_hash.shape[1] : 0,
                                            0,
                                            0,
                                            bias_mode,
                                            hash_mode,
                                            router_t,
                                            tokens_t,
                                            n_slots)) {
        exec_error(err, errlen, "FFN batch router select failed");
        goto done;
    }
    if (readback_routes) {
        if (!ds4_gpu_tensor_read(selected_t,
                                 0,
                                 selected,
                                 (uint64_t)n_slots * routes * sizeof(int32_t)) ||
            !ds4_gpu_tensor_read(weights_t,
                                 0,
                                 weights,
                                 (uint64_t)n_slots * routes * sizeof(float))) {
            exec_error(err, errlen, "FFN batch router readback failed");
            goto done;
        }
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t route = 0; route < routes; route++) {
                const int32_t expert = selected[(uint64_t)slot * routes + route];
                if (expert < 0 || (uint32_t)expert >= state->routed_experts) {
                    exec_error(err, errlen, "FFN batch selected invalid expert %d", expert);
                    goto done;
                }
            }
        }
    }

    const uint64_t ffn_input_min_bytes = (uint64_t)hidden * sizeof(float);
    const bool use_cached_input_ptrs =
        use_scratch &&
        input_ptrs_t == cfgs[0].batch_scratch->ffn_input_ptrs &&
        ffn_inputs_match_scratch_norm(cfgs[0].batch_scratch, ffn_inputs, n_slots);
    if (use_cached_input_ptrs &&
        ensure_cached_ffn_input_ptrs(cfgs[0].batch_scratch,
                                     ffn_inputs,
                                     n_slots,
                                     ffn_input_min_bytes,
                                     err,
                                     errlen)) {
        goto done;
    }

    const bool use_fused_gate_up =
        state->has_turbomind_fused_gate_up &&
        env_flag_default_enabled("DS4_V100_TURBOMIND_FUSED_GATE_UP");
    const bool use_tp2 =
        use_scratch &&
        use_cached_input_ptrs &&
        cfgs[0].batch_scratch->ffn_norm_batch &&
        tp2_routed_enabled(state, &cfgs[0], n_slots, use_fused_gate_up);
    const bool use_direct_delta =
        env_flag_enabled("DS4_V100_FFN_DIRECT_DELTA") &&
        batch_shared_f8 &&
        state->has_turbomind_routed &&
        !use_tp2 &&
        use_scratch &&
        use_cached_input_ptrs &&
        ffn_outputs_match_scratch_delta(cfgs[0].batch_scratch, ffn_deltas, n_slots) &&
        cfgs[0].batch_scratch->ffn_norm_batch &&
        cfgs[0].batch_scratch->ffn_delta_batch;
    ds4_layer_execute_report tp2_report;
    memset(&tp2_report, 0, sizeof(tp2_report));
    if (use_direct_delta) {
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        if (ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32(
                cfgs[0].arena,
                &shared_gate_v,
                &shared_up_v,
                input_ptrs_t,
                n_slots,
                shared_mid_batch_t,
                10.0f,
                1.0f) != 0 ||
            ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                        &shared_down_v,
                                                        shared_mid_batch_t,
                                                        n_slots,
                                                        cfgs[0].batch_scratch->ffn_delta_batch) != 0) {
            exec_error(err, errlen, "shared FFN direct-delta batch failed");
            goto done;
        }
        const int tm_rc = use_fused_gate_up
            ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_accum_f32(
                  cfgs[0].arena,
                  &state->turbomind_gate_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  cfgs[0].batch_scratch->ffn_norm_batch,
                  n_slots,
                  cfgs[0].batch_scratch->ffn_delta_batch)
            : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_accum_f32(
                  cfgs[0].arena,
                  &state->turbomind_gate_view,
                  &state->turbomind_up_view,
                  &state->turbomind_down_view,
                  hidden,
                  mid,
                  state->routed_experts,
                  selected_t,
                  weights_t,
                  routes,
                  cfgs[0].batch_scratch->ffn_norm_batch,
                  n_slots,
                  cfgs[0].batch_scratch->ffn_delta_batch);
        if (tm_rc != 0) {
            exec_error(err, errlen, "TurboMind routed FFN direct-delta batch failed");
            goto done;
        }
        goto fill_reports;
    }

    if (state->has_turbomind_routed) {
        if (!use_fused_gate_up && !has_turbomind_separate_gate_up(state)) {
            exec_error(err, errlen,
                       "TurboMind fused gate_up disabled but no separate gate/up tensors are bound");
            goto done;
        }
        int tm_rc = 1;
        if (use_tp2) {
            tm_rc = execute_turbomind_tp2_routed(state,
                                                 &cfgs[0],
                                                 selected_t,
                                                 weights_t,
                                                 cfgs[0].batch_scratch->ffn_norm_batch,
                                                 n_slots,
                                                 routed_out_t,
                                                 &tp2_report,
                                                 err,
                                                 errlen);
        } else if (use_cached_input_ptrs) {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_batch_ptr_table_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      n_slots,
                      routed_out_t)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_batch_ptr_table_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_view,
                      &state->turbomind_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      n_slots,
                      routed_out_t);
        } else {
            tm_rc = use_fused_gate_up
                ? ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_batch_ptrs_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      ffn_inputs,
                      n_slots,
                      routed_out_t)
                : ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                      cfgs[0].arena,
                      &state->turbomind_gate_view,
                      &state->turbomind_up_view,
                      &state->turbomind_down_view,
                      hidden,
                      mid,
                      state->routed_experts,
                      selected_t,
                      weights_t,
                      routes,
                      input_ptrs_t,
                      ffn_inputs,
                      n_slots,
                      routed_out_t);
        }
        if (tm_rc != 0) {
            if (!err || !errlen || !err[0]) {
                exec_error(err, errlen, "TurboMind routed FFN batch failed");
            }
            goto done;
        }
    } else {
        ds4_route_matrices route0;
        if (ds4_layer_state_route_matrices(state, 0, &route0, err, errlen)) {
            goto done;
        }
        if (route0.gate.bytes != route0.up.bytes ||
            route0.gate.row_bytes != route0.up.row_bytes ||
            route0.gate.rows != route0.up.rows ||
            route0.gate.cols != route0.up.cols) {
            exec_error(err, errlen, "routed gate/up MXFP4 layouts differ");
            goto done;
        }
        if (ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32(
                cfgs[0].arena,
                state->routed_gate_binding.shard_offset,
                state->routed_gate_binding.byte_length,
                state->routed_up_binding.shard_offset,
                state->routed_up_binding.byte_length,
                state->routed_down_binding.shard_offset,
                state->routed_down_binding.byte_length,
                route0.gate.bytes,
                (uint32_t)route0.gate.row_bytes,
                route0.down.bytes,
                (uint32_t)route0.down.row_bytes,
                hidden,
                mid,
                state->routed_experts,
                selected_t,
                weights_t,
                routes,
                input_ptrs_t,
                ffn_inputs,
                n_slots,
                routed_mid_t,
                routed_out_t) != 0) {
            exec_error(err, errlen, "grouped routed FFN batch failed");
            goto done;
        }
    }

    if (batch_shared_f8) {
        const bool use_batch_shared_down_add =
            use_shared_down_add &&
            use_scratch &&
            ffn_outputs_match_scratch_delta(cfgs[0].batch_scratch, ffn_deltas, n_slots) &&
            cfgs[0].batch_scratch->ffn_delta_batch;
        if (ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_batch_ptr_table_f32(
                cfgs[0].arena,
                &shared_gate_v,
                &shared_up_v,
                input_ptrs_t,
                n_slots,
                shared_mid_batch_t,
                10.0f,
                1.0f) != 0) {
            exec_error(err, errlen, "shared FFN batch failed");
            goto done;
        }
        if (use_batch_shared_down_add) {
            if (ds4_gpu_arena_f8_e4m3_b128_matmul_batch_add_f32(
                    cfgs[0].arena,
                    &shared_down_v,
                    shared_mid_batch_t,
                    n_slots,
                    routed_out_t,
                    cfgs[0].batch_scratch->ffn_delta_batch) != 0) {
                exec_error(err, errlen, "shared FFN batch fused-add failed");
                goto done;
            }
        } else {
            if (ds4_gpu_arena_f8_e4m3_b128_matmul_batch_f32(cfgs[0].arena,
                                                            &shared_down_v,
                                                            shared_mid_batch_t,
                                                            n_slots,
                                                            shared_batch_t) != 0) {
                exec_error(err, errlen, "shared FFN batch failed");
                goto done;
            }
            for (uint32_t slot = 0; slot < n_slots; slot++) {
                ds4_gpu_tensor *routed_view = use_scratch
                    ? cfgs[0].batch_scratch->ffn_routed_out_view[slot]
                    : ds4_gpu_tensor_view(routed_out_t,
                                          (uint64_t)slot * hidden * sizeof(float),
                                          (uint64_t)hidden * sizeof(float));
                ds4_gpu_tensor *shared_view = use_scratch
                    ? cfgs[0].batch_scratch->ffn_shared_batch_view[slot]
                    : ds4_gpu_tensor_view(shared_batch_t,
                                          (uint64_t)slot * hidden * sizeof(float),
                                          (uint64_t)hidden * sizeof(float));
                if (!routed_view || !shared_view) {
                    if (!use_scratch) {
                        ds4_gpu_tensor_free(shared_view);
                        ds4_gpu_tensor_free(routed_view);
                    }
                    exec_error(err, errlen, "FFN batch output view failed");
                    goto done;
                }
                const int shared_rc = !ds4_gpu_add_tensor(ffn_deltas[slot],
                                                          routed_view,
                                                          shared_view,
                                                          hidden);
                if (!use_scratch) {
                    ds4_gpu_tensor_free(shared_view);
                    ds4_gpu_tensor_free(routed_view);
                }
                if (shared_rc) {
                    exec_error(err, errlen, "shared FFN batch slot %u failed", slot);
                    goto done;
                }
            }
        }
    } else {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_gpu_tensor *routed_view = use_scratch
                ? cfgs[0].batch_scratch->ffn_routed_out_view[slot]
                : ds4_gpu_tensor_view(routed_out_t,
                                      (uint64_t)slot * hidden * sizeof(float),
                                      (uint64_t)hidden * sizeof(float));
            if (!routed_view) {
                if (!use_scratch) ds4_gpu_tensor_free(routed_view);
                exec_error(err, errlen, "FFN batch routed output view failed");
                goto done;
            }
            int shared_rc = 0;
            if (use_single_shared_pair) {
                shared_rc =
                    ds4_gpu_arena_f8_e4m3_b128_pair_swiglu_f32(cfgs[slot].arena,
                                                               &shared_gate_v,
                                                               &shared_up_v,
                                                               ffn_inputs[slot],
                                                               shared_mid_t,
                                                               10.0f,
                                                               1.0f) != 0;
            } else {
                shared_rc =
                    ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                         &shared_gate_v,
                                                         ffn_inputs[slot],
                                                         shared_gate_t) != 0 ||
                    ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                         &shared_up_v,
                                                         ffn_inputs[slot],
                                                         shared_up_t) != 0 ||
                    !ds4_gpu_swiglu_tensor(shared_mid_t,
                                          shared_gate_t,
                                          shared_up_t,
                                          mid,
                                          10.0f,
                                          1.0f);
            }
            if (!shared_rc) {
                if (use_shared_down_add) {
                    shared_rc =
                        ds4_gpu_arena_f8_e4m3_b128_matmul_add_f32(cfgs[slot].arena,
                                                                  &shared_down_v,
                                                                  shared_mid_t,
                                                                  routed_view,
                                                                  ffn_deltas[slot]) != 0;
                } else {
                    shared_rc =
                        ds4_gpu_arena_f8_e4m3_b128_matmul_f32(cfgs[slot].arena,
                                                             &shared_down_v,
                                                             shared_mid_t,
                                                             shared_t) != 0 ||
                        !ds4_gpu_add_tensor(ffn_deltas[slot], routed_view, shared_t, hidden);
                }
            }
            if (!use_scratch) ds4_gpu_tensor_free(routed_view);
            if (shared_rc) {
                exec_error(err, errlen, "shared FFN batch slot %u failed", slot);
                goto done;
            }
        }
    }

fill_reports:
    if (reports) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_layer_execute_report *report = &reports[slot];
            memset(report, 0, sizeof(*report));
            report->routes = routes;
            report->turbomind_routed = state->has_turbomind_routed ? 1u : 0u;
            report->turbomind_tp2_routed = use_tp2 ? 1u : 0u;
            if (use_tp2) {
                report->timing_tp2_copy_in_ms = tp2_report.timing_tp2_copy_in_ms;
                report->timing_tp2_owner_ms = tp2_report.timing_tp2_owner_ms;
                report->timing_tp2_peer_ms = tp2_report.timing_tp2_peer_ms;
                report->timing_tp2_copy_out_ms = tp2_report.timing_tp2_copy_out_ms;
                report->timing_tp2_reduce_ms = tp2_report.timing_tp2_reduce_ms;
                report->timing_tp2_total_ms = tp2_report.timing_tp2_total_ms;
            }
            if (readback_routes) {
                for (uint32_t i = 0; i < routes && i < 6u; i++) {
                    report->selected_experts[i] = selected[(uint64_t)slot * routes + i];
                    report->route_weights[i] = weights[(uint64_t)slot * routes + i];
                }
            }
        }
    }

    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(shared_batch_t);
        ds4_gpu_tensor_free(shared_mid_batch_t);
        ds4_gpu_tensor_free(shared_t);
        ds4_gpu_tensor_free(shared_mid_t);
        ds4_gpu_tensor_free(shared_up_t);
        ds4_gpu_tensor_free(shared_gate_t);
        ds4_gpu_tensor_free(routed_out_t);
        ds4_gpu_tensor_free(routed_mid_t);
        ds4_gpu_tensor_free(input_ptrs_t);
        ds4_gpu_tensor_free(tokens_t);
        ds4_gpu_tensor_free(weights_t);
        ds4_gpu_tensor_free(selected_t);
        ds4_gpu_tensor_free(probs_t);
        ds4_gpu_tensor_free(router_t);
    }
    return rc;
}

int ds4_layer_execute_decode(
        const ds4_layer_state          *state,
        const ds4_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden,
        ds4_gpu_tensor                      *next_hidden,
        ds4_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen) {
    if (!hidden || !next_hidden) return exec_error(err, errlen, "missing hidden tensor");
    if (validate_execute_common(state, cfg, err, errlen)) return 1;
    const uint32_t hidden_n = state->hidden_size;
    if (ds4_gpu_tensor_bytes(hidden) < (uint64_t)hidden_n * sizeof(float) ||
        ds4_gpu_tensor_bytes(next_hidden) < (uint64_t)hidden_n * sizeof(float)) {
        return exec_error(err, errlen, "hidden tensor is too small");
    }

    ds4_gpu_tensor *attn_out = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    ds4_gpu_tensor *ffn_delta = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));

    int rc = 1;
    if (!attn_out || !residual || !ffn_norm || !ffn_delta) {
        exec_error(err, errlen, "failed to allocate layer executor tensors");
        goto done;
    }

    if (execute_attention_output(state, cfg, hidden, attn_out, report, err, errlen) ||
        !ds4_gpu_add_tensor(residual, hidden, attn_out, hidden_n) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        residual,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->ffn_norm),
                                        hidden_n,
                                        DS4_V100_RMS_EPS) ||
        execute_ffn_delta(state, cfg, ffn_norm, ffn_delta, report, err, errlen) ||
        !ds4_gpu_add_tensor(next_hidden, residual, ffn_delta, hidden_n)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "layer residual/FFN sequence failed");
        goto done;
    }

    rc = 0;

done:
    ds4_gpu_tensor_free(ffn_delta);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(attn_out);
    return rc;
}

int ds4_layer_execute_hc_decode(
        const ds4_layer_state          *state,
        const ds4_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden_hc,
        ds4_gpu_tensor                      *next_hidden_hc,
        ds4_layer_execute_report       *report,
        char                                *err,
        size_t                               errlen) {
    if (!hidden_hc || !next_hidden_hc) return exec_error(err, errlen, "missing HC tensor");
    if (validate_execute_common(state, cfg, err, errlen)) return 1;
    const uint32_t hidden_n = state->hidden_size;
    const uint64_t hc_values = (uint64_t)DS4_V100_N_HC * hidden_n;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        ds4_gpu_tensor_bytes(hidden_hc) < hc_bytes ||
        ds4_gpu_tensor_bytes(next_hidden_hc) < hc_bytes) {
        return exec_error(err, errlen, "HC tensor is too small");
    }
    if (state->hc_attn_fn.n_shape_dims != 2 ||
        state->hc_ffn_fn.n_shape_dims != 2 ||
        state->hc_attn_fn.shape[0] != hc_values ||
        state->hc_ffn_fn.shape[0] != hc_values ||
        state->hc_attn_fn.shape[1] != DS4_V100_HC_MIX ||
        state->hc_ffn_fn.shape[1] != DS4_V100_HC_MIX) {
        return exec_error(err, errlen, "HC control dimensions do not match DS4");
    }

    int rc = 1;
    const bool use_scratch =
        cfg->batch_scratch != NULL && single_slot_batch_scratch_enabled();
    ds4_gpu_tensor *hc_norm = NULL;
    ds4_gpu_tensor *hc_mix = NULL;
    ds4_gpu_tensor *attn_split = NULL;
    ds4_gpu_tensor *ffn_split = NULL;
    ds4_gpu_tensor *attn_cur = NULL;
    ds4_gpu_tensor *attn_out = NULL;
    ds4_gpu_tensor *after_attn_hc = NULL;
    ds4_gpu_tensor *ffn_cur = NULL;
    ds4_gpu_tensor *ffn_norm = NULL;
    ds4_gpu_tensor *ffn_delta = NULL;
    if (use_scratch &&
        ensure_batch_scratch(cfg->batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    if (use_scratch) {
        hc_norm = cfg->batch_scratch->hc_norm[0];
        hc_mix = cfg->batch_scratch->hc_mix[0];
        attn_split = cfg->batch_scratch->attn_split[0];
        ffn_split = cfg->batch_scratch->ffn_split[0];
        attn_cur = cfg->batch_scratch->attn_cur[0];
        attn_out = cfg->batch_scratch->attn_out[0];
        after_attn_hc = cfg->batch_scratch->after_attn_hc[0];
        ffn_cur = cfg->batch_scratch->ffn_cur[0];
        ffn_norm = cfg->batch_scratch->ffn_norm[0];
        ffn_delta = cfg->batch_scratch->ffn_delta[0];
    } else {
        hc_norm = ds4_gpu_tensor_alloc(hc_bytes);
        hc_mix = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        attn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        ffn_split = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
        attn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        attn_out = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        after_attn_hc = ds4_gpu_tensor_alloc(hc_bytes);
        ffn_cur = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        ffn_norm = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        ffn_delta = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
    }
    if (!hc_norm || !hc_mix || !attn_split || !ffn_split || !attn_cur ||
        !attn_out || !after_attn_hc || !ffn_cur || !ffn_norm || !ffn_delta) {
        exec_error(err, errlen, "failed to allocate HC layer executor tensors");
        goto done;
    }

    const bool profile = env_flag_enabled("DS4_V100_PROFILE_DECODE");
    double profile_last_ms = 0.0;
    double timing_hc_attn_ms = 0.0;
    double timing_attention_ms = 0.0;
    double timing_hc_ffn_ms = 0.0;
    double timing_ffn_ms = 0.0;
    double timing_hc_final_ms = 0.0;
    if (profile) {
        if (!ds4_gpu_synchronize()) {
            exec_error(err, errlen, "HC profile start sync failed");
            goto done;
        }
        profile_last_ms = monotonic_ms();
    }

    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm, hidden_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   model_offset_for_binding(cfg, &state->hc_attn_fn),
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(attn_cur,
                                              attn_split,
                                              hc_mix,
                                              hidden_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              model_offset_for_binding(cfg, &state->hc_attn_scale),
                                              model_offset_for_binding(cfg, &state->hc_attn_base),
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "HC attention prep failed");
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_hc_attn_ms)) {
        exec_error(err, errlen, "HC attention prep profile sync failed");
        goto done;
    }

    if (execute_attention_output(state, cfg, attn_cur, attn_out, report, err, errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_attention_ms)) {
        exec_error(err, errlen, "HC attention profile sync failed");
        goto done;
    }

    if (!ds4_gpu_hc_expand_split_tensor(after_attn_hc,
                                        attn_out,
                                        hidden_hc,
                                        attn_split,
                                        hidden_n,
                                        DS4_V100_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc_norm, after_attn_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   model_offset_for_binding(cfg, &state->hc_ffn_fn),
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur,
                                              ffn_split,
                                              hc_mix,
                                              after_attn_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              model_offset_for_binding(cfg, &state->hc_ffn_scale),
                                              model_offset_for_binding(cfg, &state->hc_ffn_base),
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        ffn_cur,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->ffn_norm),
                                        hidden_n,
                                        DS4_V100_RMS_EPS)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "HC layer execution sequence failed");
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_hc_ffn_ms)) {
        exec_error(err, errlen, "HC FFN prep profile sync failed");
        goto done;
    }

    if (execute_ffn_delta(state, cfg, ffn_norm, ffn_delta, report, err, errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_ffn_ms)) {
        exec_error(err, errlen, "HC FFN profile sync failed");
        goto done;
    }

    if (!ds4_gpu_hc_expand_split_tensor(next_hidden_hc,
                                        ffn_delta,
                                        after_attn_hc,
                                        ffn_split,
                                        hidden_n,
                                        DS4_V100_N_HC)) {
        exec_error(err, errlen, "HC final expansion failed");
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_hc_final_ms)) {
        exec_error(err, errlen, "HC final profile sync failed");
        goto done;
    }
    if (report && profile) {
        report->timing_hc_attn_ms = timing_hc_attn_ms;
        report->timing_attention_ms = timing_attention_ms;
        report->timing_hc_ffn_ms = timing_hc_ffn_ms;
        report->timing_ffn_ms = timing_ffn_ms;
        report->timing_hc_final_ms = timing_hc_final_ms;
        report->timing_total_ms = timing_hc_attn_ms +
                                  timing_attention_ms +
                                  timing_hc_ffn_ms +
                                  timing_ffn_ms +
                                  timing_hc_final_ms;
    }

    if (cfg->checkpoint_fn) {
        ds4_layer_execute_checkpoint cp = {
            .layer = cfg->checkpoint_layer,
            .kind = DS4_V100_HC_CHECKPOINT_AFTER_ATTN,
            .hc = after_attn_hc,
            .hc_bytes = hc_bytes,
        };
        if (cfg->checkpoint_fn(&cp, cfg->checkpoint_user, err, errlen)) goto done;
    }

    rc = 0;

done:
    if (!use_scratch) {
        ds4_gpu_tensor_free(ffn_delta);
        ds4_gpu_tensor_free(ffn_norm);
        ds4_gpu_tensor_free(ffn_cur);
        ds4_gpu_tensor_free(after_attn_hc);
        ds4_gpu_tensor_free(attn_out);
        ds4_gpu_tensor_free(attn_cur);
        ds4_gpu_tensor_free(ffn_split);
        ds4_gpu_tensor_free(attn_split);
        ds4_gpu_tensor_free(hc_mix);
        ds4_gpu_tensor_free(hc_norm);
    }
    return rc;
}

int ds4_layer_execute_hc_prepare_ffn(
        const ds4_layer_state          *state,
        const ds4_layer_execute_config *cfg,
        const ds4_gpu_tensor                *hidden_hc,
        ds4_gpu_tensor                      *next_hidden_hc,
        uint32_t                             scratch_slot,
        ds4_layer_prepared_ffn         *prepared,
        char                                *err,
        size_t                               errlen) {
    if (!hidden_hc || !next_hidden_hc || !prepared) {
        return exec_error(err, errlen, "missing HC FFN-prepare input");
    }
    if (!cfg || !cfg->batch_scratch) {
        return exec_error(err, errlen, "HC FFN-prepare requires batch scratch");
    }
    if (scratch_slot >= DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "HC FFN-prepare scratch slot out of range");
    }
    if (cfg->checkpoint_fn) {
        return exec_error(err, errlen, "HC FFN-prepare does not support checkpoints");
    }
    if (validate_execute_common(state, cfg, err, errlen)) return 1;
    const uint32_t hidden_n = state->hidden_size;
    const uint64_t hc_values = (uint64_t)DS4_V100_N_HC * hidden_n;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        ds4_gpu_tensor_bytes(hidden_hc) < hc_bytes ||
        ds4_gpu_tensor_bytes(next_hidden_hc) < hc_bytes) {
        return exec_error(err, errlen, "HC FFN-prepare tensor is too small");
    }
    if (state->hc_attn_fn.n_shape_dims != 2 ||
        state->hc_ffn_fn.n_shape_dims != 2 ||
        state->hc_attn_fn.shape[0] != hc_values ||
        state->hc_ffn_fn.shape[0] != hc_values ||
        state->hc_attn_fn.shape[1] != DS4_V100_HC_MIX ||
        state->hc_ffn_fn.shape[1] != DS4_V100_HC_MIX) {
        return exec_error(err, errlen, "HC FFN-prepare control dimensions do not match DS4");
    }
    if (ensure_batch_scratch(cfg->batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        return 1;
    }

    ds4_gpu_tensor *hc_norm = cfg->batch_scratch->hc_norm[scratch_slot];
    ds4_gpu_tensor *hc_mix = cfg->batch_scratch->hc_mix[scratch_slot];
    ds4_gpu_tensor *attn_split = cfg->batch_scratch->attn_split[scratch_slot];
    ds4_gpu_tensor *ffn_split = cfg->batch_scratch->ffn_split[scratch_slot];
    ds4_gpu_tensor *attn_cur = cfg->batch_scratch->attn_cur[scratch_slot];
    ds4_gpu_tensor *attn_out = cfg->batch_scratch->attn_out[scratch_slot];
    ds4_gpu_tensor *after_attn_hc = cfg->batch_scratch->after_attn_hc[scratch_slot];
    ds4_gpu_tensor *ffn_cur = cfg->batch_scratch->ffn_cur[scratch_slot];
    ds4_gpu_tensor *ffn_norm = cfg->batch_scratch->ffn_norm[scratch_slot];
    ds4_gpu_tensor *ffn_delta = cfg->batch_scratch->ffn_delta[scratch_slot];
    if (!hc_norm || !hc_mix || !attn_split || !ffn_split || !attn_cur ||
        !attn_out || !after_attn_hc || !ffn_cur || !ffn_norm || !ffn_delta) {
        return exec_error(err, errlen, "HC FFN-prepare scratch tensors are missing");
    }

    if (!ds4_gpu_rms_norm_plain_tensor(hc_norm, hidden_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   model_offset_for_binding(cfg, &state->hc_attn_fn),
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(attn_cur,
                                              attn_split,
                                              hc_mix,
                                              hidden_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              model_offset_for_binding(cfg, &state->hc_attn_scale),
                                              model_offset_for_binding(cfg, &state->hc_attn_base),
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        execute_attention_output(state, cfg, attn_cur, attn_out, NULL, err, errlen) ||
        !ds4_gpu_hc_expand_split_tensor(after_attn_hc,
                                        attn_out,
                                        hidden_hc,
                                        attn_split,
                                        hidden_n,
                                        DS4_V100_N_HC) ||
        !ds4_gpu_rms_norm_plain_tensor(hc_norm, after_attn_hc, (uint32_t)hc_values, DS4_V100_RMS_EPS) ||
        !ds4_gpu_matmul_f32_tensor(hc_mix,
                                   cfg->model_map,
                                   cfg->model_size,
                                   model_offset_for_binding(cfg, &state->hc_ffn_fn),
                                   hc_values,
                                   DS4_V100_HC_MIX,
                                   hc_norm,
                                   1) ||
        !ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur,
                                              ffn_split,
                                              hc_mix,
                                              after_attn_hc,
                                              cfg->model_map,
                                              cfg->model_size,
                                              model_offset_for_binding(cfg, &state->hc_ffn_scale),
                                              model_offset_for_binding(cfg, &state->hc_ffn_base),
                                              hidden_n,
                                              DS4_V100_N_HC,
                                              DS4_V100_HC_SINKHORN_ITERS,
                                              DS4_V100_RMS_EPS) ||
        !ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                        ffn_cur,
                                        cfg->model_map,
                                        cfg->model_size,
                                        model_offset_for_binding(cfg, &state->ffn_norm),
                                        hidden_n,
                                        DS4_V100_RMS_EPS)) {
        if (err && err[0] == '\0') exec_error(err, errlen, "HC FFN-prepare sequence failed");
        return 1;
    }

    *prepared = (ds4_layer_prepared_ffn) {
        .scratch_slot = scratch_slot,
        .hidden_hc = hidden_hc,
        .next_hidden_hc = next_hidden_hc,
        .after_attn_hc = after_attn_hc,
        .ffn_split = ffn_split,
        .ffn_norm = ffn_norm,
        .ffn_delta = ffn_delta,
    };
    return 0;
}

int ds4_layer_execute_hc_finish_ffn_batch(
        const ds4_layer_state           *state,
        const ds4_layer_execute_config  *cfgs,
        ds4_layer_prepared_ffn          *prepared,
        uint32_t                              n_slots,
        ds4_layer_execute_report        *reports,
        char                                 *err,
        size_t                                errlen) {
    if (!state || !cfgs || !prepared || n_slots == 0 ||
        n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid HC FFN-finish inputs");
    }
    const uint32_t hidden_n = state->hidden_size;
    ds4_gpu_tensor *ffn_norm[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_delta[DS4_V100_LAYER_MAX_BATCH] = {0};
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!prepared[slot].after_attn_hc ||
            !prepared[slot].ffn_split ||
            !prepared[slot].ffn_norm ||
            !prepared[slot].ffn_delta ||
            !prepared[slot].next_hidden_hc) {
            return exec_error(err, errlen, "HC FFN-finish slot is not prepared");
        }
        ffn_norm[slot] = prepared[slot].ffn_norm;
        ffn_delta[slot] = prepared[slot].ffn_delta;
    }
    if (execute_ffn_delta_batch(state,
                                cfgs,
                                (const ds4_gpu_tensor *const *)ffn_norm,
                                ffn_delta,
                                n_slots,
                                reports,
                                err,
                                errlen)) {
        return 1;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_hc_expand_split_tensor(prepared[slot].next_hidden_hc,
                                            prepared[slot].ffn_delta,
                                            prepared[slot].after_attn_hc,
                                            prepared[slot].ffn_split,
                                            hidden_n,
                                            DS4_V100_N_HC)) {
            return exec_error(err, errlen, "HC FFN-finish expansion failed");
        }
    }
    return 0;
}

int ds4_layer_execute_hc_decode_batch(
        const ds4_layer_state           *state,
        const ds4_layer_execute_config  *cfgs,
        const ds4_gpu_tensor *const          *hidden_hc,
        ds4_gpu_tensor *const                *next_hidden_hc,
        uint32_t                              n_slots,
        ds4_layer_execute_report        *reports,
        char                                 *err,
        size_t                                errlen) {
    if (!state || !cfgs || !hidden_hc || !next_hidden_hc ||
        n_slots == 0 || n_slots > DS4_V100_LAYER_MAX_BATCH) {
        return exec_error(err, errlen, "invalid HC batch inputs");
    }
    if (n_slots == 1) {
        return ds4_layer_execute_hc_decode(state,
                                                &cfgs[0],
                                                hidden_hc[0],
                                                next_hidden_hc[0],
                                                reports ? &reports[0] : NULL,
                                                err,
                                                errlen);
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (cfgs[slot].checkpoint_fn) {
            for (uint32_t i = 0; i < n_slots; i++) {
                if (ds4_layer_execute_hc_decode(state,
                                                     &cfgs[i],
                                                     hidden_hc[i],
                                                     next_hidden_hc[i],
                                                     reports ? &reports[i] : NULL,
                                                     err,
                                                     errlen)) {
                    return 1;
                }
            }
            return 0;
        }
    }

    const uint32_t hidden_n = state->hidden_size;
    const uint64_t hc_values = (uint64_t)DS4_V100_N_HC * hidden_n;
    const uint64_t hc_bytes = hc_values * sizeof(float);
    if (hidden_n != DS4_V100_OUT_GROUP_DIM ||
        state->hc_attn_fn.n_shape_dims != 2 ||
        state->hc_ffn_fn.n_shape_dims != 2 ||
        state->hc_attn_fn.shape[0] != hc_values ||
        state->hc_ffn_fn.shape[0] != hc_values ||
        state->hc_attn_fn.shape[1] != DS4_V100_HC_MIX ||
        state->hc_ffn_fn.shape[1] != DS4_V100_HC_MIX) {
        return exec_error(err, errlen, "HC batch control dimensions do not match DS4");
    }

    ds4_gpu_tensor *hc_norm[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *hc_mix[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_split[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_split[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_cur[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *attn_out[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *after_attn_hc[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_cur[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_norm[DS4_V100_LAYER_MAX_BATCH] = {0};
    ds4_gpu_tensor *ffn_delta[DS4_V100_LAYER_MAX_BATCH] = {0};

    int rc = 1;
    const bool use_scratch = cfgs[0].batch_scratch != NULL;
    if (use_scratch &&
        ensure_batch_scratch(cfgs[0].batch_scratch,
                             hidden_n,
                             state->intermediate_size,
                             state->routes_per_token,
                             state->q_lora_rank,
                             state->q_width,
                             state->kv_latent_width,
                             state->attention_output_rank,
                             err,
                             errlen)) {
        goto done;
    }
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!hidden_hc[slot] || !next_hidden_hc[slot] ||
            ds4_gpu_tensor_bytes(hidden_hc[slot]) < hc_bytes ||
            ds4_gpu_tensor_bytes(next_hidden_hc[slot]) < hc_bytes ||
            validate_execute_common(state, &cfgs[slot], err, errlen)) {
            goto done;
        }
        if (cfgs[slot].arena != cfgs[0].arena ||
            cfgs[slot].model_map != cfgs[0].model_map ||
            cfgs[slot].model_size != cfgs[0].model_size ||
            cfgs[slot].batch_scratch != cfgs[0].batch_scratch) {
            exec_error(err, errlen, "HC batch cfgs must share arena/model");
            goto done;
        }
        if (use_scratch) {
            hc_norm[slot] = cfgs[0].batch_scratch->hc_norm[slot];
            hc_mix[slot] = cfgs[0].batch_scratch->hc_mix[slot];
            attn_split[slot] = cfgs[0].batch_scratch->attn_split[slot];
            ffn_split[slot] = cfgs[0].batch_scratch->ffn_split[slot];
            attn_cur[slot] = cfgs[0].batch_scratch->attn_cur[slot];
            attn_out[slot] = cfgs[0].batch_scratch->attn_out[slot];
            after_attn_hc[slot] = cfgs[0].batch_scratch->after_attn_hc[slot];
            ffn_cur[slot] = cfgs[0].batch_scratch->ffn_cur[slot];
            ffn_norm[slot] = cfgs[0].batch_scratch->ffn_norm[slot];
            ffn_delta[slot] = cfgs[0].batch_scratch->ffn_delta[slot];
        } else {
            hc_norm[slot] = ds4_gpu_tensor_alloc(hc_bytes);
            hc_mix[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            attn_split[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            ffn_split[slot] = ds4_gpu_tensor_alloc(DS4_V100_HC_MIX * sizeof(float));
            attn_cur[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            attn_out[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            after_attn_hc[slot] = ds4_gpu_tensor_alloc(hc_bytes);
            ffn_cur[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            ffn_norm[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
            ffn_delta[slot] = ds4_gpu_tensor_alloc((uint64_t)hidden_n * sizeof(float));
        }
        if (!hc_norm[slot] || !hc_mix[slot] || !attn_split[slot] || !ffn_split[slot] ||
            !attn_cur[slot] || !attn_out[slot] || !after_attn_hc[slot] ||
            !ffn_cur[slot] || !ffn_norm[slot] || !ffn_delta[slot]) {
            exec_error(err, errlen, "failed to allocate HC batch tensors");
            goto done;
        }
    }

    const bool profile = env_flag_enabled("DS4_V100_PROFILE_DECODE");
    double profile_last_ms = 0.0;
    double timing_hc_attn_ms = 0.0;
    double timing_attention_ms = 0.0;
    double timing_hc_ffn_ms = 0.0;
    double timing_ffn_ms = 0.0;
    double timing_hc_final_ms = 0.0;
    if (profile) {
        if (!ds4_gpu_synchronize()) {
            exec_error(err, errlen, "HC batch profile start sync failed");
            goto done;
        }
        profile_last_ms = monotonic_ms();
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_rms_norm_plain_tensor(hc_norm[slot],
                                           hidden_hc[slot],
                                           (uint32_t)hc_values,
                                           DS4_V100_RMS_EPS) ||
            !ds4_gpu_matmul_f32_tensor(hc_mix[slot],
                                       cfgs[slot].model_map,
                                       cfgs[slot].model_size,
                                       model_offset_for_binding(&cfgs[slot], &state->hc_attn_fn),
                                       hc_values,
                                       DS4_V100_HC_MIX,
                                       hc_norm[slot],
                                       1) ||
            !ds4_gpu_hc_split_weighted_sum_tensor(attn_cur[slot],
                                                  attn_split[slot],
                                                  hc_mix[slot],
                                                  hidden_hc[slot],
                                                  cfgs[slot].model_map,
                                                  cfgs[slot].model_size,
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_attn_scale),
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_attn_base),
                                                  hidden_n,
                                                  DS4_V100_N_HC,
                                                  DS4_V100_HC_SINKHORN_ITERS,
                                                  DS4_V100_RMS_EPS)) {
            if (err && err[0] == '\0') exec_error(err, errlen, "HC batch attention prep failed");
            goto done;
        }
        if (profile && !profile_mark(&profile_last_ms, &timing_hc_attn_ms)) {
            exec_error(err, errlen, "HC batch attention prep profile sync failed");
            goto done;
        }
    }

    if (execute_attention_output_batch(state,
                                       cfgs,
                                       (const ds4_gpu_tensor *const *)attn_cur,
                                       attn_out,
                                       n_slots,
                                       err,
                                       errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_attention_ms)) {
        exec_error(err, errlen, "HC batch attention profile sync failed");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_hc_expand_split_tensor(after_attn_hc[slot],
                                            attn_out[slot],
                                            hidden_hc[slot],
                                            attn_split[slot],
                                            hidden_n,
                                            DS4_V100_N_HC) ||
            !ds4_gpu_rms_norm_plain_tensor(hc_norm[slot],
                                           after_attn_hc[slot],
                                           (uint32_t)hc_values,
                                           DS4_V100_RMS_EPS) ||
            !ds4_gpu_matmul_f32_tensor(hc_mix[slot],
                                       cfgs[slot].model_map,
                                       cfgs[slot].model_size,
                                       model_offset_for_binding(&cfgs[slot], &state->hc_ffn_fn),
                                       hc_values,
                                       DS4_V100_HC_MIX,
                                       hc_norm[slot],
                                       1) ||
            !ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur[slot],
                                                  ffn_split[slot],
                                                  hc_mix[slot],
                                                  after_attn_hc[slot],
                                                  cfgs[slot].model_map,
                                                  cfgs[slot].model_size,
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_ffn_scale),
                                                  model_offset_for_binding(&cfgs[slot], &state->hc_ffn_base),
                                                  hidden_n,
                                                  DS4_V100_N_HC,
                                                  DS4_V100_HC_SINKHORN_ITERS,
                                                  DS4_V100_RMS_EPS) ||
            !ds4_gpu_rms_norm_weight_tensor(ffn_norm[slot],
                                            ffn_cur[slot],
                                            cfgs[slot].model_map,
                                            cfgs[slot].model_size,
                                            model_offset_for_binding(&cfgs[slot], &state->ffn_norm),
                                            hidden_n,
                                            DS4_V100_RMS_EPS)) {
            if (err && err[0] == '\0') exec_error(err, errlen, "HC batch FFN prep failed");
            goto done;
        }
        if (profile && !profile_mark(&profile_last_ms, &timing_hc_ffn_ms)) {
            exec_error(err, errlen, "HC batch FFN prep profile sync failed");
            goto done;
        }
    }

    if (execute_ffn_delta_batch(state,
                                cfgs,
                                (const ds4_gpu_tensor *const *)ffn_norm,
                                ffn_delta,
                                n_slots,
                                reports,
                                err,
                                errlen)) {
        goto done;
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_ffn_ms)) {
        exec_error(err, errlen, "HC batch FFN profile sync failed");
        goto done;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if (!ds4_gpu_hc_expand_split_tensor(next_hidden_hc[slot],
                                            ffn_delta[slot],
                                            after_attn_hc[slot],
                                            ffn_split[slot],
                                            hidden_n,
                                            DS4_V100_N_HC)) {
            exec_error(err, errlen, "HC batch final expansion failed");
            goto done;
        }
    }
    if (profile && !profile_mark(&profile_last_ms, &timing_hc_final_ms)) {
        exec_error(err, errlen, "HC batch final profile sync failed");
        goto done;
    }
    if (reports && profile) {
        const double timing_total_ms = timing_hc_attn_ms +
                                       timing_attention_ms +
                                       timing_hc_ffn_ms +
                                       timing_ffn_ms +
                                       timing_hc_final_ms;
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            reports[slot].timing_hc_attn_ms = timing_hc_attn_ms;
            reports[slot].timing_attention_ms = timing_attention_ms;
            reports[slot].timing_hc_ffn_ms = timing_hc_ffn_ms;
            reports[slot].timing_ffn_ms = timing_ffn_ms;
            reports[slot].timing_hc_final_ms = timing_hc_final_ms;
            reports[slot].timing_total_ms = timing_total_ms;
        }
    }

    rc = 0;

done:
    if (!use_scratch) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            ds4_gpu_tensor_free(ffn_delta[slot]);
            ds4_gpu_tensor_free(ffn_norm[slot]);
            ds4_gpu_tensor_free(ffn_cur[slot]);
            ds4_gpu_tensor_free(after_attn_hc[slot]);
            ds4_gpu_tensor_free(attn_out[slot]);
            ds4_gpu_tensor_free(attn_cur[slot]);
            ds4_gpu_tensor_free(ffn_split[slot]);
            ds4_gpu_tensor_free(attn_split[slot]);
            ds4_gpu_tensor_free(hc_mix[slot]);
            ds4_gpu_tensor_free(hc_norm[slot]);
        }
    }
    return rc;
}
