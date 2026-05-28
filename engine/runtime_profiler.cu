void log_hc_current_full_rank_parity(const Options &opt,
                                     RankState ranks[kGpus],
                                     int layer,
                                     size_t elems);
int nccl_broadcast_f32_from_device0_to_current_full(
    const Options &opt,
    RankState ranks[kGpus],
    const float *src_device0,
    uint64_t elems,
    const char *label);

int enqueue_cross_gpu_stream_barrier(RankState ranks[kGpus],
                                     bool include_copy_streams);

void sync_typed_kv_boundary(const Options &opt, RankState ranks[kGpus]) {
    if (opt.decode_cudagraph_gate) {
        const int rc = enqueue_cross_gpu_stream_barrier(ranks, false);
        if (rc != 0) {
            std::fprintf(stderr,
                         "tp_ep_typed_kv_graph_boundary_failed\trc\t%d\n",
                         rc);
            std::abort();
        }
        return;
    }
    for (int rank = 0; rank < kGpus; ++rank) {
        CHECK_CUDA(cudaSetDevice(ranks[rank].device));
        if (opt.true_ds4_attention_typed_kv_stream_sync_gate) {
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
}

struct TensorF32Stats {
    int finite_bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
};

struct TensorF32DiffStats {
    int bad = 0;
    size_t first_bad = (size_t)-1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

struct HalfInputDiffStats {
    unsigned long long compared = 0;
    unsigned long long mismatches = 0;
    int first_mismatch = -1;
    float max_abs = 0.0f;
};

TensorF32Stats collect_tensor_f32_stats(const float *ptr, size_t elems,
                                        cudaStream_t stream);
TensorF32Stats collect_raw_swa_row_stats(const float *ptr, uint32_t slots,
                                         uint32_t raw_rows, uint32_t raw_row,
                                         uint32_t head_dim,
                                         cudaStream_t stream);
TensorF32DiffStats collect_tensor_f32_diff_stats(const float *a, const float *b,
                                                 size_t elems,
                                                 cudaStream_t stream);
void merge_tensor_stats(TensorF32Stats *dst, const TensorF32Stats &src);
void log_tensor_f32_stats(const char *tag, int layer, int rank_id,
                          const float *ptr, size_t elems, cudaStream_t stream);
bool should_log_routed_semantic_stats(const Options &opt);
bool should_log_reference_hc_window(const Options &opt);
