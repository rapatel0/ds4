#include "engine/tp_runtime.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned long long parse_u64(const char *s) {
    return strtoull(s, nullptr, 10);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s [--ctx N] [--slots N] [--scratch-mib N] "
                 "[--kv-dtype f16|f8|f8_e4m3_b128|f8_e5m2_b128|q8_0] [--dense-kv-slice] "
                 "[--typed-kv-row] [--device-kv-row] [--device-kv-row-at-position] "
                 "[--device-kv-row-at-position-bounded] [--device-kv-history-row] "
                 "[--kind attn|attn_raw|indexer] "
                 "[--layer N] [--slot N] [--position N] [--history-row N] "
                 "[--indexer on|off]\n",
                 argv0);
}

static bool parse_kv(const char *s, ds4_tp_kv_dtype *out) {
    if (std::strcmp(s, "f16") == 0) *out = DS4_V100_TP_KV_F16;
    else if (std::strcmp(s, "f8") == 0 || std::strcmp(s, "f8_e4m3_b128") == 0)
        *out = DS4_V100_TP_KV_F8_E4M3_B128;
    else if (std::strcmp(s, "f8_e5m2_b128") == 0 || std::strcmp(s, "e5m2") == 0)
        *out = DS4_V100_TP_KV_F8_E5M2_B128;
    else if (std::strcmp(s, "q8") == 0 || std::strcmp(s, "q8_0") == 0)
        *out = DS4_V100_TP_KV_Q8_0;
    else return false;
    return true;
}

int main(int argc, char **argv) {
    ds4_tp_runtime_config cfg;
    ds4_tp_runtime_default_config(&cfg);
    bool dense_kv_slice = false;
    bool typed_kv_row = false;
    bool device_kv_row = false;
    bool device_kv_row_at_position = false;
    bool device_kv_row_at_position_bounded = false;
    bool device_kv_history_row = false;
    ds4_tp_kv_row_kind row_kind = DS4_V100_TP_KV_ROW_ATTN;
    int layer = 2;
    uint32_t slot = 0;
    unsigned long long position = 0;
    uint32_t history_row = (uint32_t)-1;
    int indexer = 0;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *val = i + 1 < argc ? argv[i + 1] : nullptr;
        if (std::strcmp(arg, "--ctx") == 0 && val) {
            cfg.ctx = parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--slots") == 0 && val) {
            cfg.slots = (uint32_t)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--scratch-mib") == 0 && val) {
            cfg.scratch_bytes = parse_u64(val) * 1024ull * 1024ull;
            ++i;
        } else if (std::strcmp(arg, "--kv-dtype") == 0 && val) {
            if (!parse_kv(val, &cfg.kv_dtype)) {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (std::strcmp(arg, "--dense-kv-slice") == 0) {
            dense_kv_slice = true;
        } else if (std::strcmp(arg, "--typed-kv-row") == 0) {
            typed_kv_row = true;
        } else if (std::strcmp(arg, "--device-kv-row") == 0) {
            device_kv_row = true;
        } else if (std::strcmp(arg, "--device-kv-row-at-position") == 0) {
            device_kv_row_at_position = true;
        } else if (std::strcmp(arg, "--device-kv-row-at-position-bounded") == 0) {
            device_kv_row_at_position_bounded = true;
        } else if (std::strcmp(arg, "--device-kv-history-row") == 0) {
            device_kv_history_row = true;
        } else if (std::strcmp(arg, "--kind") == 0 && val) {
            if (std::strcmp(val, "attn") == 0 || std::strcmp(val, "attention") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_ATTN;
            } else if (std::strcmp(val, "attn_raw") == 0 ||
                       std::strcmp(val, "attention_raw") == 0 ||
                       std::strcmp(val, "raw") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_ATTN_RAW;
            } else if (std::strcmp(val, "indexer") == 0) {
                row_kind = DS4_V100_TP_KV_ROW_INDEXER;
            } else {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (std::strcmp(arg, "--layer") == 0 && val) {
            layer = (int)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--slot") == 0 && val) {
            slot = (uint32_t)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--position") == 0 && val) {
            position = parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--history-row") == 0 && val) {
            history_row = (uint32_t)parse_u64(val);
            ++i;
        } else if (std::strcmp(arg, "--indexer") == 0 && val) {
            if (std::strcmp(val, "on") == 0 || std::strcmp(val, "true") == 0 ||
                std::strcmp(val, "1") == 0) {
                indexer = 1;
            } else if (std::strcmp(val, "off") == 0 || std::strcmp(val, "false") == 0 ||
                       std::strcmp(val, "0") == 0) {
                indexer = 0;
            } else {
                usage(argv[0]);
                return 2;
            }
            ++i;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    char err[512] = {0};
    ds4_tp_runtime *rt = nullptr;
    if (ds4_tp_runtime_open(&rt, &cfg, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_open_failed\t%s\n", err);
        return 1;
    }

    if (dense_kv_slice) {
        ds4_tp_dense_kv_result result;
        if (ds4_tp_runtime_dense_kv_slice(rt, layer, slot, position, indexer,
                                               &result, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dense_kv_slice_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        std::printf("tp_dense_kv_slice\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\t"
                    "attn_row=%llu\tindexer_row=%llu\tmax_abs=%.9f\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.layer, result.ratio, result.slot,
                    (unsigned long long)result.position,
                    (unsigned long long)result.attn_row,
                    (unsigned long long)result.indexer_row,
                    result.max_abs);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\tattn_offset\t%llu\tattn_row_bytes\t%llu\t"
                        "indexer_offset\t%llu\tindexer_row_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.attn_offset[gpu],
                        (unsigned long long)result.attn_row_bytes[gpu],
                        (unsigned long long)result.indexer_offset[gpu],
                        (unsigned long long)result.indexer_row_bytes[gpu]);
        }
        ds4_tp_runtime_close(rt);
        return result.max_abs <= 0.0 ? 0 : 1;
    }

    if (typed_kv_row) {
        ds4_tp_kv_row_roundtrip_result result;
        if (ds4_tp_runtime_kv_row_roundtrip_f32(rt, layer, slot, position,
                                                     row_kind, &result, err,
                                                     sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_typed_kv_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const char *kind_name =
            result.view.kind == DS4_V100_TP_KV_ROW_INDEXER
                ? "indexer"
                : (result.view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW ? "attn_raw"
                                                                    : "attn");
        std::printf("tp_typed_kv_row\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\tkind=%s\t"
                    "physical_row=%llu\tlogical_cols=%u\tlogical_row_bytes=%llu\t"
                    "row_bytes_per_gpu=%llu\tbad_values=%u\tbyte_mismatches=%u\tfirst_bad_index=%u\t"
                    "first_bad_got=%u\tfirst_bad_expected=%u\tmax_abs=%.9f\t"
                    "mean_abs=%.9f\tchecksum=%llu\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.view.layer, result.view.ratio, result.view.slot,
                    (unsigned long long)result.view.position, kind_name,
                    (unsigned long long)result.view.physical_row,
                    result.view.logical_cols,
                    (unsigned long long)result.view.logical_row_bytes,
                    (unsigned long long)result.view.row_bytes[0],
                    result.bad_values, result.byte_mismatches, result.first_bad_index,
                    (unsigned int)result.first_bad_got,
                    (unsigned int)result.first_bad_expected,
                    result.max_abs, result.mean_abs,
                    (unsigned long long)result.checksum);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\toffset\t%llu\trow_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.view.offset[gpu],
                        (unsigned long long)result.view.row_bytes[gpu]);
        }
        ds4_tp_runtime_close(rt);
        return result.bad_values == 0 && result.max_abs == 0.0 ? 0 : 1;
    }

    if (device_kv_row) {
        ds4_tp_kv_device_roundtrip_result result;
        if (ds4_tp_runtime_kv_row_device_roundtrip_f32(
                rt, layer, slot, position, row_kind, &result, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_device_kv_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const char *kind_name =
            result.view.kind == DS4_V100_TP_KV_ROW_INDEXER
                ? "indexer"
                : (result.view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW ? "attn_raw"
                                                                    : "attn");
        std::printf("tp_device_kv_row\tctx=%llu\tslots=%u\thidden=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\tkind=%s\t"
                    "physical_row=%llu\tlogical_cols=%u\tlogical_row_bytes=%llu\t"
                    "row_bytes_per_gpu=%llu\tbad_values=%u\tmax_abs=%.9f\t"
                    "mean_abs=%.9f\tchecksum=%llu\n",
                    (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                    result.view.layer, result.view.ratio, result.view.slot,
                    (unsigned long long)result.view.position, kind_name,
                    (unsigned long long)result.view.physical_row,
                    result.view.logical_cols,
                    (unsigned long long)result.view.logical_row_bytes,
                    (unsigned long long)result.view.row_bytes[0],
                    result.bad_values, result.max_abs, result.mean_abs,
                    (unsigned long long)result.checksum);
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::printf("gpu\t%d\toffset\t%llu\trow_bytes\t%llu\n",
                        gpu,
                        (unsigned long long)result.view.offset[gpu],
                        (unsigned long long)result.view.row_bytes[gpu]);
        }
        ds4_tp_runtime_close(rt);
        return result.bad_values == 0 && result.max_abs == 0.0 ? 0 : 1;
    }

    if (device_kv_row_at_position) {
        ds4_tp_kv_row_view view;
        if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, row_kind,
                                       &view, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dynamic_position_view_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const uint64_t src_values = view.logical_cols;
        const uint64_t dst_stride =
            row_kind == DS4_V100_TP_KV_ROW_ATTN_RAW
                ? 128ull * (uint64_t)view.logical_cols
                : (uint64_t)view.logical_cols;
        const uint64_t src_bytes = src_values * sizeof(float);
        const uint64_t dst_bytes = dst_stride * sizeof(float);
        void *src[DS4_V100_TP_MAX_GPUS] = {};
        void *static_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *dynamic_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *pos_dev[DS4_V100_TP_MAX_GPUS] = {};
        std::vector<float> host_src((size_t)src_values);
        for (uint64_t i = 0; i < src_values; ++i) {
            host_src[(size_t)i] =
                (float)((int)(i % 97ull) - 48) * 0.015625f +
                (float)((int)(position % 17ull) - 8) * 0.03125f;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaError_t rc = cudaSetDevice(gpu);
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "cudaSetDevice failed: %s\n",
                             cudaGetErrorString(rc));
                ds4_tp_runtime_close(rt);
                return 1;
            }
            cudaMalloc(&src[gpu], (size_t)src_bytes);
            cudaMalloc(&static_dst[gpu], (size_t)dst_bytes);
            cudaMalloc(&dynamic_dst[gpu], (size_t)dst_bytes);
            cudaMalloc(&pos_dev[gpu], sizeof(uint64_t));
            cudaMemcpy(src[gpu], host_src.data(), (size_t)src_bytes,
                       cudaMemcpyHostToDevice);
            cudaMemset(static_dst[gpu], 0, (size_t)dst_bytes);
            cudaMemset(dynamic_dst[gpu], 0, (size_t)dst_bytes);
            cudaMemcpy(pos_dev[gpu], &position, sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
        }
        if (ds4_tp_runtime_kv_row_store_f32_device(
                rt, layer, slot, position, row_kind, (const void **)src,
                err, sizeof(err)) != 0 ||
            ds4_tp_runtime_kv_row_load_f32_device(
                rt, layer, slot, position, row_kind, static_dst, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_static_device_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        void *streams[DS4_V100_TP_MAX_GPUS] = {};
        if (ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position(
                rt, layer, slot, 1, row_kind, (const void **)src, src_values,
                streams, (const void *const *)pos_dev, err, sizeof(err)) != 0 ||
            ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position(
                rt, layer, slot, 1, row_kind, dynamic_dst, dst_stride,
                streams, (const void *const *)pos_dev, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dynamic_position_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaDeviceSynchronize();
        }
        const uint64_t compare_offset =
            row_kind == DS4_V100_TP_KV_ROW_ATTN_RAW
                ? view.physical_row * (uint64_t)view.logical_cols
                : 0ull;
        uint32_t bad = 0;
        double max_abs = 0.0;
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::vector<float> static_host((size_t)view.logical_cols);
            std::vector<float> dynamic_host((size_t)view.logical_cols);
            cudaSetDevice(gpu);
            cudaMemcpy(static_host.data(), static_dst[gpu],
                       (size_t)src_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(dynamic_host.data(),
                       (float *)dynamic_dst[gpu] + compare_offset,
                       (size_t)src_bytes, cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < view.logical_cols; ++i) {
                const double diff =
                    std::fabs((double)static_host[(size_t)i] -
                              (double)dynamic_host[(size_t)i]);
                if (diff != 0.0) bad++;
                if (diff > max_abs) max_abs = diff;
            }
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaFree(pos_dev[gpu]);
            cudaFree(dynamic_dst[gpu]);
            cudaFree(static_dst[gpu]);
            cudaFree(src[gpu]);
        }
        std::printf("tp_dynamic_position_kv_row\tctx=%llu\tslots=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\t"
                    "kind=%s\tphysical_row=%llu\tlogical_cols=%u\t"
                    "bad_values=%u\tmax_abs=%.9f\n",
                    (unsigned long long)cfg.ctx, cfg.slots, view.layer,
                    view.ratio, view.slot, (unsigned long long)view.position,
                    view.kind == DS4_V100_TP_KV_ROW_INDEXER
                        ? "indexer"
                        : (view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW
                               ? "attn_raw"
                               : "attn"),
                    (unsigned long long)view.physical_row, view.logical_cols,
                    bad, max_abs);
        ds4_tp_runtime_close(rt);
        return bad == 0 && max_abs == 0.0 ? 0 : 1;
    }

    if (device_kv_history_row) {
        static const uint32_t kSmokeBoundedRows = 8u;
        ds4_tp_kv_row_view current_view;
        if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, row_kind,
                                       &current_view, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_history_view_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const uint64_t emit_ratio =
            row_kind == DS4_V100_TP_KV_ROW_INDEXER ? 4ull
                                                   : (uint64_t)current_view.ratio;
        if (row_kind == DS4_V100_TP_KV_ROW_ATTN_RAW || emit_ratio == 0ull) {
            std::fprintf(stderr, "tp_runtime_history_unsupported_kind\n");
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const uint64_t emitted_count = (position + 1ull) / emit_ratio;
        if (emitted_count == 0ull) {
            std::fprintf(stderr, "tp_runtime_history_no_emitted_rows\n");
            ds4_tp_runtime_close(rt);
            return 1;
        }
        if (history_row == (uint32_t)-1) {
            history_row = (uint32_t)((emitted_count - 1ull) % kSmokeBoundedRows);
        }
        if (history_row >= kSmokeBoundedRows ||
            (uint64_t)history_row >= emitted_count) {
            std::fprintf(stderr, "tp_runtime_history_row_out_of_range\n");
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const uint64_t offset =
            (emitted_count - 1ull + (uint64_t)kSmokeBoundedRows -
             (uint64_t)history_row) %
            (uint64_t)kSmokeBoundedRows;
        const uint64_t emission = emitted_count - offset;
        const uint64_t history_position = emission * emit_ratio - 1ull;
        ds4_tp_kv_row_view history_view;
        if (ds4_tp_runtime_kv_row_view(rt, layer, slot, history_position, row_kind,
                                       &history_view, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_history_source_view_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        const uint64_t row_values = history_view.logical_cols;
        const uint64_t bounded_stride =
            (uint64_t)kSmokeBoundedRows * (uint64_t)history_view.logical_cols;
        const uint64_t row_bytes = row_values * sizeof(float);
        const uint64_t bounded_bytes = bounded_stride * sizeof(float);
        void *src_full[DS4_V100_TP_MAX_GPUS] = {};
        void *src_row[DS4_V100_TP_MAX_GPUS] = {};
        void *static_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *dynamic_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *pos_dev[DS4_V100_TP_MAX_GPUS] = {};
        std::vector<float> host_src((size_t)bounded_stride, 0.0f);
        for (uint64_t i = 0; i < row_values; ++i) {
            host_src[(size_t)((uint64_t)history_row * row_values + i)] =
                (float)((int)(i % 89ull) - 44) * 0.017578125f +
                (float)((int)(history_position % 19ull) - 9) * 0.02734375f;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaError_t rc = cudaSetDevice(gpu);
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "cudaSetDevice failed: %s\n",
                             cudaGetErrorString(rc));
                ds4_tp_runtime_close(rt);
                return 1;
            }
            cudaMalloc(&src_full[gpu], (size_t)bounded_bytes);
            cudaMalloc(&static_dst[gpu], (size_t)row_bytes);
            cudaMalloc(&dynamic_dst[gpu], (size_t)bounded_bytes);
            cudaMalloc(&pos_dev[gpu], sizeof(uint64_t));
            cudaMemcpy(src_full[gpu], host_src.data(), (size_t)bounded_bytes,
                       cudaMemcpyHostToDevice);
            src_row[gpu] =
                (float *)src_full[gpu] + (uint64_t)history_row * row_values;
            cudaMemset(static_dst[gpu], 0, (size_t)row_bytes);
            cudaMemset(dynamic_dst[gpu], 0, (size_t)bounded_bytes);
            cudaMemcpy(pos_dev[gpu], &position, sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
        }
        if (ds4_tp_runtime_kv_row_store_f32_device(
                rt, layer, slot, history_position, row_kind,
                (const void **)src_row, err, sizeof(err)) != 0 ||
            ds4_tp_runtime_kv_row_load_f32_device(
                rt, layer, slot, history_position, row_kind, static_dst, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_static_history_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        void *streams[DS4_V100_TP_MAX_GPUS] = {};
        if (ds4_tp_runtime_kv_rows_load_f32_device_streams_at_history_row(
                rt, layer, slot, 1, row_kind, history_row, dynamic_dst,
                bounded_stride, kSmokeBoundedRows, streams,
                (const void *const *)pos_dev, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dynamic_history_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaDeviceSynchronize();
        }
        uint32_t bad = 0;
        double max_abs = 0.0;
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::vector<float> static_host((size_t)row_values);
            std::vector<float> dynamic_host((size_t)row_values);
            cudaSetDevice(gpu);
            cudaMemcpy(static_host.data(), static_dst[gpu],
                       (size_t)row_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(dynamic_host.data(),
                       (float *)dynamic_dst[gpu] +
                           (uint64_t)history_row * row_values,
                       (size_t)row_bytes, cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < history_view.logical_cols; ++i) {
                const double diff =
                    std::fabs((double)static_host[(size_t)i] -
                              (double)dynamic_host[(size_t)i]);
                if (diff != 0.0) bad++;
                if (diff > max_abs) max_abs = diff;
            }
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaFree(pos_dev[gpu]);
            cudaFree(dynamic_dst[gpu]);
            cudaFree(static_dst[gpu]);
            cudaFree(src_full[gpu]);
        }
        std::printf("tp_dynamic_history_kv_row\tctx=%llu\tslots=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\t"
                    "history_position=%llu\tkind=%s\tphysical_row=%llu\t"
                    "bounded_row=%u\tlogical_cols=%u\tbad_values=%u\t"
                    "max_abs=%.9f\n",
                    (unsigned long long)cfg.ctx, cfg.slots, history_view.layer,
                    history_view.ratio, history_view.slot,
                    (unsigned long long)position,
                    (unsigned long long)history_position,
                    history_view.kind == DS4_V100_TP_KV_ROW_INDEXER
                        ? "indexer"
                        : "attn",
                    (unsigned long long)history_view.physical_row, history_row,
                    history_view.logical_cols, bad, max_abs);
        ds4_tp_runtime_close(rt);
        return bad == 0 && max_abs == 0.0 ? 0 : 1;
    }

    if (device_kv_row_at_position_bounded) {
        static const uint32_t kSmokeBoundedRows = 8u;
        ds4_tp_kv_row_view view;
        if (ds4_tp_runtime_kv_row_view(rt, layer, slot, position, row_kind,
                                       &view, err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dynamic_bounded_view_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        uint64_t logical_row = view.physical_row;
        if (row_kind == DS4_V100_TP_KV_ROW_ATTN && view.ratio != 0) {
            logical_row -= 128ull;
        }
        const uint64_t bounded_row = logical_row % kSmokeBoundedRows;
        const uint64_t row_values = view.logical_cols;
        const uint64_t bounded_stride =
            (uint64_t)kSmokeBoundedRows * (uint64_t)view.logical_cols;
        const uint64_t row_bytes = row_values * sizeof(float);
        const uint64_t bounded_bytes = bounded_stride * sizeof(float);
        void *src_full[DS4_V100_TP_MAX_GPUS] = {};
        void *src_row[DS4_V100_TP_MAX_GPUS] = {};
        void *static_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *dynamic_dst[DS4_V100_TP_MAX_GPUS] = {};
        void *pos_dev[DS4_V100_TP_MAX_GPUS] = {};
        std::vector<float> host_src((size_t)bounded_stride, 0.0f);
        for (uint64_t i = 0; i < row_values; ++i) {
            host_src[(size_t)(bounded_row * row_values + i)] =
                (float)((int)(i % 97ull) - 48) * 0.015625f +
                (float)((int)(position % 17ull) - 8) * 0.03125f;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaError_t rc = cudaSetDevice(gpu);
            if (rc != cudaSuccess) {
                std::fprintf(stderr, "cudaSetDevice failed: %s\n",
                             cudaGetErrorString(rc));
                ds4_tp_runtime_close(rt);
                return 1;
            }
            cudaMalloc(&src_full[gpu], (size_t)bounded_bytes);
            cudaMalloc(&static_dst[gpu], (size_t)row_bytes);
            cudaMalloc(&dynamic_dst[gpu], (size_t)bounded_bytes);
            cudaMalloc(&pos_dev[gpu], sizeof(uint64_t));
            cudaMemcpy(src_full[gpu], host_src.data(), (size_t)bounded_bytes,
                       cudaMemcpyHostToDevice);
            src_row[gpu] = (float *)src_full[gpu] + bounded_row * row_values;
            cudaMemset(static_dst[gpu], 0, (size_t)row_bytes);
            cudaMemset(dynamic_dst[gpu], 0, (size_t)bounded_bytes);
            cudaMemcpy(pos_dev[gpu], &position, sizeof(uint64_t),
                       cudaMemcpyHostToDevice);
        }
        if (ds4_tp_runtime_kv_row_store_f32_device(
                rt, layer, slot, position, row_kind, (const void **)src_row,
                err, sizeof(err)) != 0 ||
            ds4_tp_runtime_kv_row_load_f32_device(
                rt, layer, slot, position, row_kind, static_dst, err,
                sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_static_bounded_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        void *streams[DS4_V100_TP_MAX_GPUS] = {};
        if (ds4_tp_runtime_kv_rows_store_f32_device_streams_at_position_bounded(
                rt, layer, slot, 1, row_kind, (const void **)src_full,
                bounded_stride, kSmokeBoundedRows, streams,
                (const void *const *)pos_dev, err, sizeof(err)) != 0 ||
            ds4_tp_runtime_kv_rows_load_f32_device_streams_at_position_bounded(
                rt, layer, slot, 1, row_kind, dynamic_dst, bounded_stride,
                kSmokeBoundedRows, streams, (const void *const *)pos_dev,
                err, sizeof(err)) != 0) {
            std::fprintf(stderr, "tp_runtime_dynamic_bounded_row_failed\t%s\n", err);
            ds4_tp_runtime_close(rt);
            return 1;
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaDeviceSynchronize();
        }
        uint32_t bad = 0;
        double max_abs = 0.0;
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            std::vector<float> static_host((size_t)row_values);
            std::vector<float> dynamic_host((size_t)row_values);
            cudaSetDevice(gpu);
            cudaMemcpy(static_host.data(), static_dst[gpu],
                       (size_t)row_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(dynamic_host.data(),
                       (float *)dynamic_dst[gpu] + bounded_row * row_values,
                       (size_t)row_bytes, cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < view.logical_cols; ++i) {
                const double diff =
                    std::fabs((double)static_host[(size_t)i] -
                              (double)dynamic_host[(size_t)i]);
                if (diff != 0.0) bad++;
                if (diff > max_abs) max_abs = diff;
            }
        }
        for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
            cudaSetDevice(gpu);
            cudaFree(pos_dev[gpu]);
            cudaFree(dynamic_dst[gpu]);
            cudaFree(static_dst[gpu]);
            cudaFree(src_full[gpu]);
        }
        std::printf("tp_dynamic_bounded_position_kv_row\tctx=%llu\tslots=%u\t"
                    "layer=%d\tratio=%d\tslot=%u\tposition=%llu\t"
                    "kind=%s\tphysical_row=%llu\tbounded_row=%llu\t"
                    "logical_cols=%u\tbad_values=%u\tmax_abs=%.9f\n",
                    (unsigned long long)cfg.ctx, cfg.slots, view.layer,
                    view.ratio, view.slot, (unsigned long long)view.position,
                    view.kind == DS4_V100_TP_KV_ROW_INDEXER
                        ? "indexer"
                        : (view.kind == DS4_V100_TP_KV_ROW_ATTN_RAW
                               ? "attn_raw"
                               : "attn"),
                    (unsigned long long)view.physical_row,
                    (unsigned long long)bounded_row, view.logical_cols, bad,
                    max_abs);
        ds4_tp_runtime_close(rt);
        return bad == 0 && max_abs == 0.0 ? 0 : 1;
    }

    double max_abs = 0.0;
    if (ds4_tp_runtime_fixture(rt, &max_abs, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tp_runtime_fixture_failed\t%s\n", err);
        ds4_tp_runtime_close(rt);
        return 1;
    }

    ds4_tp_runtime_report report;
    ds4_tp_runtime_get_report(rt, &report);
    std::printf("tp_runtime_smoke\tctx=%llu\tslots=%u\thidden=%u\t"
                "scratch_bytes=%llu\tfixture_max_abs=%.9f\n",
                (unsigned long long)cfg.ctx, cfg.slots, cfg.hidden,
                (unsigned long long)cfg.scratch_bytes, max_abs);
    for (int gpu = 0; gpu < DS4_V100_TP_MAX_GPUS; ++gpu) {
        const ds4_tp_gpu_report *g = &report.gpu[gpu];
        std::printf("gpu\t%d\thidden_bytes\t%llu\tkv_bytes\t%llu\t"
                    "comp_state_bytes\t%llu\tscratch_bytes\t%llu\ttotal_bytes\t%llu\n",
                    gpu,
                    (unsigned long long)g->hidden_bytes,
                    (unsigned long long)g->kv_bytes,
                    (unsigned long long)g->comp_state_bytes,
                    (unsigned long long)g->scratch_bytes,
                    (unsigned long long)g->total_bytes);
    }
    ds4_tp_runtime_close(rt);
    return max_abs <= 1.0e-5 ? 0 : 1;
}
