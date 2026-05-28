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

bool tp_ep_profiler_start_if_requested(const Options &opt) {
    if (!opt.cuda_profiler_window) return false;
    if (!opt.cuda_profiler_all_devices) {
        if (opt.cuda_profiler_device >= 0) {
            const cudaError_t set_err = cudaSetDevice(opt.cuda_profiler_device);
            if (set_err != cudaSuccess) {
                std::fprintf(stderr,
                             "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                             opt.cuda_profiler_device, cudaGetErrorString(set_err));
                return false;
            }
        }
        const cudaError_t err = cudaProfilerStart();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "tp_ep_cuda_profiler_start_failed\terr\t%s\n",
                         cudaGetErrorString(err));
            return false;
        }
        std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstart\tdevice\t%d\n",
                     opt.cuda_profiler_device);
        return true;
    }
    bool active = false;
    for (int rank = 0; rank < kGpus; ++rank) {
        cudaError_t err = cudaSetDevice(opt.devices[rank]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            continue;
        }
        err = cudaProfilerStart();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_start_failed\tdevice\t%d\tphase\tstart\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            continue;
        }
        active = true;
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstart\tdevices\t%d\n",
                 active ? kGpus : 0);
    return active;
}

int tp_ep_profiler_stop_if_active(const Options &opt, bool *active) {
    if (!active || !*active) return 0;
    if (!opt.cuda_profiler_all_devices) {
        if (opt.cuda_profiler_device >= 0) {
            const cudaError_t set_err = cudaSetDevice(opt.cuda_profiler_device);
            if (set_err != cudaSuccess) {
                std::fprintf(stderr,
                             "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                             opt.cuda_profiler_device, cudaGetErrorString(set_err));
                *active = false;
                return 1;
            }
        }
        const cudaError_t err = cudaProfilerStop();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "tp_ep_cuda_profiler_stop_failed\terr\t%s\n",
                         cudaGetErrorString(err));
            *active = false;
            return 1;
        }
        std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstop\n");
        *active = false;
        return 0;
    }
    int failures = 0;
    for (int rank = 0; rank < kGpus; ++rank) {
        cudaError_t err = cudaSetDevice(opt.devices[rank]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tset_device\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            failures++;
            continue;
        }
        err = cudaProfilerStop();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "tp_ep_cuda_profiler_stop_failed\tdevice\t%d\tphase\tstop\terr\t%s\n",
                         opt.devices[rank], cudaGetErrorString(err));
            failures++;
        }
    }
    std::fprintf(stderr, "tp_ep_cuda_profiler_window\tstate\tstop\tfailures\t%d\n",
                 failures);
    *active = false;
    return failures == 0 ? 0 : 1;
}

struct TpEpProfilerWindowGuard {
    bool active = false;
    const Options &opt;

    explicit TpEpProfilerWindowGuard(const Options &opt)
        : active(tp_ep_profiler_start_if_requested(opt)), opt(opt) {}

    ~TpEpProfilerWindowGuard() {
        (void)tp_ep_profiler_stop_if_active(opt, &active);
    }
};

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
