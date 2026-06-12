/* Sprint 602 Phase A0: NCCL allreduce fold-order calibration probe.
 *
 * Goal: recover the EXACT per-element float-accumulation grouping NCCL
 * (2.19.3, this pod, NCCL_P2P_LEVEL=NVL, auto rings/algo/proto) uses for
 * the four captured sum-allreduces of the hc-class set, so the s602 kernel
 * reductions can reproduce it bit-exactly and the s597 control anchor
 * survives (SPRINT-602 bit-anchor policy).
 *
 * Method: create the 8-rank communicator exactly like the appliance
 * (ncclCommInitAll, devices 0..7, one process), enqueue the same grouped
 * ncclAllReduce calls the engine issues per layer (same group structure,
 * same counts), with order-sensitive random inputs; then search the
 * hypothesis space of NCCL's ring reduce-scatter chunk schedule:
 *
 *   result[e] = left-fold of v[ring[(start+k) % 8]][e], k = 0..7
 *   start = chunk(e) + delta, chunk(e) from NCCL's ring AR loop:
 *     loop over gridOffset += nc*8*realChunk:
 *       realChunk = divUp(remaining, nc*8*minChunk)*minChunk (cap chunkCap)
 *       channel bid covers [gridOffset + bid*8*realChunk, ... + 8*realChunk)
 *       chunk c = (e - channelBase)/realChunk
 *   ring for channel bid = ringlist[(rbase + bid) % nrings]
 *
 * Search: nc in {1..8}, minChunk in {16..4096} pow2, delta in {0..7},
 * rbase in {0..nrings-1}. Hypotheses must match EVERY element of EVERY
 * trial at EVERY slot count to count as a hit.
 *
 * Usage: s602-fold-probe [--rings "0 3 2 1 5 7 6 4;..."] [--trials N]
 * (rings = per-channel ring lines from NCCL_DEBUG=INFO, semicolon-sep).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t e = (x); if (e != ncclSuccess) { \
    fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)

static const int kG = 8;

struct Instance {
    const char *name;
    int slots;
    uint64_t count;
    std::vector<std::vector<float>> in;   /* [rank][count] */
    std::vector<float> nccl_out;          /* reduced (identical on ranks; rank0) */
};

static uint32_t rng_state = 0x2433617u;
static uint32_t xorshift() {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}
/* Order-sensitive floats: random mantissa, exponent in [-12, 12]. */
static float rnd_val() {
    const uint32_t m = xorshift();
    const float frac = 0.5f + (float)(m & 0xffffff) / (float)0x2000000;
    const int ex = (int)(xorshift() % 25) - 12;
    const float sgn = (xorshift() & 1) ? 1.0f : -1.0f;
    return sgn * ldexpf(frac, ex);
}

int main(int argc, char **argv) {
    std::string rings_arg = "0 3 2 1 5 7 6 4";
    int trials = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rings") && i + 1 < argc) rings_arg = argv[++i];
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
    }
    /* parse ring list */
    std::vector<std::vector<int>> rings;
    {
        const char *p = rings_arg.c_str();
        std::vector<int> cur;
        while (*p) {
            if (*p == ';') {
                if (cur.size() == (size_t)kG) rings.push_back(cur);
                cur.clear(); ++p; continue;
            }
            if (*p == ' ' || *p == '\t') { ++p; continue; }
            cur.push_back((int)strtol(p, (char **)&p, 10));
        }
        if (cur.size() == (size_t)kG) rings.push_back(cur);
    }
    if (rings.empty()) { fprintf(stderr, "no rings parsed\n"); return 1; }
    printf("s602_fold_probe\trings\t%zu\ttrials\t%d\n", rings.size(), trials);
    for (size_t r = 0; r < rings.size(); ++r) {
        printf("  ring %zu:", r);
        for (int i = 0; i < kG; ++i) printf(" %d", rings[r][i]);
        printf("\n");
    }

    int ndev = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    if (ndev < kG) { fprintf(stderr, "need 8 GPUs, got %d\n", ndev); return 1; }
    int devs[kG]; for (int i = 0; i < kG; ++i) devs[i] = i;
    ncclComm_t comms[kG];
    CHECK_NCCL(ncclCommInitAll(comms, kG, devs));
    cudaStream_t streams[kG];
    for (int i = 0; i < kG; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    printf("comm init OK\n"); fflush(stdout);

    const int slot_cases[] = {1, 2, 4, 8, 16, 24, 32};
    std::vector<Instance> insts;

    /* device scratch: max count = 32*256 = 8192 floats; one buffer per rank
     * per op in the largest group (max+mix grouped = 2 buffers). */
    float *d_a[kG], *d_b[kG];
    for (int g = 0; g < kG; ++g) {
        CHECK_CUDA(cudaSetDevice(g));
        CHECK_CUDA(cudaMalloc(&d_a[g], 8192 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b[g], 8192 * sizeof(float)));
    }

    /* Run one grouped allreduce wave mirroring an engine call site.
     * kind: 0 = hc group {Max(count_a) + Sum(count_b)} (two ops per rank
     * in ONE group); 1 = single Sum(count_a); 2 = single Max(count_a). */
    auto run_wave = [&](int kind, uint64_t count_a, uint64_t count_b,
                        Instance *ia, Instance *ib) {
        for (int t = 0; t < trials; ++t) {
            std::vector<std::vector<float>> ha(kG), hb(kG);
            for (int g = 0; g < kG; ++g) {
                ha[g].resize(count_a);
                for (auto &v : ha[g]) v = rnd_val();
                if (count_b) {
                    hb[g].resize(count_b);
                    for (auto &v : hb[g]) v = rnd_val();
                }
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaMemcpy(d_a[g], ha[g].data(),
                                      count_a * sizeof(float),
                                      cudaMemcpyHostToDevice));
                if (count_b)
                    CHECK_CUDA(cudaMemcpy(d_b[g], hb[g].data(),
                                          count_b * sizeof(float),
                                          cudaMemcpyHostToDevice));
            }
            CHECK_NCCL(ncclGroupStart());
            for (int g = 0; g < kG; ++g) {
                CHECK_CUDA(cudaSetDevice(g));
                if (kind == 0) {
                    CHECK_NCCL(ncclAllReduce(d_a[g], d_a[g], count_a,
                                             ncclFloat, ncclMax, comms[g],
                                             streams[g]));
                    CHECK_NCCL(ncclAllReduce(d_b[g], d_b[g], count_b,
                                             ncclFloat, ncclSum, comms[g],
                                             streams[g]));
                } else if (kind == 1) {
                    CHECK_NCCL(ncclAllReduce(d_a[g], d_a[g], count_a,
                                             ncclFloat, ncclSum, comms[g],
                                             streams[g]));
                } else {
                    CHECK_NCCL(ncclAllReduce(d_a[g], d_a[g], count_a,
                                             ncclFloat, ncclMax, comms[g],
                                             streams[g]));
                }
            }
            CHECK_NCCL(ncclGroupEnd());
            for (int g = 0; g < kG; ++g) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaStreamSynchronize(streams[g]));
            }
            /* also verify the reduction is identical across ranks (it is an
             * ALLreduce) -- catches any probe bug. */
            std::vector<float> out0(count_a), outg(count_a);
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemcpy(out0.data(), d_a[0], count_a * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (int g = 1; g < kG; ++g) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaMemcpy(outg.data(), d_a[g],
                                      count_a * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                if (memcmp(out0.data(), outg.data(),
                           count_a * sizeof(float)) != 0) {
                    printf("RANK-DISAGREEMENT kind=%d count=%llu rank=%d\n",
                           kind, (unsigned long long)count_a, g);
                }
            }
            if (ia) {
                ia->in = ha;
                ia->nccl_out = out0;
                insts.push_back(*ia);
            }
            if (ib && count_b) {
                std::vector<float> outb(count_b);
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(outb.data(), d_b[0],
                                      count_b * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                ib->in = hb;
                ib->nccl_out = outb;
                insts.push_back(*ib);
            }
        }
    };

    for (int si = 0; si < (int)(sizeof(slot_cases) / sizeof(int)); ++si) {
        const int slots = slot_cases[si];
        /* engine call structure per layer:
         * 1. group { AR(max, slots, Max); AR(mix, slots*24, Sum) }
         * 2. group { AR(sumsq, slots, Sum) }
         * 3. group { AR(rmax, slots, Max) }          (router)
         * 4. group { AR(rsumsq, slots, Sum) }        (router)
         * 5. group { AR(logits, slots*256, Sum) }    (router) */
        Instance ia, ib;
        ia = {"hc_max(set-max sanity)", slots, (uint64_t)slots, {}, {}};
        ib = {"hc_mix", slots, (uint64_t)slots * 24, {}, {}};
        run_wave(0, (uint64_t)slots, (uint64_t)slots * 24, nullptr, &ib);
        ia = {"hc_sumsq", slots, (uint64_t)slots, {}, {}};
        run_wave(1, (uint64_t)slots, 0, &ia, nullptr);
        ia = {"r_sumsq", slots, (uint64_t)slots, {}, {}};
        run_wave(1, (uint64_t)slots, 0, &ia, nullptr);
        ia = {"r_logits", slots, (uint64_t)slots * 256, {}, {}};
        run_wave(1, (uint64_t)slots * 256, 0, &ia, nullptr);
        printf("collected slots=%d\n", slots); fflush(stdout);
    }

    /* ---------------- hypothesis search ---------------- */
    /* fold emulation for hypothesis (nc, minChunk, delta, rbase):
     * matches NCCL 2.19 ring AR chunk loop. chunkCap large (1<<20). */
    struct Hyp { int nc, minChunk, delta, rbase; };
    std::vector<Hyp> hyps;
    /* pow2 grid (LL/Simple-style) + multiples-of-30 grid (LL128 works in
     * 128B lines carrying 120B = 30 floats of data). */
    const int mcs[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                       24, 30, 48, 60, 96, 120, 192, 240, 384, 480, 960, 1920};
    for (int nc = 1; nc <= 8; ++nc)
        for (size_t mi = 0; mi < sizeof(mcs) / sizeof(int); ++mi)
            for (int delta = 0; delta < 8; ++delta)
                for (int rbase = 0; rbase < (int)rings.size(); ++rbase)
                    hyps.push_back({nc, mcs[mi], delta, rbase});

    auto emulate = [&](const Instance &I, const Hyp &h,
                       std::vector<float> *out) -> void {
        const uint64_t C = I.count;
        out->assign(C, 0.0f);
        const uint64_t loop = (uint64_t)h.nc * kG;
        uint64_t gridOffset = 0;
        while (gridOffset < C) {
            const uint64_t remaining = C - gridOffset;
            uint64_t realChunk =
                ((remaining + loop * h.minChunk - 1) / (loop * h.minChunk)) *
                h.minChunk;
            if (realChunk > (1ull << 20)) realChunk = (1ull << 20);
            const uint64_t span = loop * realChunk;
            for (uint64_t e = gridOffset;
                 e < gridOffset + span && e < C; ++e) {
                const uint64_t off = e - gridOffset;
                const int bid = (int)(off / ((uint64_t)kG * realChunk));
                const int c = (int)((off % ((uint64_t)kG * realChunk)) /
                                    realChunk);
                const std::vector<int> &ring =
                    rings[(h.rbase + bid) % rings.size()];
                const int start = (c + h.delta) & 7;
                float acc = I.in[(size_t)ring[start]][e];
                for (int k = 1; k < kG; ++k) {
                    acc += I.in[(size_t)ring[(start + k) & 7]][e];
                }
                (*out)[e] = acc;
            }
            gridOffset += span;
        }
    };

    /* group instances by name (across slots+trials); a hypothesis must
     * explain all instances of the name. Also report per-(name,slots) hits
     * in case NCCL's channel count is size-dependent. */
    std::vector<std::string> names = {"hc_mix", "hc_sumsq", "r_sumsq",
                                      "r_logits"};
    std::vector<std::vector<int>> global_hits(names.size());
    for (size_t ni = 0; ni < names.size(); ++ni) {
        /* per-(name,slots) first */
        for (int si = 0; si < (int)(sizeof(slot_cases) / sizeof(int)); ++si) {
            const int slots = slot_cases[si];
            int nhits = 0;
            int first_hit = -1;
            for (size_t hi = 0; hi < hyps.size(); ++hi) {
                bool all_ok = true;
                bool any = false;
                for (const Instance &I : insts) {
                    if (names[ni] != I.name || I.slots != slots) continue;
                    any = true;
                    std::vector<float> emu;
                    emulate(I, hyps[hi], &emu);
                    if (memcmp(emu.data(), I.nccl_out.data(),
                               I.count * sizeof(float)) != 0) {
                        all_ok = false;
                        break;
                    }
                }
                if (any && all_ok) {
                    ++nhits;
                    if (first_hit < 0) first_hit = (int)hi;
                }
            }
            if (first_hit >= 0) {
                const Hyp &h = hyps[first_hit];
                printf("  per-shape %-10s slots=%-2d hits=%d first: nc=%d "
                       "minChunk=%d delta=%d rbase=%d\n",
                       names[ni].c_str(), slots, nhits, h.nc, h.minChunk,
                       h.delta, h.rbase);
            } else {
                printf("  per-shape %-10s slots=%-2d hits=0 NO-MATCH\n",
                       names[ni].c_str(), slots);
            }
            fflush(stdout);
        }
        std::vector<int> hits;
        for (size_t hi = 0; hi < hyps.size(); ++hi) {
            bool all_ok = true;
            for (const Instance &I : insts) {
                if (names[ni] != I.name) continue;
                std::vector<float> emu;
                emulate(I, hyps[hi], &emu);
                if (memcmp(emu.data(), I.nccl_out.data(),
                           I.count * sizeof(float)) != 0) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) hits.push_back((int)hi);
        }
        global_hits[ni] = hits;
        printf("collective %-10s matching hypotheses: %zu\n",
               names[ni].c_str(), hits.size());
        for (size_t k = 0; k < hits.size() && k < 12; ++k) {
            const Hyp &h = hyps[hits[k]];
            printf("  HIT nc=%d minChunk=%d delta=%d rbase=%d\n",
                   h.nc, h.minChunk, h.delta, h.rbase);
        }
        fflush(stdout);
    }
    /* intersection across all four collectives */
    printf("---- intersection over all four sum collectives ----\n");
    int inter = 0;
    for (int hi : global_hits[0]) {
        bool in_all = true;
        for (size_t ni = 1; ni < names.size(); ++ni) {
            bool found = false;
            for (int hj : global_hits[ni]) if (hj == hi) { found = true; break; }
            if (!found) { in_all = false; break; }
        }
        if (in_all) {
            const Hyp &h = hyps[hi];
            printf("GLOBAL-HIT nc=%d minChunk=%d delta=%d rbase=%d\n",
                   h.nc, h.minChunk, h.delta, h.rbase);
            ++inter;
        }
    }
    printf("s602_fold_probe_done\tglobal_hits\t%d\t%s\n", inter,
           inter > 0 ? "PASS" : "NO-MATCH");
    return inter > 0 ? 0 : 2;
}
