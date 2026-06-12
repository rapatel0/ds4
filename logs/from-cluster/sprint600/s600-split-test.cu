// S600: does ncclCommSplit(splitShare=1) avoid the per-rank 4MB proxy shm pools?
#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>
#define K 8
int main() {
    int devs[K]; for (int i = 0; i < K; ++i) devs[i] = i;
    ncclComm_t a[K], b[K];
    printf("init A...\n"); fflush(stdout);
    ncclResult_t rc = ncclCommInitAll(a, K, devs);
    printf("A rc=%d\n", (int)rc); fflush(stdout);
    if (rc != ncclSuccess) return 1;
    if (system("df -h /dev/shm | tail -1")) {}
    printf("split B (splitShare=1)...\n"); fflush(stdout);
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.splitShare = 1;
    rc = ncclGroupStart();
    for (int i = 0; i < K; ++i) {
        cudaSetDevice(devs[i]);
        ncclResult_t r2 = ncclCommSplit(a[i], 0, i, &b[i], &cfg);
        if (r2 != ncclSuccess) { printf("split rank %d rc=%d (%s)\n", i, (int)r2, ncclGetErrorString(r2)); rc = r2; }
    }
    {
        ncclResult_t r3 = ncclGroupEnd();
        if (r3 != ncclSuccess) rc = r3;
    }
    printf("B rc=%d (%s)\n", (int)rc, ncclGetErrorString(rc)); fflush(stdout);
    if (system("df -h /dev/shm | tail -1")) {}
    if (rc != ncclSuccess) return 2;
    // quick functional check: tiny allreduce on B
    float *buf[K];
    cudaStream_t st[K];
    for (int i = 0; i < K; ++i) { cudaSetDevice(devs[i]); cudaMalloc(&buf[i], 128 * 4); cudaMemset(buf[i], 0, 128 * 4); cudaStreamCreate(&st[i]); }
    ncclGroupStart();
    for (int i = 0; i < K; ++i) { cudaSetDevice(devs[i]); ncclAllReduce(buf[i], buf[i], 128, ncclFloat, ncclSum, b[i], st[i]); }
    ncclGroupEnd();
    for (int i = 0; i < K; ++i) { cudaSetDevice(devs[i]); cudaStreamSynchronize(st[i]); }
    printf("allreduce on B OK\n");
    return 0;
}
