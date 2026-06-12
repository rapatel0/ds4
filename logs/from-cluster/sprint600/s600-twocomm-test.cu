// S600: can two 8-rank single-process NCCL comms coexist under a 64MB /dev/shm?
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
    printf("A rc=%d (%s)\n", (int)rc, ncclGetErrorString(rc)); fflush(stdout);
    if (rc != ncclSuccess) return 1;
    system("df -h /dev/shm | tail -1; ls /dev/shm | wc -l");
    printf("init B...\n"); fflush(stdout);
    rc = ncclCommInitAll(b, K, devs);
    printf("B rc=%d (%s)\n", (int)rc, ncclGetErrorString(rc)); fflush(stdout);
    system("df -h /dev/shm | tail -1; ls /dev/shm | wc -l");
    return rc == ncclSuccess ? 0 : 2;
}
