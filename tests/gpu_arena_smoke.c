#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char in[32];
    unsigned char out[32];
    for (size_t i = 0; i < sizeof(in); i++) in[i] = (unsigned char)(i + 1);
    memset(out, 0, sizeof(out));

    ds4_gpu_arena *arena = NULL;
    if (ds4_gpu_arena_open(&arena, 0, sizeof(in)) != 0) {
        fprintf(stderr, "arena open failed\n");
        return 1;
    }
    if (ds4_gpu_arena_upload(arena, 0, in, sizeof(in)) != 0) {
        fprintf(stderr, "arena upload failed\n");
        return 1;
    }
    if (ds4_gpu_arena_read(arena, 0, out, sizeof(out)) != 0) {
        fprintf(stderr, "arena read failed\n");
        return 1;
    }
    if (memcmp(in, out, sizeof(in)) != 0) {
        fprintf(stderr, "arena readback mismatch\n");
        return 1;
    }
    ds4_gpu_arena_print_memory_report(stdout, &arena, 1);
    ds4_gpu_print_topology_report(stdout);
    if (ds4_gpu_arena_upload(arena, 31, in, 2) == 0) {
        fprintf(stderr, "arena accepted out-of-range upload\n");
        return 1;
    }
    ds4_gpu_arena_close(arena);
    puts("gpu_arena_smoke: ok");
    return 0;
}
