#include "ds4_pack.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        exit(1);
    }
    fputs(text, fp);
    fclose(fp);
}

static char *temp_path(const char *suffix) {
    char tmpl[256];
    snprintf(tmpl, sizeof(tmpl), "/tmp/ds4-pack-smoke-%ld-%s", (long)getpid(), suffix);
    size_t n = strlen(tmpl);
    char *out = (char *)malloc(n + 1);
    if (!out) exit(1);
    memcpy(out, tmpl, n + 1);
    return out;
}

static const char *header =
    "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
    "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
    "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum\n";

static void expect_open_ok(const char *path) {
    char err[256];
    ds4_pack *pack = NULL;
    if (ds4_pack_open(&pack, path, err, sizeof(err)) != 0) {
        fprintf(stderr, "expected open ok: %s\n", err);
        exit(1);
    }
    if (ds4_pack_count(pack) != 3) {
        fprintf(stderr, "unexpected row count\n");
        exit(1);
    }
    if (ds4_pack_payload_bytes(pack, 0) != 48 ||
        ds4_pack_payload_bytes(pack, 1) != 16 ||
        ds4_pack_arena_bytes(pack, 0) != 48 ||
        ds4_pack_tensor_count(pack, 0) != 2) {
        fprintf(stderr, "unexpected aggregate bytes/count\n");
        exit(1);
    }

    ds4_pack_source_tensor source[] = {
        {"tok", 3, "bf16", "[4x4]", 0, 32},
        {"ctl", 3, "f32", "[4]", 32, 16},
        {"lyr", 3, "f8_e4m3_b128", "[16x8]", 48, 16},
    };
    ds4_pack_reconcile_summary summary;
    FILE *sink = tmpfile();
    if (ds4_pack_reconcile(pack, source, 3, 128, 2, sink, &summary, err, sizeof(err)) != 0) {
        fprintf(stderr, "expected reconcile ok: %s\n", err);
        exit(1);
    }
    fclose(sink);
    if (summary.ok_rows != 3 || summary.failed_rows != 0) {
        fprintf(stderr, "unexpected reconcile summary\n");
        exit(1);
    }
    ds4_pack_close(pack);
}

static void expect_open_fail(const char *path) {
    char err[256];
    ds4_pack *pack = NULL;
    if (ds4_pack_open(&pack, path, err, sizeof(err)) == 0) {
        fprintf(stderr, "expected open failure\n");
        ds4_pack_close(pack);
        exit(1);
    }
}

static void expect_reconcile_fail(const char *path,
                                  const ds4_pack_source_tensor *source,
                                  size_t n_source) {
    char err[256];
    ds4_pack *pack = NULL;
    if (ds4_pack_open(&pack, path, err, sizeof(err)) != 0) {
        fprintf(stderr, "open before reconcile failure failed: %s\n", err);
        exit(1);
    }
    FILE *sink = tmpfile();
    if (ds4_pack_reconcile(pack, source, n_source, 128, 2, sink, NULL, err, sizeof(err)) == 0) {
        fprintf(stderr, "expected reconcile failure\n");
        exit(1);
    }
    fclose(sink);
    ds4_pack_close(pack);
}

int main(void) {
    char *ok = temp_path("ok.tsv");
    char *dup = temp_path("dup.tsv");
    char *bad_gpu = temp_path("bad-gpu.tsv");

    write_file(ok,
        "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
        "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
        "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum\n"
        "tok\ttok\tbf16\t[4x4]\tsource_bf16\t0\t-1\tembed\t0\t32\tgpu0.weights\t0\t-1\tpending\n"
        "ctl\tctl\tf32\t[4]\tsource_f32_control\t0\t0\tcontrol\t32\t16\tgpu0.weights\t32\t-1\tpending\n"
        "lyr\tlyr\tf8_e4m3_b128\t[16x8]\tsource_f8\t1\t1\tfp8\t48\t16\tgpu1.weights\t0\t-1\tpending\n");
    expect_open_ok(ok);

    write_file(dup,
        "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
        "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
        "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum\n"
        "tok\ttok\tbf16\t[4x4]\tsource_bf16\t0\t-1\tembed\t0\t32\tgpu0.weights\t0\t-1\tpending\n"
        "tok\tother\tbf16\t[4x4]\tsource_bf16\t0\t-1\tembed\t32\t32\tgpu0.weights\t32\t-1\tpending\n");
    expect_open_fail(dup);

    write_file(bad_gpu,
        "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
        "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
        "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum\n"
        "tok\ttok\tbf16\t[4x4]\tsource_bf16\t3\t-1\tembed\t0\t32\tgpu3.weights\t0\t-1\tpending\n");
    ds4_pack_source_tensor gpu_source[] = {
        {"tok", 3, "bf16", "[4x4]", 0, 32},
    };
    expect_reconcile_fail(bad_gpu, gpu_source, 1);

    ds4_pack_source_tensor dtype_bad[] = {
        {"tok", 3, "f32", "[4x4]", 0, 32},
        {"ctl", 3, "f32", "[4]", 32, 16},
        {"lyr", 3, "f8_e4m3_b128", "[16x8]", 48, 16},
    };
    expect_reconcile_fail(ok, dtype_bad, 3);

    ds4_pack_source_tensor shape_bad[] = {
        {"tok", 3, "bf16", "[8x2]", 0, 32},
        {"ctl", 3, "f32", "[4]", 32, 16},
        {"lyr", 3, "f8_e4m3_b128", "[16x8]", 48, 16},
    };
    expect_reconcile_fail(ok, shape_bad, 3);

    ds4_pack_source_tensor len_bad[] = {
        {"tok", 3, "bf16", "[4x4]", 0, 16},
        {"ctl", 3, "f32", "[4]", 32, 16},
        {"lyr", 3, "f8_e4m3_b128", "[16x8]", 48, 16},
    };
    expect_reconcile_fail(ok, len_bad, 3);

    unlink(ok);
    unlink(dup);
    unlink(bad_gpu);
    free(ok);
    free(dup);
    free(bad_gpu);
    puts("pack_index_smoke: ok");
    (void)header;
    return 0;
}
