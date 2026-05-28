#include "engine/mtp_sidecar.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DS4_V100_MTP_DEFAULT_UPLOAD_CHUNK = 8 * 1024 * 1024,
    DS4_V100_MTP_SPOT_BYTES = 32,
};

struct ds4_mtp_sidecar {
    ds4_mtp_sidecar_info info;
    int fd;
    const unsigned char *map;
    uint64_t size;
    ds4_gpu_arena *arena;
    uint64_t uploaded_tensors;
    uint64_t uploaded_bytes;
    uint64_t spot_checks;
    int gpu;
};

static int mtp_error(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "MTP sidecar error");
    return 1;
}

static int mtp_errorf(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

void ds4_mtp_sidecar_options_init(ds4_mtp_sidecar_options *opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
    opts->gpu = 7;
    opts->upload_chunk_bytes = DS4_V100_MTP_DEFAULT_UPLOAD_CHUNK;
    opts->require_device_arena = true;
}

static uint64_t file_size_or_error(int fd, const char *path, char *err, size_t errlen) {
    struct stat st;
    if (fstat(fd, &st) != 0) {
        mtp_errorf(err, errlen, "cannot stat MTP sidecar %s: %s", path, strerror(errno));
        return 0;
    }
    if (st.st_size <= 0) {
        mtp_errorf(err, errlen, "invalid MTP sidecar size for %s", path);
        return 0;
    }
    return (uint64_t)st.st_size;
}

static int map_sidecar(ds4_mtp_sidecar *s,
                       const char *path,
                       char *err,
                       size_t errlen) {
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) {
        return mtp_errorf(err, errlen, "cannot open MTP sidecar %s: %s", path, strerror(errno));
    }
    s->size = file_size_or_error(s->fd, path, err, errlen);
    if (!s->size) return 1;
    s->map = (const unsigned char *)mmap(NULL,
                                         (size_t)s->size,
                                         PROT_READ,
                                         MAP_PRIVATE,
                                         s->fd,
                                         0);
    if (s->map == MAP_FAILED) {
        s->map = NULL;
        return mtp_errorf(err, errlen, "cannot mmap MTP sidecar %s: %s", path, strerror(errno));
    }
    return 0;
}

static int validate_ranges(const ds4_mtp_sidecar *s, char *err, size_t errlen) {
    if (s->size != s->info.file_bytes) {
        return mtp_errorf(err,
                          errlen,
                          "MTP sidecar mapped size %" PRIu64 " != inspected size %" PRIu64,
                          s->size,
                          s->info.file_bytes);
    }
    for (uint32_t i = 0; i < DS4_MTP_SIDECAR_TENSOR_COUNT; i++) {
        const ds4_mtp_sidecar_tensor_info *t = &s->info.tensors[i];
        if (t->source_offset > s->size ||
            t->byte_length > s->size - t->source_offset) {
            return mtp_errorf(err, errlen, "MTP tensor %s is outside mapped sidecar", t->name);
        }
        if (t->resident_offset > s->info.resident_bytes ||
            t->byte_length > s->info.resident_bytes - t->resident_offset) {
            return mtp_errorf(err, errlen, "MTP tensor %s is outside resident arena", t->name);
        }
    }
    return 0;
}

static int upload_tensor(ds4_mtp_sidecar *s,
                         const ds4_mtp_sidecar_tensor_info *t,
                         unsigned char *chunk,
                         uint64_t chunk_bytes,
                         char *err,
                         size_t errlen) {
    uint64_t done = 0;
    while (done < t->byte_length) {
        uint64_t n = t->byte_length - done;
        if (n > chunk_bytes) n = chunk_bytes;
        memcpy(chunk, s->map + t->source_offset + done, (size_t)n);
        if (ds4_gpu_arena_upload(s->arena, t->resident_offset + done, chunk, n) != 0) {
            return mtp_errorf(err, errlen, "MTP upload failed for %s on gpu%d", t->name, s->gpu);
        }
        done += n;
    }
    s->uploaded_tensors++;
    s->uploaded_bytes += t->byte_length;
    return 0;
}

static int spot_check_range(ds4_mtp_sidecar *s,
                            const ds4_mtp_sidecar_tensor_info *t,
                            uint64_t rel,
                            uint64_t bytes,
                            char *err,
                            size_t errlen) {
    unsigned char actual[DS4_V100_MTP_SPOT_BYTES];
    if (bytes > sizeof(actual)) bytes = sizeof(actual);
    if (ds4_gpu_arena_read(s->arena, t->resident_offset + rel, actual, bytes) != 0) {
        return mtp_errorf(err, errlen, "MTP readback failed for %s", t->name);
    }
    if (memcmp(actual, s->map + t->source_offset + rel, (size_t)bytes) != 0) {
        return mtp_errorf(err, errlen, "MTP resident bytes mismatch for %s", t->name);
    }
    s->spot_checks++;
    return 0;
}

static int spot_check_tensor(ds4_mtp_sidecar *s,
                             const ds4_mtp_sidecar_tensor_info *t,
                             char *err,
                             size_t errlen) {
    if (!t->byte_length) return 0;
    uint64_t n = t->byte_length < DS4_V100_MTP_SPOT_BYTES
        ? t->byte_length
        : DS4_V100_MTP_SPOT_BYTES;
    if (spot_check_range(s, t, 0, n, err, errlen)) return 1;
    if (t->byte_length > n) {
        if (spot_check_range(s, t, t->byte_length - n, n, err, errlen)) return 1;
    }
    return 0;
}

int ds4_mtp_sidecar_open(ds4_mtp_sidecar **out,
                              const ds4_mtp_sidecar_options *opts,
                              FILE *report,
                              char *err,
                              size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!out) return mtp_error(err, errlen, "missing MTP sidecar output");
    *out = NULL;
    if (!opts || !opts->mtp_path || !opts->mtp_path[0]) {
        return mtp_error(err, errlen, "missing MTP sidecar path");
    }
    if (opts->gpu < 0) return mtp_error(err, errlen, "invalid MTP sidecar gpu");

    ds4_mtp_sidecar *s = (ds4_mtp_sidecar *)calloc(1, sizeof(*s));
    if (!s) return mtp_error(err, errlen, "failed to allocate MTP sidecar");
    s->fd = -1;
    s->gpu = opts->gpu;

    int rc = 1;
    unsigned char *chunk = NULL;
    const uint64_t chunk_bytes = opts->upload_chunk_bytes
        ? opts->upload_chunk_bytes
        : DS4_V100_MTP_DEFAULT_UPLOAD_CHUNK;

    if (ds4_mtp_sidecar_inspect(opts->mtp_path, &s->info, report, err, errlen) != 0 ||
        map_sidecar(s, opts->mtp_path, err, errlen) ||
        validate_ranges(s, err, errlen)) {
        goto done;
    }

    if (ds4_gpu_arena_open(&s->arena, s->gpu, s->info.resident_bytes) != 0) {
        mtp_errorf(err,
                   errlen,
                   "failed to allocate MTP resident arena on gpu%d for %" PRIu64 " bytes",
                   s->gpu,
                   s->info.resident_bytes);
        goto done;
    }
    if (opts->require_device_arena && !ds4_gpu_arena_is_device_memory(s->arena)) {
        mtp_errorf(err, errlen, "MTP resident arena on gpu%d is not device memory", s->gpu);
        goto done;
    }

    chunk = (unsigned char *)malloc((size_t)chunk_bytes);
    if (!chunk) {
        mtp_error(err, errlen, "failed to allocate MTP upload chunk");
        goto done;
    }

    for (uint32_t i = 0; i < DS4_MTP_SIDECAR_TENSOR_COUNT; i++) {
        if (upload_tensor(s, &s->info.tensors[i], chunk, chunk_bytes, err, errlen)) goto done;
    }
    for (uint32_t i = 0; i < DS4_MTP_SIDECAR_TENSOR_COUNT; i++) {
        if (spot_check_tensor(s, &s->info.tensors[i], err, errlen)) goto done;
    }

    if (report) {
        fprintf(report, "mtp_runtime\tgpu\t%d\n", s->gpu);
        fprintf(report, "mtp_runtime\tarena_kind\t%s\n", ds4_gpu_arena_memory_kind(s->arena));
        fprintf(report, "mtp_runtime\tarena_bytes\t%" PRIu64 "\n", ds4_gpu_arena_bytes(s->arena));
        fprintf(report, "mtp_runtime\tuploaded_tensors\t%" PRIu64 "\n", s->uploaded_tensors);
        fprintf(report, "mtp_runtime\tuploaded_bytes\t%" PRIu64 "\n", s->uploaded_bytes);
        fprintf(report, "mtp_runtime\tspot_checks\t%" PRIu64 "\n", s->spot_checks);
        fprintf(report,
                "mtp_runtime\tfree_after_upload_bytes\t%" PRIu64 "\n",
                ds4_gpu_arena_free_after_upload_bytes(s->arena));
        fprintf(report, "mtp_runtime\tPASS\tresident_sidecar=1\n");
    }

    *out = s;
    s = NULL;
    rc = 0;

done:
    free(chunk);
    ds4_mtp_sidecar_close(s);
    return rc;
}

void ds4_mtp_sidecar_close(ds4_mtp_sidecar *sidecar) {
    if (!sidecar) return;
    ds4_gpu_arena_close(sidecar->arena);
    if (sidecar->map) munmap((void *)sidecar->map, (size_t)sidecar->size);
    if (sidecar->fd >= 0) close(sidecar->fd);
    free(sidecar);
}

const ds4_mtp_sidecar_info *ds4_mtp_sidecar_get_info(
        const ds4_mtp_sidecar *sidecar) {
    return sidecar ? &sidecar->info : NULL;
}

const ds4_mtp_sidecar_tensor_info *ds4_mtp_sidecar_tensor(
        const ds4_mtp_sidecar *sidecar,
        const char *name) {
    if (!sidecar || !name) return NULL;
    for (uint32_t i = 0; i < DS4_MTP_SIDECAR_TENSOR_COUNT; i++) {
        const ds4_mtp_sidecar_tensor_info *t = &sidecar->info.tensors[i];
        if (!strcmp(t->name, name)) return t;
    }
    return NULL;
}

const void *ds4_mtp_sidecar_map(const ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->map : NULL;
}

uint64_t ds4_mtp_sidecar_size(const ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->size : 0;
}

int ds4_mtp_sidecar_q8_0_view(
        const ds4_mtp_sidecar *sidecar,
        const char *name,
        ds4_gpu_source_row_view *out,
        char *err,
        size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!sidecar || !name || !out) {
        return mtp_error(err, errlen, "missing MTP Q8_0 view argument");
    }
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_mtp_sidecar_tensor(sidecar, name);
    if (!t) {
        return mtp_errorf(err, errlen, "missing MTP tensor %s", name);
    }
    if (strcmp(t->dtype, "q8_0") != 0 || t->n_dims != 2 ||
        t->shape[0] == 0 || t->shape[1] == 0 ||
        t->shape[0] > UINT32_MAX || t->shape[1] > UINT32_MAX) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s is not a supported 2D Q8_0 tensor",
                          name);
    }

    const uint64_t blocks = (t->shape[0] + 31ull) / 32ull;
    if (blocks > UINT64_MAX / 34ull) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q8_0 row stride overflows", name);
    }
    const uint64_t row_stride = blocks * 34ull;
    if (t->shape[1] > UINT64_MAX / row_stride) {
        return mtp_errorf(err, errlen, "MTP tensor %s byte length overflows", name);
    }
    const uint64_t expected_bytes = t->shape[1] * row_stride;
    if (expected_bytes != t->byte_length) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s byte length %" PRIu64
                          " != expected Q8_0 bytes %" PRIu64,
                          name,
                          t->byte_length,
                          expected_bytes);
    }
    if (t->source_offset > sidecar->size ||
        t->byte_length > sidecar->size - t->source_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s source range is invalid", name);
    }
    if (t->resident_offset > sidecar->info.resident_bytes ||
        t->byte_length > sidecar->info.resident_bytes - t->resident_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s resident range is invalid", name);
    }

    memset(out, 0, sizeof(*out));
    out->arena_offset = t->resident_offset;
    out->byte_length = t->byte_length;
    out->rows = (uint32_t)t->shape[1];
    out->cols = (uint32_t)t->shape[0];
    out->row_stride_bytes = (uint32_t)row_stride;
    return 0;
}

int ds4_mtp_sidecar_f32_vector_view(
        const ds4_mtp_sidecar *sidecar,
        const char *name,
        ds4_gpu_source_row_view *out,
        char *err,
        size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!sidecar || !name || !out) {
        return mtp_error(err, errlen, "missing MTP F32 view argument");
    }
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_mtp_sidecar_tensor(sidecar, name);
    if (!t) {
        return mtp_errorf(err, errlen, "missing MTP tensor %s", name);
    }
    if (strcmp(t->dtype, "f32") != 0 || t->n_dims != 1 ||
        t->shape[0] == 0 || t->shape[0] > UINT32_MAX ||
        t->shape[0] > UINT32_MAX / sizeof(float)) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s is not a supported 1D F32 tensor",
                          name);
    }

    const uint64_t expected_bytes = t->shape[0] * sizeof(float);
    if (expected_bytes != t->byte_length) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s byte length %" PRIu64
                          " != expected F32 bytes %" PRIu64,
                          name,
                          t->byte_length,
                          expected_bytes);
    }
    if ((t->source_offset % sizeof(float)) != 0 ||
        (t->resident_offset % sizeof(float)) != 0) {
        return mtp_errorf(err, errlen, "MTP tensor %s F32 offset is unaligned", name);
    }
    if (t->source_offset > sidecar->size ||
        t->byte_length > sidecar->size - t->source_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s source range is invalid", name);
    }
    if (t->resident_offset > sidecar->info.resident_bytes ||
        t->byte_length > sidecar->info.resident_bytes - t->resident_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s resident range is invalid", name);
    }

    memset(out, 0, sizeof(*out));
    out->arena_offset = t->resident_offset;
    out->byte_length = t->byte_length;
    out->rows = 1;
    out->cols = (uint32_t)t->shape[0];
    out->row_stride_bytes = (uint32_t)expected_bytes;
    return 0;
}

int ds4_mtp_sidecar_f32_matrix_view(
        const ds4_mtp_sidecar *sidecar,
        const char *name,
        ds4_gpu_source_row_view *out,
        char *err,
        size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!sidecar || !name || !out) {
        return mtp_error(err, errlen, "missing MTP F32 matrix view argument");
    }
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_mtp_sidecar_tensor(sidecar, name);
    if (!t) {
        return mtp_errorf(err, errlen, "missing MTP tensor %s", name);
    }
    if (strcmp(t->dtype, "f32") != 0 || t->n_dims != 2 ||
        t->shape[0] == 0 || t->shape[1] == 0 ||
        t->shape[0] > UINT32_MAX || t->shape[1] > UINT32_MAX ||
        t->shape[0] > UINT32_MAX / sizeof(float)) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s is not a supported 2D F32 tensor",
                          name);
    }

    const uint64_t row_stride = t->shape[0] * sizeof(float);
    if (t->shape[1] > UINT64_MAX / row_stride) {
        return mtp_errorf(err, errlen, "MTP tensor %s F32 byte length overflows", name);
    }
    const uint64_t expected_bytes = t->shape[1] * row_stride;
    if (expected_bytes != t->byte_length) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s byte length %" PRIu64
                          " != expected F32 matrix bytes %" PRIu64,
                          name,
                          t->byte_length,
                          expected_bytes);
    }
    if ((t->source_offset % sizeof(float)) != 0 ||
        (t->resident_offset % sizeof(float)) != 0) {
        return mtp_errorf(err, errlen, "MTP tensor %s F32 offset is unaligned", name);
    }
    if (t->source_offset > sidecar->size ||
        t->byte_length > sidecar->size - t->source_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s source range is invalid", name);
    }
    if (t->resident_offset > sidecar->info.resident_bytes ||
        t->byte_length > sidecar->info.resident_bytes - t->resident_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s resident range is invalid", name);
    }

    memset(out, 0, sizeof(*out));
    out->arena_offset = t->resident_offset;
    out->byte_length = t->byte_length;
    out->rows = (uint32_t)t->shape[1];
    out->cols = (uint32_t)t->shape[0];
    out->row_stride_bytes = (uint32_t)row_stride;
    return 0;
}

int ds4_mtp_sidecar_q4_k_expert_view(
        const ds4_mtp_sidecar *sidecar,
        const char *name,
        ds4_gpu_q4_k_expert_view *out,
        char *err,
        size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!sidecar || !name || !out) {
        return mtp_error(err, errlen, "missing MTP Q4_K expert view argument");
    }
    const ds4_mtp_sidecar_tensor_info *t =
        ds4_mtp_sidecar_tensor(sidecar, name);
    if (!t) {
        return mtp_errorf(err, errlen, "missing MTP tensor %s", name);
    }
    if (strcmp(t->dtype, "q4_k") != 0 || t->n_dims != 3 ||
        t->shape[0] == 0 || t->shape[1] == 0 || t->shape[2] == 0 ||
        t->shape[0] > UINT32_MAX || t->shape[1] > UINT32_MAX ||
        t->shape[2] > UINT32_MAX) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s is not a supported 3D Q4_K expert tensor",
                          name);
    }
    if ((t->shape[0] % 256ull) != 0) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q4_K cols are not 256-aligned", name);
    }

    const uint64_t blocks = t->shape[0] / 256ull;
    if (blocks > UINT64_MAX / 144ull) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q4_K row stride overflows", name);
    }
    const uint64_t row_stride = blocks * 144ull;
    if (t->shape[1] > UINT64_MAX / row_stride) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q4_K expert stride overflows", name);
    }
    const uint64_t expert_stride = t->shape[1] * row_stride;
    if (t->shape[2] > UINT64_MAX / expert_stride) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q4_K byte length overflows", name);
    }
    const uint64_t expected_bytes = t->shape[2] * expert_stride;
    if (expected_bytes != t->byte_length) {
        return mtp_errorf(err,
                          errlen,
                          "MTP tensor %s byte length %" PRIu64
                          " != expected Q4_K bytes %" PRIu64,
                          name,
                          t->byte_length,
                          expected_bytes);
    }
    if (row_stride > UINT32_MAX) {
        return mtp_errorf(err, errlen, "MTP tensor %s Q4_K row stride exceeds uint32", name);
    }
    if (t->source_offset > sidecar->size ||
        t->byte_length > sidecar->size - t->source_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s source range is invalid", name);
    }
    if (t->resident_offset > sidecar->info.resident_bytes ||
        t->byte_length > sidecar->info.resident_bytes - t->resident_offset) {
        return mtp_errorf(err, errlen, "MTP tensor %s resident range is invalid", name);
    }

    memset(out, 0, sizeof(*out));
    out->arena_offset = t->resident_offset;
    out->byte_length = t->byte_length;
    out->experts = (uint32_t)t->shape[2];
    out->rows = (uint32_t)t->shape[1];
    out->cols = (uint32_t)t->shape[0];
    out->row_stride_bytes = (uint32_t)row_stride;
    out->expert_stride_bytes = expert_stride;
    return 0;
}

ds4_gpu_arena *ds4_mtp_sidecar_arena(ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->arena : NULL;
}

uint64_t ds4_mtp_sidecar_uploaded_bytes(const ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->uploaded_bytes : 0;
}

uint64_t ds4_mtp_sidecar_spot_checks(const ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->spot_checks : 0;
}

int ds4_mtp_sidecar_gpu(const ds4_mtp_sidecar *sidecar) {
    return sidecar ? sidecar->gpu : -1;
}
