#include "ds4_turbomind_pack.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DS4_TM_PACK_COLS = 25
};

static const char *ds4_tm_pack_header =
    "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
    "runtime_layout\towning_gpu\tlayer_id\tkernel_family\t"
    "n\tk\texperts_packed\texperts_total\tweight_bytes_per_expert\t"
    "scale_bytes_per_expert\tk_pack\tweight_stride\tscale_stride\t"
    "sidecar_file\tweight_offset\tscale_offset\tsource_shard_file\t"
    "source_shard_offset\tsource_byte_length\tsource_checksum\t"
    "tm_abi_version";

struct ds4_tm_pack {
    ds4_tm_pack_entry *entries;
    uint64_t n_entries;
    uint64_t cap_entries;
    char *path;
};

static int tm_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static char *tm_strdup_len(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *tm_strdup(const char *s) {
    return tm_strdup_len(s, strlen(s));
}

static void tm_entry_free(ds4_tm_pack_entry *e) {
    if (!e) return;
    free((char *)e->semantic_tensor_id);
    free((char *)e->source_name);
    free((char *)e->source_dtype);
    free((char *)e->source_shape);
    free((char *)e->runtime_layout);
    free((char *)e->kernel_family);
    free((char *)e->sidecar_file);
    free((char *)e->source_shard_file);
    free((char *)e->source_checksum);
}

void ds4_tm_pack_close(ds4_tm_pack *pack) {
    if (!pack) return;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        tm_entry_free(&pack->entries[i]);
    }
    free(pack->entries);
    free(pack->path);
    free(pack);
}

static int read_small_file(const char *path, char **out, size_t *out_len,
                           char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return tm_error(err, errlen, "cannot open %s: %s", path, strerror(errno));
    if (fseek(fp, 0, SEEK_END) != 0) {
        int rc = tm_error(err, errlen, "cannot seek %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    long n = ftell(fp);
    if (n < 0) {
        int rc = tm_error(err, errlen, "cannot tell %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    if ((unsigned long)n > 64ul * 1024ul * 1024ul) {
        int rc = tm_error(err, errlen, "TurboMind pack index is unexpectedly large: %s", path);
        fclose(fp);
        return rc;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        int rc = tm_error(err, errlen, "cannot rewind %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        return tm_error(err, errlen, "out of memory reading %s", path);
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    if (got != (size_t)n) {
        int rc = tm_error(err, errlen, "short read from %s", path);
        free(buf);
        fclose(fp);
        return rc;
    }
    fclose(fp);
    buf[n] = '\0';
    *out = buf;
    *out_len = (size_t)n;
    return 0;
}

static int parse_u64_field(const char *s, uint64_t *out) {
    if (!s || !*s || *s == '-') return 1;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end != '\0') return 1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_i64_field(const char *s, int64_t *out) {
    if (!s || !*s) return 1;
    errno = 0;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (errno || !end || *end != '\0') return 1;
    *out = (int64_t)v;
    return 0;
}

static int parse_i32_field(const char *s, int *out) {
    int64_t v = 0;
    if (parse_i64_field(s, &v)) return 1;
    if (v < INT_MIN || v > INT_MAX) return 1;
    *out = (int)v;
    return 0;
}

static int parse_u32_field(const char *s, uint32_t *out) {
    uint64_t v = 0;
    if (parse_u64_field(s, &v) || v > UINT32_MAX) return 1;
    *out = (uint32_t)v;
    return 0;
}

static int split_tabs(char *line, char **fields, int n_fields) {
    int n = 0;
    fields[n++] = line;
    for (char *p = line; *p; p++) {
        if (*p == '\t') {
            *p = '\0';
            if (n >= n_fields) return -1;
            fields[n++] = p + 1;
        }
    }
    return n;
}

static bool valid_sidecar_name(const char *name, int owning_gpu) {
    if (!name || !*name) return false;
    if (strchr(name, '/') || strchr(name, '\\')) return false;
    char expect[32];
    snprintf(expect, sizeof(expect), "gpu%d.turbomind", owning_gpu);
    return strcmp(name, expect) == 0;
}

static int tm_append(ds4_tm_pack *pack, const ds4_tm_pack_entry *entry,
                     char *err, size_t errlen) {
    if (pack->n_entries == pack->cap_entries) {
        uint64_t next = pack->cap_entries ? pack->cap_entries * 2 : 256;
        if (next < pack->cap_entries || next > (uint64_t)SIZE_MAX / sizeof(pack->entries[0])) {
            return tm_error(err, errlen, "TurboMind pack index has too many rows");
        }
        ds4_tm_pack_entry *p =
            (ds4_tm_pack_entry *)realloc(pack->entries, (size_t)next * sizeof(pack->entries[0]));
        if (!p) return tm_error(err, errlen, "out of memory growing TurboMind pack index");
        pack->entries = p;
        pack->cap_entries = next;
    }
    pack->entries[pack->n_entries++] = *entry;
    return 0;
}

static int tm_find_semantic(const ds4_tm_pack *pack, const char *id) {
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (!strcmp(pack->entries[i].semantic_tensor_id, id)) return (int)i;
    }
    return -1;
}

static int parse_tm_row(ds4_tm_pack *pack, char *line, uint64_t line_no,
                        char *err, size_t errlen) {
    char *fields[DS4_TM_PACK_COLS];
    int n = split_tabs(line, fields, DS4_TM_PACK_COLS);
    if (n != DS4_TM_PACK_COLS) {
        return tm_error(err, errlen,
                        "TurboMind pack index line %" PRIu64 " has %d columns, expected %d",
                        line_no, n < 0 ? DS4_TM_PACK_COLS + 1 : n, DS4_TM_PACK_COLS);
    }

    ds4_tm_pack_entry e;
    memset(&e, 0, sizeof(e));
    e.semantic_tensor_id = tm_strdup(fields[0]);
    e.source_name = tm_strdup(fields[1]);
    e.source_dtype = tm_strdup(fields[2]);
    e.source_shape = tm_strdup(fields[3]);
    e.runtime_layout = tm_strdup(fields[4]);
    e.kernel_family = tm_strdup(fields[7]);
    e.sidecar_file = tm_strdup(fields[17]);
    e.source_shard_file = tm_strdup(fields[20]);
    e.source_checksum = tm_strdup(fields[23]);
    if (!e.semantic_tensor_id || !e.source_name || !e.source_dtype ||
        !e.source_shape || !e.runtime_layout || !e.kernel_family ||
        !e.sidecar_file || !e.source_shard_file || !e.source_checksum) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "out of memory parsing line %" PRIu64, line_no);
    }

    if (parse_i32_field(fields[5], &e.owning_gpu) || e.owning_gpu < 0 ||
        parse_i32_field(fields[6], &e.layer_id) ||
        parse_u32_field(fields[8], &e.n) ||
        parse_u32_field(fields[9], &e.k) ||
        parse_u32_field(fields[10], &e.experts_packed) ||
        parse_u32_field(fields[11], &e.experts_total) ||
        parse_u64_field(fields[12], &e.weight_bytes_per_expert) ||
        parse_u64_field(fields[13], &e.scale_bytes_per_expert) ||
        parse_i32_field(fields[14], &e.k_pack) ||
        parse_i32_field(fields[15], &e.weight_stride) ||
        parse_i32_field(fields[16], &e.scale_stride) ||
        parse_u64_field(fields[18], &e.weight_offset) ||
        parse_u64_field(fields[19], &e.scale_offset) ||
        parse_u64_field(fields[21], &e.source_shard_offset) ||
        parse_u64_field(fields[22], &e.source_byte_length) ||
        parse_i32_field(fields[24], &e.tm_abi_version)) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "bad numeric field on line %" PRIu64, line_no);
    }
    if (!e.n || !e.k || !e.experts_packed || !e.experts_total ||
        !e.weight_bytes_per_expert || !e.scale_bytes_per_expert) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "zero size field on line %" PRIu64, line_no);
    }
    if (e.experts_packed > e.experts_total) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "experts_packed exceeds experts_total on line %" PRIu64, line_no);
    }
    if (e.weight_offset > UINT64_MAX - (uint64_t)e.experts_packed * e.weight_bytes_per_expert ||
        e.scale_offset > UINT64_MAX - (uint64_t)e.experts_packed * e.scale_bytes_per_expert) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "sidecar offset overflow on line %" PRIu64, line_no);
    }
    if (tm_find_semantic(pack, e.semantic_tensor_id) >= 0) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "duplicate semantic_tensor_id on line %" PRIu64, line_no);
    }
    if (!valid_sidecar_name(e.sidecar_file, e.owning_gpu)) {
        tm_entry_free(&e);
        return tm_error(err, errlen, "unexpected sidecar_file '%s' on line %" PRIu64,
                        fields[17], line_no);
    }

    return tm_append(pack, &e, err, errlen);
}

int ds4_tm_pack_open(ds4_tm_pack **out, const char *path, char *err, size_t errlen) {
    if (!out || !path || !path[0]) return tm_error(err, errlen, "missing TurboMind pack index path");
    *out = NULL;

    char *buf = NULL;
    size_t len = 0;
    if (read_small_file(path, &buf, &len, err, errlen)) return 1;
    if (len >= 3 &&
        (unsigned char)buf[0] == 0xef &&
        (unsigned char)buf[1] == 0xbb &&
        (unsigned char)buf[2] == 0xbf) {
        free(buf);
        return tm_error(err, errlen, "TurboMind pack index has a UTF-8 BOM");
    }
    if (memchr(buf, '\r', len)) {
        free(buf);
        return tm_error(err, errlen, "TurboMind pack index must use LF line endings");
    }

    ds4_tm_pack *pack = (ds4_tm_pack *)calloc(1, sizeof(*pack));
    if (!pack) {
        free(buf);
        return tm_error(err, errlen, "out of memory allocating TurboMind pack index");
    }
    pack->path = tm_strdup(path);
    if (!pack->path) {
        ds4_tm_pack_close(pack);
        free(buf);
        return tm_error(err, errlen, "out of memory storing TurboMind pack path");
    }

    uint64_t line_no = 0;
    char *cursor = buf;
    char *end = buf + len;
    while (cursor <= end) {
        char *line = cursor;
        char *nl = memchr(cursor, '\n', (size_t)(end - cursor));
        if (nl) {
            *nl = '\0';
            cursor = nl + 1;
        } else {
            cursor = end + 1;
        }
        line_no++;
        if (line[0] == '\0') {
            if (cursor > end) break;
            continue;
        }
        if (line_no == 1) {
            if (strcmp(line, ds4_tm_pack_header)) {
                ds4_tm_pack_close(pack);
                free(buf);
                return tm_error(err, errlen, "TurboMind pack index header does not match expected schema");
            }
            continue;
        }
        if (parse_tm_row(pack, line, line_no, err, errlen)) {
            ds4_tm_pack_close(pack);
            free(buf);
            return 1;
        }
        if (cursor > end) break;
    }
    free(buf);
    if (pack->n_entries == 0) {
        ds4_tm_pack_close(pack);
        return tm_error(err, errlen, "TurboMind pack index has no tensor rows");
    }
    *out = pack;
    return 0;
}

uint64_t ds4_tm_pack_count(const ds4_tm_pack *pack) {
    return pack ? pack->n_entries : 0;
}

int ds4_tm_pack_max_gpu(const ds4_tm_pack *pack) {
    int max_gpu = -1;
    if (!pack) return max_gpu;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (pack->entries[i].owning_gpu > max_gpu) max_gpu = pack->entries[i].owning_gpu;
    }
    return max_gpu;
}

int ds4_tm_pack_lookup(const ds4_tm_pack *pack,
                       const char *semantic_tensor_id,
                       ds4_tm_pack_entry *out) {
    if (!pack || !semantic_tensor_id) return 1;
    int idx = tm_find_semantic(pack, semantic_tensor_id);
    if (idx < 0) return 1;
    if (out) *out = pack->entries[idx];
    return 0;
}

int ds4_tm_pack_for_each(const ds4_tm_pack *pack,
                         int (*cb)(const ds4_tm_pack_entry *entry, void *ud),
                         void *ud) {
    if (!pack || !cb) return 1;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        int rc = cb(&pack->entries[i], ud);
        if (rc) return rc;
    }
    return 0;
}

int ds4_tm_pack_sidecar_bytes(const ds4_tm_pack *pack,
                              const char *sidecar_file,
                              uint64_t *out) {
    if (!pack || !sidecar_file || !out) return 1;
    uint64_t end = 0;
    bool found = false;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        const ds4_tm_pack_entry *e = &pack->entries[i];
        if (strcmp(e->sidecar_file, sidecar_file)) continue;
        found = true;
        uint64_t weight_end =
            e->weight_offset + (uint64_t)e->experts_packed * e->weight_bytes_per_expert;
        uint64_t scale_end =
            e->scale_offset + (uint64_t)e->experts_packed * e->scale_bytes_per_expert;
        if (weight_end > end) end = weight_end;
        if (scale_end > end) end = scale_end;
    }
    if (!found) return 1;
    *out = end;
    return 0;
}
