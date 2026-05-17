#include "ds4_pack.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    DS4_PACK_COLS = 14
};

static const char *ds4_pack_header =
    "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\t"
    "runtime_layout\towning_gpu\tlayer_id\tkernel_family\tsource_offset\t"
    "byte_length\tshard_file\tshard_offset\tscale_offset\tchecksum";

struct ds4_pack {
    ds4_pack_entry *entries;
    uint64_t n_entries;
    uint64_t cap_entries;
    char *path;
};

static int pack_error(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static char *pack_strdup_len(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *pack_strdup(const char *s) {
    return pack_strdup_len(s, strlen(s));
}

static void pack_entry_free(ds4_pack_entry *e) {
    if (!e) return;
    free((char *)e->semantic_tensor_id);
    free((char *)e->source_name);
    free((char *)e->source_dtype);
    free((char *)e->source_shape);
    free((char *)e->runtime_layout);
    free((char *)e->kernel_family);
    free((char *)e->shard_file);
}

void ds4_pack_close(ds4_pack *pack) {
    if (!pack) return;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        pack_entry_free(&pack->entries[i]);
    }
    free(pack->entries);
    free(pack->path);
    free(pack);
}

static int read_small_file(const char *path, char **out, size_t *out_len,
                           char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return pack_error(err, errlen, "cannot open %s: %s", path, strerror(errno));
    if (fseek(fp, 0, SEEK_END) != 0) {
        int rc = pack_error(err, errlen, "cannot seek %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    long n = ftell(fp);
    if (n < 0) {
        int rc = pack_error(err, errlen, "cannot tell %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    if ((unsigned long)n > 64ul * 1024ul * 1024ul) {
        int rc = pack_error(err, errlen, "pack index is unexpectedly large: %s", path);
        fclose(fp);
        return rc;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        int rc = pack_error(err, errlen, "cannot rewind %s: %s", path, strerror(errno));
        fclose(fp);
        return rc;
    }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        return pack_error(err, errlen, "out of memory reading %s", path);
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    if (got != (size_t)n) {
        int rc = pack_error(err, errlen, "short read from %s", path);
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
    int64_t v;
    if (parse_i64_field(s, &v)) return 1;
    if (v < INT_MIN || v > INT_MAX) return 1;
    *out = (int)v;
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

static bool valid_shard_name(const char *name, int owning_gpu) {
    if (!name || !*name) return false;
    if (strchr(name, '/') || strchr(name, '\\')) return false;
    char expect[32];
    snprintf(expect, sizeof(expect), "gpu%d.weights", owning_gpu);
    return strcmp(name, expect) == 0;
}

static int pack_append(ds4_pack *pack, const ds4_pack_entry *entry,
                       char *err, size_t errlen) {
    if (pack->n_entries == pack->cap_entries) {
        uint64_t next = pack->cap_entries ? pack->cap_entries * 2 : 256;
        if (next < pack->cap_entries || next > (uint64_t)SIZE_MAX / sizeof(pack->entries[0])) {
            return pack_error(err, errlen, "pack index has too many rows");
        }
        ds4_pack_entry *p = (ds4_pack_entry *)realloc(pack->entries, (size_t)next * sizeof(pack->entries[0]));
        if (!p) return pack_error(err, errlen, "out of memory growing pack index");
        pack->entries = p;
        pack->cap_entries = next;
    }
    pack->entries[pack->n_entries++] = *entry;
    return 0;
}

static int pack_find_semantic(const ds4_pack *pack, const char *id) {
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (!strcmp(pack->entries[i].semantic_tensor_id, id)) return (int)i;
    }
    return -1;
}

static int pack_find_source_name(const ds4_pack *pack, const char *name, size_t name_len) {
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        const char *row = pack->entries[i].source_name;
        if (strlen(row) == name_len && memcmp(row, name, name_len) == 0) return (int)i;
    }
    return -1;
}

static int parse_pack_row(ds4_pack *pack, char *line, uint64_t line_no,
                          char *err, size_t errlen) {
    char *fields[DS4_PACK_COLS];
    int n = split_tabs(line, fields, DS4_PACK_COLS);
    if (n != DS4_PACK_COLS) {
        return pack_error(err, errlen,
                          "pack index line %" PRIu64 " has %d columns, expected %d",
                          line_no, n < 0 ? DS4_PACK_COLS + 1 : n, DS4_PACK_COLS);
    }

    ds4_pack_entry e;
    memset(&e, 0, sizeof(e));
    e.semantic_tensor_id = pack_strdup(fields[0]);
    e.source_name = pack_strdup(fields[1]);
    e.source_dtype = pack_strdup(fields[2]);
    e.source_shape = pack_strdup(fields[3]);
    e.runtime_layout = pack_strdup(fields[4]);
    e.kernel_family = pack_strdup(fields[7]);
    e.shard_file = pack_strdup(fields[10]);
    if (!e.semantic_tensor_id || !e.source_name || !e.source_dtype ||
        !e.source_shape || !e.runtime_layout || !e.kernel_family || !e.shard_file) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "out of memory parsing line %" PRIu64, line_no);
    }
    if (parse_i32_field(fields[5], &e.owning_gpu) || e.owning_gpu < 0) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "bad owning_gpu on line %" PRIu64, line_no);
    }
    if (parse_i32_field(fields[6], &e.layer_id)) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "bad layer_id on line %" PRIu64, line_no);
    }
    if (parse_u64_field(fields[8], &e.source_offset) ||
        parse_u64_field(fields[9], &e.byte_length) ||
        parse_u64_field(fields[11], &e.shard_offset) ||
        parse_i64_field(fields[12], &e.scale_offset)) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "bad numeric field on line %" PRIu64, line_no);
    }
    if (e.byte_length == 0) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "zero byte_length on line %" PRIu64, line_no);
    }
    if (e.source_offset > UINT64_MAX - e.byte_length ||
        e.shard_offset > UINT64_MAX - e.byte_length) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "offset overflow on line %" PRIu64, line_no);
    }
    if (pack_find_semantic(pack, e.semantic_tensor_id) >= 0) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "duplicate semantic_tensor_id on line %" PRIu64, line_no);
    }
    if (pack_find_source_name(pack, e.source_name, strlen(e.source_name)) >= 0) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "duplicate source_name on line %" PRIu64, line_no);
    }
    if (e.shard_file[0] && !valid_shard_name(e.shard_file, e.owning_gpu)) {
        pack_entry_free(&e);
        return pack_error(err, errlen, "unexpected shard_file '%s' on line %" PRIu64,
                          fields[10], line_no);
    }

    return pack_append(pack, &e, err, errlen);
}

int ds4_pack_open(ds4_pack **out, const char *path, char *err, size_t errlen) {
    if (!out || !path || !path[0]) return pack_error(err, errlen, "missing pack index path");
    *out = NULL;

    char *buf = NULL;
    size_t len = 0;
    if (read_small_file(path, &buf, &len, err, errlen)) return 1;
    if (len >= 3 &&
        (unsigned char)buf[0] == 0xef &&
        (unsigned char)buf[1] == 0xbb &&
        (unsigned char)buf[2] == 0xbf) {
        free(buf);
        return pack_error(err, errlen, "pack index has a UTF-8 BOM");
    }
    if (memchr(buf, '\r', len)) {
        free(buf);
        return pack_error(err, errlen, "pack index must use LF line endings");
    }

    ds4_pack *pack = (ds4_pack *)calloc(1, sizeof(*pack));
    if (!pack) {
        free(buf);
        return pack_error(err, errlen, "out of memory allocating pack index");
    }
    pack->path = pack_strdup(path);
    if (!pack->path) {
        ds4_pack_close(pack);
        free(buf);
        return pack_error(err, errlen, "out of memory storing pack path");
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
            if (strcmp(line, ds4_pack_header)) {
                ds4_pack_close(pack);
                free(buf);
                return pack_error(err, errlen, "pack index header does not match expected schema");
            }
            continue;
        }
        if (parse_pack_row(pack, line, line_no, err, errlen)) {
            ds4_pack_close(pack);
            free(buf);
            return 1;
        }
        if (cursor > end) break;
    }
    free(buf);
    if (pack->n_entries == 0) {
        ds4_pack_close(pack);
        return pack_error(err, errlen, "pack index has no tensor rows");
    }
    *out = pack;
    return 0;
}

uint64_t ds4_pack_count(const ds4_pack *pack) {
    return pack ? pack->n_entries : 0;
}

int ds4_pack_max_gpu(const ds4_pack *pack) {
    int max_gpu = -1;
    if (!pack) return max_gpu;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (pack->entries[i].owning_gpu > max_gpu) max_gpu = pack->entries[i].owning_gpu;
    }
    return max_gpu;
}

uint64_t ds4_pack_payload_bytes(const ds4_pack *pack, int gpu) {
    uint64_t total = 0;
    if (!pack || gpu < 0) return 0;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        const ds4_pack_entry *e = &pack->entries[i];
        if (e->owning_gpu == gpu) total += e->byte_length;
    }
    return total;
}

uint64_t ds4_pack_arena_bytes(const ds4_pack *pack, int gpu) {
    uint64_t end = 0;
    if (!pack || gpu < 0) return 0;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        const ds4_pack_entry *e = &pack->entries[i];
        if (e->owning_gpu == gpu && e->shard_offset + e->byte_length > end) {
            end = e->shard_offset + e->byte_length;
        }
    }
    return end;
}

uint64_t ds4_pack_tensor_count(const ds4_pack *pack, int gpu) {
    uint64_t total = 0;
    if (!pack || gpu < 0) return 0;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (pack->entries[i].owning_gpu == gpu) total++;
    }
    return total;
}

int ds4_pack_lookup(const ds4_pack *pack,
                    const char *semantic_tensor_id,
                    ds4_pack_entry *out) {
    if (!pack || !semantic_tensor_id) return 1;
    int idx = pack_find_semantic(pack, semantic_tensor_id);
    if (idx < 0) return 1;
    if (out) *out = pack->entries[idx];
    return 0;
}

int ds4_pack_for_each(const ds4_pack *pack,
                      int (*cb)(const ds4_pack_entry *entry, void *ud),
                      void *ud) {
    if (!pack || !cb) return 1;
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        int rc = cb(&pack->entries[i], ud);
        if (rc) return rc;
    }
    return 0;
}

static const char *row_status(const ds4_pack_entry *e,
                              const ds4_pack_source_tensor *s,
                              uint64_t source_file_size,
                              int n_gpus) {
    if (!e) return "MISSING_PACK_ROW";
    if (n_gpus > 0 && (e->owning_gpu < 0 || e->owning_gpu >= n_gpus)) return "BAD_OWNING_GPU";
    if (e->source_offset > source_file_size ||
        e->byte_length > source_file_size - e->source_offset) return "OFFSET_OUT_OF_RANGE";
    if (strcmp(e->source_dtype, s->source_dtype)) return "DTYPE_MISMATCH";
    if (strcmp(e->source_shape, s->source_shape)) return "SHAPE_MISMATCH";
    if (e->source_offset != s->source_offset) return "OFFSET_MISMATCH";
    if (e->byte_length != s->byte_length) return "BYTE_LENGTH_MISMATCH";
    return "OK";
}

int ds4_pack_reconcile(const ds4_pack *pack,
                       const ds4_pack_source_tensor *source,
                       size_t n_source,
                       uint64_t source_file_size,
                       int n_gpus,
                       FILE *report,
                       ds4_pack_reconcile_summary *summary,
                       char *err,
                       size_t errlen) {
    if (!pack || !source) return pack_error(err, errlen, "missing reconcile input");
    ds4_pack_reconcile_summary local;
    memset(&local, 0, sizeof(local));
    local.source_tensors = (uint64_t)n_source;
    local.pack_rows = pack->n_entries;

    unsigned char *matched = (unsigned char *)calloc((size_t)pack->n_entries, 1);
    if (!matched) return pack_error(err, errlen, "out of memory reconciling pack");

    if (report) {
        fprintf(report, "semantic_tensor_id\tsource_name\tsource_dtype\tsource_shape\towning_gpu\tlayer_id\tbyte_length\tstatus\n");
    }

    for (size_t i = 0; i < n_source; i++) {
        const ds4_pack_source_tensor *s = &source[i];
        int idx = pack_find_source_name(pack, s->name, s->name_len);
        const ds4_pack_entry *e = idx >= 0 ? &pack->entries[idx] : NULL;
        if (idx >= 0) matched[idx] = 1;
        const char *status = row_status(e, s, source_file_size, n_gpus);
        if (!strcmp(status, "OK")) local.ok_rows++;
        else local.failed_rows++;
        if (report) {
            if (e) {
                fprintf(report, "%s\t%s\t%s\t%s\t%d\t%d\t%" PRIu64 "\t%s\n",
                        e->semantic_tensor_id,
                        e->source_name,
                        e->source_dtype,
                        e->source_shape,
                        e->owning_gpu,
                        e->layer_id,
                        e->byte_length,
                        status);
            } else {
                fprintf(report, "-\t%.*s\t%s\t%s\t-\t-\t%" PRIu64 "\t%s\n",
                        (int)s->name_len,
                        s->name,
                        s->source_dtype,
                        s->source_shape,
                        s->byte_length,
                        status);
            }
        }
    }

    for (uint64_t i = 0; i < pack->n_entries; i++) {
        if (matched[i]) continue;
        local.extra_pack_rows++;
        local.failed_rows++;
        if (report) {
            const ds4_pack_entry *e = &pack->entries[i];
            fprintf(report, "%s\t%s\t%s\t%s\t%d\t%d\t%" PRIu64 "\tEXTRA_PACK_ROW\n",
                    e->semantic_tensor_id,
                    e->source_name,
                    e->source_dtype,
                    e->source_shape,
                    e->owning_gpu,
                    e->layer_id,
                    e->byte_length);
        }
    }
    free(matched);

    if (summary) *summary = local;
    if (local.failed_rows || local.extra_pack_rows || local.ok_rows != local.pack_rows) {
        return pack_error(err, errlen,
                          "pack reconcile failed: ok=%" PRIu64 " failed=%" PRIu64 " extra=%" PRIu64,
                          local.ok_rows, local.failed_rows, local.extra_pack_rows);
    }
    return 0;
}

static int join_path(char *out, size_t outlen, const char *dir, const char *name) {
    size_t dlen = strlen(dir);
    while (dlen && dir[dlen - 1] == '/') dlen--;
    int n = snprintf(out, outlen, "%.*s/%s", (int)dlen, dir, name);
    return n < 0 || (size_t)n >= outlen;
}

static int file_size(const char *path, uint64_t *out, char *err, size_t errlen) {
    struct stat st;
    if (stat(path, &st) != 0) return pack_error(err, errlen, "cannot stat %s: %s", path, strerror(errno));
    if (st.st_size < 0) return pack_error(err, errlen, "negative size for %s", path);
    *out = (uint64_t)st.st_size;
    return 0;
}

int ds4_pack_validate_shards(const ds4_pack *pack,
                             const char *shard_dir,
                             FILE *report,
                             char *err,
                             size_t errlen) {
    if (!pack || !shard_dir || !shard_dir[0]) {
        return pack_error(err, errlen, "missing shard validation input");
    }
    if (report) fprintf(report, "shard_file\towning_gpu\trequired_bytes\tfile_bytes\tstatus\n");
    for (int gpu = 0; gpu <= ds4_pack_max_gpu(pack); gpu++) {
        uint64_t required = ds4_pack_arena_bytes(pack, gpu);
        if (required == 0) continue;
        char name[32];
        char path[4096];
        snprintf(name, sizeof(name), "gpu%d.weights", gpu);
        if (join_path(path, sizeof(path), shard_dir, name)) {
            return pack_error(err, errlen, "shard path is too long for %s", name);
        }
        uint64_t size = 0;
        if (file_size(path, &size, err, errlen)) return 1;
        const char *status = size >= required ? "OK" : "SHORT_SHARD";
        if (report) fprintf(report, "%s\t%d\t%" PRIu64 "\t%" PRIu64 "\t%s\n",
                            name, gpu, required, size, status);
        if (size < required) {
            return pack_error(err, errlen, "shard %s is short: have %" PRIu64 ", need %" PRIu64,
                              path, size, required);
        }
    }
    for (uint64_t i = 0; i < pack->n_entries; i++) {
        const ds4_pack_entry *e = &pack->entries[i];
        if (!valid_shard_name(e->shard_file, e->owning_gpu)) {
            return pack_error(err, errlen, "bad shard file for %s", e->semantic_tensor_id);
        }
    }
    return 0;
}
