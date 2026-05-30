struct TpEpHttpSessionSlot {
    int id = -1;
    bool occupied = false;
    bool kv_valid = false;
    bool hc_valid = false;
    bool prompt_fingerprint_known = false;
    std::string key;
    uint64_t prompt_fingerprint = 0;
    std::vector<uint32_t> prompt_token_ids;
    std::vector<uint32_t> generated_token_ids;
    uint64_t pos = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t last_used = 0;
    uint32_t last_selected_token = UINT32_MAX;
    uint32_t mtp_raw_rows = 0;
};

struct TpEpHttpSessionAssignment {
    int slot = -1;
    bool hit = false;
    bool prompt_match = true;
    bool evicted = false;
    std::string evicted_key;
    uint64_t pos_in = 0;
    uint64_t pos_out = 0;
};

struct TpEpHttpContextAdmission {
    bool ok = false;
    bool cache_hit = false;
    uint64_t start_position = 0;
    uint64_t prompt_prefill_steps = 0;
    uint64_t requested_decode_steps = 0;
    uint64_t final_position = 0;
    uint64_t ctx = 262144ull;
};

struct TpEpHttpSessionTable {
    std::vector<TpEpHttpSessionSlot> slots;
    uint64_t clock = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    void init(int n_slots) {
        slots.resize((size_t)n_slots);
        for (int i = 0; i < n_slots; ++i) {
            slots[(size_t)i].id = i;
        }
    }

    int find(const std::string &key) const {
        for (const auto &slot : slots) {
            if (slot.occupied && slot.key == key) return slot.id;
        }
        return -1;
    }

    bool slot_prompt_matches(const TpEpHttpSessionSlot &slot,
                             bool prompt_present,
                             uint64_t prompt_fingerprint) const {
        if (!prompt_present) return true;
        return slot.prompt_fingerprint_known &&
               slot.prompt_fingerprint == prompt_fingerprint;
    }

    uint64_t preview_position(const std::string &key,
                              bool prompt_present,
                              uint64_t prompt_fingerprint,
                              uint64_t base_pos) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            return slots[(size_t)idx].pos;
        }
        return base_pos;
    }

    bool preview_hit(const std::string &key,
                     bool prompt_present,
                     uint64_t prompt_fingerprint,
                     uint64_t *pos_out) const {
        const int idx = find(key);
        if (idx >= 0 &&
            slots[(size_t)idx].kv_valid &&
            slots[(size_t)idx].hc_valid &&
            slot_prompt_matches(slots[(size_t)idx],
                                prompt_present,
                                prompt_fingerprint)) {
            if (pos_out) *pos_out = slots[(size_t)idx].pos;
            return true;
        }
        return false;
    }

    TpEpHttpSessionAssignment assign(const std::string &key,
                                     bool prompt_present,
                                     uint64_t prompt_fingerprint,
                                     const std::vector<uint32_t> &prompt_tokens,
                                     uint64_t base_pos,
                                     const std::vector<bool> &protected_slots) {
        TpEpHttpSessionAssignment a;
        ++clock;

        int idx = find(key);
        if (idx >= 0) {
            auto &slot = slots[(size_t)idx];
            const bool prompt_match =
                slot_prompt_matches(slot, prompt_present, prompt_fingerprint);
            a.slot = idx;
            a.prompt_match = prompt_match;
            a.hit = slot.kv_valid && slot.hc_valid && prompt_match;
            a.pos_in = a.hit ? slot.pos : base_pos;
            slot.last_used = clock;
            if (a.hit) {
                hits++;
                slot.hits++;
            } else {
                misses++;
                slot.misses++;
                slot.kv_valid = false;
                slot.hc_valid = false;
                slot.pos = base_pos;
                if (prompt_present) {
                    slot.prompt_fingerprint_known = true;
                    slot.prompt_fingerprint = prompt_fingerprint;
                }
                slot.prompt_token_ids = prompt_tokens;
                slot.generated_token_ids.clear();
                slot.prompt_tokens = 0;
                slot.generated_tokens = 0;
                slot.last_selected_token = UINT32_MAX;
                slot.mtp_raw_rows = 0;
            }
            return a;
        }

        for (auto &slot : slots) {
            if (!slot.occupied) {
                idx = slot.id;
                break;
            }
        }
        if (idx < 0) {
            uint64_t best_last = UINT64_MAX;
            for (const auto &slot : slots) {
                if (slot.id >= 0 && slot.id < (int)protected_slots.size() &&
                    protected_slots[(size_t)slot.id]) {
                    continue;
                }
                if (slot.last_used < best_last) {
                    best_last = slot.last_used;
                    idx = slot.id;
                }
            }
        }
        if (idx < 0) return a;

        auto &slot = slots[(size_t)idx];
        if (slot.occupied) {
            a.evicted = true;
            a.evicted_key = slot.key;
            evictions++;
        }
        slot.occupied = true;
        slot.kv_valid = false;
        slot.hc_valid = false;
        slot.prompt_fingerprint_known = prompt_present;
        slot.key = key;
        slot.prompt_fingerprint = prompt_present ? prompt_fingerprint : 0;
        slot.prompt_token_ids = prompt_tokens;
        slot.generated_token_ids.clear();
        slot.pos = base_pos;
        slot.prompt_tokens = 0;
        slot.generated_tokens = 0;
        slot.hits = 0;
        slot.misses = 1;
        slot.last_used = clock;
        slot.last_selected_token = UINT32_MAX;
        slot.mtp_raw_rows = 0;
        misses++;

        a.slot = idx;
        a.hit = false;
        a.prompt_match = true;
        a.pos_in = base_pos;
        return a;
    }

    void commit(const TpEpHttpSessionAssignment &a,
                uint64_t prompt_tokens,
                uint64_t generated_tokens,
                uint64_t position_advance,
                const std::vector<uint32_t> &selected_tokens) {
        if (a.slot < 0 || a.slot >= (int)slots.size()) return;
        auto &slot = slots[(size_t)a.slot];
        slot.kv_valid = true;
        slot.hc_valid = true;
        slot.pos = a.pos_in + position_advance;
        slot.prompt_tokens += prompt_tokens;
        slot.generated_tokens += generated_tokens;
        for (uint32_t selected_token : selected_tokens) {
            if (selected_token != UINT32_MAX) {
                slot.generated_token_ids.push_back(selected_token);
                slot.last_selected_token = selected_token;
            }
        }
        slot.last_used = ++clock;
    }

    int used() const {
        int n = 0;
        for (const auto &slot : slots) n += slot.occupied ? 1 : 0;
        return n;
    }

    void slots_json(char *out, size_t out_size) const {
        size_t used_bytes = 0;
        int n = std::snprintf(out, out_size,
                              "{\"slots_total\":%zu,\"slots_used\":%d,"
                              "\"cache_hits\":%llu,\"cache_misses\":%llu,"
                              "\"cache_evictions\":%llu,\"slots\":[",
                              slots.size(), used(),
                              (unsigned long long)hits,
                              (unsigned long long)misses,
                              (unsigned long long)evictions);
        if (n < 0) return;
        used_bytes = (size_t)std::min(n, (int)out_size);
        for (size_t i = 0; i < slots.size() && used_bytes + 256 < out_size; ++i) {
            const auto &slot = slots[i];
            const std::string key = http_json_escape(slot.key);
            n = std::snprintf(out + used_bytes, out_size - used_bytes,
                              "%s{\"id\":%d,\"occupied\":%d,\"key\":\"%s\","
                              "\"pos\":%llu,\"kv_valid\":%d,\"hc_valid\":%d,"
                              "\"prompt_fingerprint_known\":%d,"
                              "\"prompt_fingerprint\":%llu,"
                              "\"prompt_tokens\":%llu,\"generated_tokens\":%llu,"
                              "\"prompt_token_ids\":%zu,"
                              "\"generated_token_ids\":%zu,"
                              "\"last_selected_token\":%u,"
                              "\"mtp_raw_rows\":%u,"
                              "\"hits\":%llu,\"misses\":%llu}",
                              i == 0 ? "" : ",",
                              slot.id, slot.occupied ? 1 : 0, key.c_str(),
                              (unsigned long long)slot.pos,
                              slot.kv_valid ? 1 : 0,
                              slot.hc_valid ? 1 : 0,
                              slot.prompt_fingerprint_known ? 1 : 0,
                              (unsigned long long)slot.prompt_fingerprint,
                              (unsigned long long)slot.prompt_tokens,
                              (unsigned long long)slot.generated_tokens,
                              slot.prompt_token_ids.size(),
                              slot.generated_token_ids.size(),
                              slot.last_selected_token,
                              slot.mtp_raw_rows,
                              (unsigned long long)slot.hits,
                              (unsigned long long)slot.misses);
            if (n < 0) break;
            used_bytes += (size_t)n;
        }
        if (used_bytes + 4 < out_size) {
            std::snprintf(out + used_bytes, out_size - used_bytes, "]}\n");
        }
    }
};

static TpEpHttpContextAdmission tp_ep_http_context_admission(
    const TpEpHttpSessionTable &sessions,
    const HttpParsedRequest &req,
    uint64_t base_position,
    uint64_t ctx) {
    TpEpHttpContextAdmission out;
    out.ctx = ctx;
    out.requested_decode_steps =
        req.requested_tokens > 0 ? (uint64_t)req.requested_tokens : 0ull;
    uint64_t hit_pos = 0;
    out.cache_hit = sessions.preview_hit(req.cache_key,
                                         req.prompt_fingerprint_present,
                                         req.prompt_fingerprint,
                                         &hit_pos);
    out.start_position = out.cache_hit ? hit_pos : base_position;
    if (!out.cache_hit && req.prompt_token_ids.size() > 1) {
        out.prompt_prefill_steps =
            (uint64_t)req.prompt_token_ids.size() - 1ull;
    }
    out.final_position = out.start_position + out.prompt_prefill_steps +
                         out.requested_decode_steps;
    out.ok = out.final_position <= out.ctx;
    return out;
}

static std::string tp_ep_http_context_error_json(
    const TpEpHttpContextAdmission &admission) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"error\":\"context_window_exceeded\","
                  "\"ctx\":%llu,"
                  "\"start_position\":%llu,"
                  "\"prompt_prefill_steps\":%llu,"
                  "\"requested_decode_steps\":%llu,"
                  "\"final_position\":%llu,"
                  "\"cache_hit\":%d}\n",
                  (unsigned long long)admission.ctx,
                  (unsigned long long)admission.start_position,
                  (unsigned long long)admission.prompt_prefill_steps,
                  (unsigned long long)admission.requested_decode_steps,
                  (unsigned long long)admission.final_position,
                  admission.cache_hit ? 1 : 0);
    return std::string(buf);
}

static unsigned long long http_epoch_seconds() {
    using namespace std::chrono;
    return (unsigned long long)duration_cast<seconds>(
        system_clock::now().time_since_epoch()).count();
}

static void http_drain_matching_pending(std::deque<HttpParsedRequest> *pending,
                                        int requested_tokens,
                                        uint64_t cache_position,
                                        int max_batch,
                                        std::vector<HttpParsedRequest> *batch) {
    for (auto it = pending->begin();
         it != pending->end() && (int)batch->size() < max_batch;) {
        bool duplicate_key = false;
        for (const auto &req : *batch) {
            if (req.cache_key == it->cache_key) {
                duplicate_key = true;
                break;
            }
        }
        if (it->requested_tokens == requested_tokens &&
            it->cache_position == cache_position &&
            !duplicate_key) {
            batch->push_back(std::move(*it));
            it = pending->erase(it);
        } else {
            ++it;
        }
    }
}
