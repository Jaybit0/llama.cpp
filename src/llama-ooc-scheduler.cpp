#include "llama-ooc-scheduler.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-expert-ooc.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <strings.h>

// Global singleton pointer (non-owning). Users manage lifetime externally.
static std::atomic<llama_ooc_scheduler_i *> g_llama_ooc_sched{nullptr};

void llama_set_ooc_scheduler(llama_ooc_scheduler_i * sched) {
    g_llama_ooc_sched.store(sched, std::memory_order_release);
}

llama_ooc_scheduler_i * llama_get_ooc_scheduler() {
    return g_llama_ooc_sched.load(std::memory_order_acquire);
}

namespace {

class llama_ooc_scheduler_basic final : public llama_ooc_scheduler_i {
public:
    llama_ooc_scheduler_basic() {
        const char * v = std::getenv("LLAMA_OOC_DISABLE_UNUSED");
        disable_unused_ = v && (*v == '1' || strcasecmp(v, "true") == 0 || strcasecmp(v, "on") == 0);
        const char * w = std::getenv("LLAMA_OOC_WARMUP_TOKENS");
        warmup_tokens_ = w ? std::max(1, atoi(w)) : 8;
        const char * e = std::getenv("LLAMA_OOC_EPS");
        eps_ = e ? std::max(0.0, atof(e)) : 1e-5;
        if (disable_unused_) {
            LLAMA_LOG_INFO("ooc: will disable unused MoE layers after %d tokens (eps=%.1e)\n", warmup_tokens_, eps_);
        }
    }

    ggml_backend_buffer_type_t override_weight_buffer_type(
        const llama_model & /*model*/, const char * tensor_name, int /*layer_index*/, int usage_op,
        ggml_backend_dev_t /*assigned_dev*/, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & candidates,
        ggml_backend_buffer_type_t suggested) override {

        if (true)
            return nullptr; // disable override for now

        // Heuristic: if a weight is used with MUL_MAT_ID/ADD_ID (typical for MoE experts) and its name indicates expert tensors,
        // prefer CPU buffer type to avoid VRAM pressure. Otherwise, keep default.
        // usage_op values are ggml_op enum; match by name to avoid including ggml headers here beyond backend types.
        const bool is_moe_op = (usage_op == GGML_OP_MUL_MAT_ID) || (usage_op == GGML_OP_ADD_ID);
        const bool name_moe = tensor_name && (std::strstr(tensor_name, "ffn_up_exps") ||
                                              std::strstr(tensor_name, "ffn_down_exps") ||
                                              std::strstr(tensor_name, "ffn_gate_exps") ||
                                              std::strstr(tensor_name, "ffn_gate_inp"));
        if (!(is_moe_op && name_moe)) {
            return nullptr; // no override
        }

        // Prefer the default CPU buffer type (avoid CPU extra/repack bufts)
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
            if (cpu_buft) {
                LLAMA_LOG_INFO("ooc/basic: placing %s on CPU (%s) instead of %s\n",
                    tensor_name,
                    ggml_backend_buft_name(cpu_buft),
                    suggested ? ggml_backend_buft_name(suggested) : "<none>");
                return cpu_buft;
            }
        }
        return nullptr;
    }

    void on_graph_tensor(const llama_model & /*model*/, const llama_ubatch & /*ubatch*/, ggml_backend_sched_t /*sched*/, ggml_tensor * /*node*/, const char * name, int il) override {
        // Build-time only; keep debug-level noise minimal
        if (name && std::strstr(name, "ffn_moe_")) {
            LLAMA_LOG_DEBUG("ooc/basic: build has node %s (il=%d)\n", name, il);
        }
    }

    static bool fetch_bytes(ggml_backend_sched_t sched, ggml_tensor * t, std::vector<uint8_t> & out) {
        if (!t) return false;
        size_t nbytes = ggml_nbytes(t);
        if (nbytes == 0) return false;
        out.resize(nbytes);
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
        if (!b) return false;
        ggml_backend_tensor_get_async(b, t, out.data(), 0, nbytes);
        ggml_backend_sched_synchronize(sched);
        return true;
    }

    void on_eval_node(const llama_model & model, ggml_backend_sched_t sched, ggml_tensor * node, bool ask) override {
        if (ask) return;
        const char * name = ggml_get_name(node);
        if (!name) return;

        // Clean base name and layer id
        std::string sname(name);
        if (auto sp = sname.find(' '); sp != std::string::npos) sname.resize(sp);
        int layer = -1;
        if (auto pos = sname.rfind('-'); pos != std::string::npos) {
            layer = std::atoi(sname.c_str() + pos + 1);
            sname.resize(pos);
        }

        const bool is_logits  = (sname == "ffn_moe_logits");
        const bool is_logitsb = (sname == "ffn_moe_logits_biased");
        const bool is_probs   = (sname == "ffn_moe_probs");
        const bool is_probsb  = (sname == "ffn_moe_probs_biased");
        const bool is_topk    = (sname == "ffn_moe_topk"); // I32 indices of selected experts
        if (!(is_logits || is_logitsb || is_probs || is_probsb || is_topk)) return;

        // Filter reshaped/derived nodes: ensure expert dimension matches model's n_expert (if available)
        const int64_t n_expert_expected = model.hparams.n_expert;
        if (!is_topk && n_expert_expected > 0 && node->ne[0] != n_expert_expected) {
            return;
        }

        // Fetch bytes
        std::vector<uint8_t> buf;
        if (!fetch_bytes(sched, node, buf)) return;

        //if (node->type != GGML_TYPE_F32) return; // scores are F32

        const int il = layer;
        init_once(model);
        const int64_t n_expert = is_topk ? (int64_t) model.hparams.n_expert : node->ne[0];
        const size_t  nb0      = node->nb[0];
        const size_t  nb1      = node->nb[1];
        const int64_t tok      = 0; // first token column
        const uint8_t * base   = buf.data();

        struct Pair { float v; int i; };
        std::vector<Pair> v; v.reserve((size_t) n_expert);
        for (int64_t e = 0; e < n_expert; ++e) {
            const float * p = reinterpret_cast<const float *>(base + tok*nb1 + e*nb0);
            v.push_back({*p, (int)e});
        }

        if (is_topk && node->type == GGML_TYPE_I32) {
            // Record used experts for application before next graph build
            if (pending_used_.empty()) pending_used_.resize((size_t) model.hparams.n_layer);
            auto & used = pending_used_[(size_t) il];
            used.assign((size_t) n_expert, 0);
            const int64_t k = node->ne[0]; // top-k
            for (int64_t i = 0; i < k; ++i) {
                const int32_t * p = reinterpret_cast<const int32_t *>(base + tok*nb1 + i*nb0);
                int idx = *p;
                if (idx >= 0 && idx < n_expert) used[(size_t) idx] = 1;
            }
            pending_apply_ = true;
            return;
        }

        // (Optional) report top-8 logits/probs for analysis
        const size_t K = std::min<size_t>(8, v.size());
        std::partial_sort(v.begin(), v.begin() + K, v.end(), [](const Pair &a, const Pair &b){ return a.v > b.v; });
        std::string s_idx = "[", s_val = "[";
        for (size_t i = 0; i < K; ++i) {
            s_idx += std::to_string(v[i].i);
            char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%.4f", v[i].v);
            s_val += tmp;
            if (i + 1 < K) { s_idx += ","; s_val += ","; }
        }
        s_idx += "]"; s_val += "]";
        const char * kind = is_logits ? "logits" : is_logitsb ? "logits_biased" : is_probs ? "probs" : "probs_biased";
        //LLAMA_LOG_INFO("ooc/basic: layer=%d %s_top8 idx=%s vals=%s\n", il, kind, s_idx.c_str(), s_val.c_str());

        // Track MoE usage and optionally disable unused layers after warmup
        if (disable_unused_ && (is_probs || is_logits)) {
            seen_tokens_[il] += 1;
            // detect signal variance above eps
            double mn = 1e300, mx = -1e300;
            for (const auto &p : v) { mn = std::min<double>(mn, p.v); mx = std::max<double>(mx, p.v); }
            if (mx - mn > eps_) has_signal_[il] = true;
            if (!has_disabled_[il] && seen_tokens_[il] >= warmup_tokens_ && !has_signal_[il]) {
                enabled_[il] = false;
                has_disabled_[il] = true;
                LLAMA_LOG_WARN("ooc: disabling MoE layer %d due to no router signal after %d tokens\n", il, warmup_tokens_);
            }
        }
    }

    bool moe_layer_enabled(int il) const override {
        if (il < 0 || il >= (int) enabled_.size()) return true;
        return enabled_[il];
    }

private:
    void init_once(const llama_model & model) {
        if (!inited_) {
            int n = (int) model.hparams.n_layer;
            enabled_      = std::vector<bool>(n, true);
            seen_tokens_  = std::vector<int>(n, 0);
            has_signal_   = std::vector<bool>(n, false);
            has_disabled_ = std::vector<bool>(n, false);
            resident_.resize(n);
            inited_ = true;
        }
    }

    bool disable_unused_ = false;
    int  warmup_tokens_  = 8;
    double eps_          = 1e-5;

    bool inited_ = false;
    std::vector<bool> enabled_;
    std::vector<int>  seen_tokens_;
    std::vector<bool> has_signal_;
    std::vector<bool> has_disabled_;
    std::vector<std::vector<char>> resident_;
    std::vector<std::vector<char>> pending_used_;
    bool pending_apply_ = false;
public:
    void on_graph_build_tick(const llama_model & model, ggml_backend_sched_t /*sched*/) override {
        if (!pending_apply_) return;
        init_once(model);
        const int n_layer = (int) model.hparams.n_layer;
        for (int il = 0; il < n_layer; ++il) {
            auto & used = pending_used_[(size_t) il];
            if (used.empty()) continue;
            if (resident_[(size_t) il].empty()) resident_[(size_t) il] = std::vector<char>(used.size(), 1);
            for (int e = 0; e < (int) used.size(); ++e) {
                if (used[(size_t) e]) {
                    if (!resident_[(size_t) il][(size_t) e]) {
                        llama_expert_load(const_cast<llama_model*>(&model), il, e);
                        resident_[(size_t) il][(size_t) e] = 1;
                        LLAMA_LOG_INFO("ooc: load layer %d expert %d\n", il, e);
                    }
                } else {
                    if (resident_[(size_t) il][(size_t) e]) {
                        llama_expert_evict(const_cast<llama_model*>(&model), il, e);
                        resident_[(size_t) il][(size_t) e] = 0;
                        LLAMA_LOG_INFO("ooc: evict layer %d expert %d\n", il, e);
                    }
                }
            }
        }
        pending_apply_ = false;
    }
};

} // namespace

void llama_ooc_init_from_env() {
    const char * v = std::getenv("LLAMA_OOC_SCHED");
    if (!v || !*v) {
        return; // unset → leave as-is
    }
    std::string s(v);
    // trim
    s.erase(0, s.find_first_not_of(" \t\n\r"));
    s.erase(s.find_last_not_of(" \t\n\r") + 1);
    // lowercase
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });

    if (s == "0" || s == "off" || s == "false" || s == "none" || s == "disable" || s == "disabled") {
        llama_set_ooc_scheduler(nullptr);
        LLAMA_LOG_INFO("ooc: scheduler disabled via LLAMA_OOC_SCHED=%s\n", v);
        return;
    }

    // currently supported: basic
    if (s == "1" || s == "on" || s == "true" || s == "basic") {
        static std::unique_ptr<llama_ooc_scheduler_basic> holder;
        holder.reset(new llama_ooc_scheduler_basic());
        llama_set_ooc_scheduler(holder.get());
        LLAMA_LOG_INFO("ooc: installed basic scheduler via LLAMA_OOC_SCHED=%s\n", v);
        return;
    }

    // fallback: unknown value → install basic but warn
    static std::unique_ptr<llama_ooc_scheduler_basic> holder;
    holder.reset(new llama_ooc_scheduler_basic());
    llama_set_ooc_scheduler(holder.get());
    LLAMA_LOG_WARN("ooc: unknown LLAMA_OOC_SCHED='%s' → defaulting to 'basic'\n", v);
}
