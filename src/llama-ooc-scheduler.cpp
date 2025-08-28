#include "llama-ooc-scheduler.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdint>

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
    ggml_backend_buffer_type_t override_weight_buffer_type(
        const llama_model & /*model*/, const char * tensor_name, int /*layer_index*/, int usage_op,
        ggml_backend_dev_t /*assigned_dev*/, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & candidates,
        ggml_backend_buffer_type_t suggested) override {

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

    void on_eval_node(const llama_model & /*model*/, ggml_backend_sched_t sched, ggml_tensor * node, bool ask) override {
        if (ask) return; // post-compute observation follows when ask == false
        const char * name = ggml_get_name(node);
        if (!name) return;
        const bool is_args = std::strstr(name, "ffn_moe_argsort") != nullptr;
        const bool is_topk = std::strstr(name, "ffn_moe_topk")    != nullptr;
        if (!(is_args || is_topk)) return;

        // fetch raw bytes
        std::vector<uint8_t> buf;
        if (!fetch_bytes(sched, node, buf)) {
            return;
        }

        const int64_t ne0 = node->ne[0]; // n_expert_used
        const int64_t ne1 = node->ne[1]; // n_tokens
        const size_t  nb0 = node->nb[0]; // stride bytes expert index
        const size_t  nb1 = node->nb[1]; // stride bytes token

        // Log only the first token column to avoid spam in batched cases
        const int64_t tok = 0;

        if (is_args && node->type == GGML_TYPE_I32) {
            const uint8_t * base = buf.data();
            std::vector<int32_t> ids;
            ids.reserve((size_t) ne0);
            for (int64_t k = 0; k < ne0; ++k) {
                const int32_t * p = reinterpret_cast<const int32_t *>(base + tok*nb1 + k*nb0);
                ids.push_back(*p);
            }
            // print indices as [e0,e1,...]
            std::string s = "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                s += std::to_string(ids[i]);
                if (i + 1 < ids.size()) s += ",";
            }
            s += "]";
            LLAMA_LOG_INFO("ooc/basic: %s layer=%s experts=%s\n", name, std::strrchr(name, '-') ? std::strrchr(name, '-') + 1 : "?", s.c_str());
        } else if (is_topk) {
            std::string s;

            if (node->type == GGML_TYPE_F32) {
                const uint8_t * base = buf.data();
                std::vector<float> vals;
                vals.reserve((size_t) ne0);
                for (int64_t k = 0; k < ne0; ++k) {
                    const float * p = reinterpret_cast<const float *>(base + tok*nb1 + k*nb0);
                    vals.push_back(*p);
                }
                // print probs as [p0,p1,...]
                s = "[";
                for (size_t i = 0; i < vals.size(); ++i) {
                    char tmp[32];
                    std::snprintf(tmp, sizeof(tmp), "%.4f", vals[i]);
                    s += tmp;
                    if (i + 1 < vals.size()) s += ",";
                }
                s += "]";
            } else if (node->type == GGML_TYPE_I32) {
                const uint8_t * base = buf.data();
                std::vector<int32_t> ids;
                ids.reserve((size_t) ne0);
                for (int64_t k = 0; k < ne0; ++k) {
                    const int32_t * p = reinterpret_cast<const int32_t *>(base + tok*nb1 + k*nb0);
                    ids.push_back(*p);
                }
                // print indices as [e0,e1,...]
                s = "[";
                for (size_t i = 0; i < ids.size(); ++i) {
                    s += std::to_string(ids[i]);
                    if (i + 1 < ids.size()) s += ",";
                }
                s += "]";
            } else {
                // Print unsupported node type
                LLAMA_LOG_INFO("ooc/basic: %s layer=%s unsupported type=%s\n", name, std::strrchr(name, '-') ? std::strrchr(name, '-') + 1 : "?", ggml_type_name(node->type));
                return;
            }

            LLAMA_LOG_INFO("ooc/basic: %s layer=%s scores=%s\n", name, std::strrchr(name, '-') ? std::strrchr(name, '-') + 1 : "?", s.c_str());
        }
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
