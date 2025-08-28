#include "llama-ooc-scheduler.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <algorithm>

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
        // For testing, just log MoE nodes once in a while to confirm activation
        if (name && std::strstr(name, "ffn_moe_")) {
            LLAMA_LOG_DEBUG("ooc/basic: saw node %s (il=%d)\n", name, il);
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
