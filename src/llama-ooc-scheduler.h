#pragma once

#include <vector>
#include <utility>

#include "ggml-backend.h"

struct llama_model;
struct llama_ubatch;
struct ggml_tensor;

// Lightweight hook interface for out-of-core (OOC) scheduling.
// Implementors can override buffer placement during model load and observe graph builds
// to implement prefetch/evict policies for MoE experts.
class llama_ooc_scheduler_i {
public:
    virtual ~llama_ooc_scheduler_i() = default;

    // Called once the model finishes loading all tensors successfully.
    // Use this to index important tensors (e.g., MoE experts) you want to manage.
    virtual void on_model_loaded(const llama_model & /*model*/) {}

    // Called during model load when selecting a backend buffer type for a weight tensor.
    // Return a non-null buffer type to override placement, or nullptr to accept the default.
    // - tensor_name: fully qualified GGUF tensor name (e.g. "blk.12.ffn_up_exps.weight")
    // - layer_index: repeating layer index, or -1 for input/output tensors
    // - usage_op: ggml op expected to use this tensor (e.g., GGML_OP_MUL_MAT_ID for MoE experts)
    // - assigned_dev: device chosen for this layer
    // - candidates: ordered list of candidate (device, buffer type) pairs considered
    // - suggested: buffer type selected by llama.cpp before override
    virtual ggml_backend_buffer_type_t override_weight_buffer_type(
        const llama_model & /*model*/, const char * /*tensor_name*/, int /*layer_index*/, int /*usage_op*/,
        ggml_backend_dev_t /*assigned_dev*/, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & /*candidates*/, ggml_backend_buffer_type_t /*suggested*/) {
        return nullptr;
    }

    // Called for each named tensor node while building the compute graph.
    // Use this to steer per-node backend assignment or to trigger prefetch/eviction decisions.
    virtual void on_graph_tensor(const llama_model & /*model*/, const llama_ubatch & /*ubatch*/, ggml_backend_sched_t /*sched*/, ggml_tensor * /*node*/, const char * /*name*/, int /*layer_index*/) {}

    // Called by the scheduler during graph execution for each node when an eval callback is installed.
    // ask == true: query if you intend to observe this node (for batching decisions). Most implementations can ignore.
    // ask == false: node has been scheduled for observation; you may inspect results (e.g., expert selection indices)
    // Note: do not perform heavy synchronous work here; prefer scheduling async prefetch.
    virtual void on_eval_node(const llama_model & /*model*/, ggml_backend_sched_t /*sched*/, ggml_tensor * /*node*/, bool /*ask*/) {}
};

// Global registration for a single scheduler instance.
// llama.cpp will check for nullptr and fall back to built-in behavior if unset.
void llama_set_ooc_scheduler(llama_ooc_scheduler_i * sched);
llama_ooc_scheduler_i * llama_get_ooc_scheduler();

// Optional: auto-register a simple scheduler from environment variable.
// If env var LLAMA_OOC_SCHED is set (e.g. to "basic" or "1"),
// llama.cpp will install a simple default scheduler.
void llama_ooc_init_from_env();
