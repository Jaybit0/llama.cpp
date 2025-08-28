#include "llama-expert-ooc.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool make_dirs(const std::string & path) {
    // naive mkdir -p for up to 10 levels
    size_t pos = 1;
    while (pos < path.size()) {
        pos = path.find('/', pos);
        if (pos == std::string::npos) pos = path.size();
        std::string sub = path.substr(0, pos);
        if (!sub.empty()) {
            if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        ++pos;
    }
    return true;
}

static std::string expert_cache_dir(const llama_model * model, int il) {
    // Use model name; caller ensures uniqueness per test
    std::string base = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.cache/llama_experts/";
    base += model->name;
    base += "/layer_" + std::to_string(il);
    return base;
}

static bool write_file(const std::string & path, const void * data, size_t size) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(data, 1, size, f);
    fclose(f);
    return w == size;
}

static bool read_file(const std::string & path, void * data, size_t size) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t r = fread(data, 1, size, f);
    fclose(f);
    return r == size;
}

static bool evict_one(ggml_tensor * t, int expert_id, const std::string & dir, const char * tag) {
    if (!t) return true; // optional tensor
    const size_t slice_off   = (size_t) expert_id * (size_t) t->nb[2];
    const size_t slice_bytes = (size_t) t->nb[2];
    std::vector<uint8_t> buf(slice_bytes);
    ggml_backend_tensor_get(t, buf.data(), slice_off, slice_bytes);

    std::string path = dir + "/" + tag + "_e" + std::to_string(expert_id) + ".bin";
    if (!write_file(path, buf.data(), buf.size())) {
        LLAMA_LOG_WARN("ooc/evict: failed to write %s\n", path.c_str());
        return false;
    }
    // only zero if buffer is a writable CPU buffer; Metal-mapped buffers may be read-only and cause SIGBUS
    if (t->buffer) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                ggml_backend_tensor_memset(t, 0, slice_off, slice_bytes);
            }
        }
    }
    return true;
}

static bool load_one(ggml_tensor * t, int expert_id, const std::string & dir, const char * tag) {
    if (!t) return true; // optional tensor
    const size_t slice_off   = (size_t) expert_id * (size_t) t->nb[2];
    const size_t slice_bytes = (size_t) t->nb[2];
    std::vector<uint8_t> buf(slice_bytes);
    std::string path = dir + "/" + tag + "_e" + std::to_string(expert_id) + ".bin";
    if (!read_file(path, buf.data(), buf.size())) {
        LLAMA_LOG_WARN("ooc/load: missing %s (expert not previously evicted?)\n", path.c_str());
        return false;
    }
    // only write back if buffer is a writable CPU buffer; otherwise, skip (mmap will page-in on demand when used)
    if (t->buffer) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                ggml_backend_tensor_set(t, buf.data(), slice_off, slice_bytes);
            }
        }
    }
    return true;
}

bool llama_expert_evict(struct llama_model * model, int il, int expert_id) {
    if (il < 0 || il >= (int) model->hparams.n_layer) return false;
    auto & L = model->layers[il];
    // must be MoE layer
    if (!L.ffn_up_exps && !L.ffn_gate_exps && !L.ffn_down_exps) return true;
    std::string dir = expert_cache_dir(model, il);
    if (!make_dirs(dir)) {
        LLAMA_LOG_WARN("ooc/evict: failed to create %s\n", dir.c_str());
        return false;
    }
    bool ok = true;
    ok &= evict_one(L.ffn_up_exps,   expert_id, dir, "up");
    ok &= evict_one(L.ffn_gate_exps, expert_id, dir, "gate");
    ok &= evict_one(L.ffn_down_exps, expert_id, dir, "down");
    return ok;
}

bool llama_expert_load(struct llama_model * model, int il, int expert_id) {
    if (il < 0 || il >= (int) model->hparams.n_layer) return false;
    auto & L = model->layers[il];
    if (!L.ffn_up_exps && !L.ffn_gate_exps && !L.ffn_down_exps) return true;
    std::string dir = expert_cache_dir(model, il);
    bool ok = true;
    ok &= load_one(L.ffn_up_exps,   expert_id, dir, "up");
    ok &= load_one(L.ffn_gate_exps, expert_id, dir, "gate");
    ok &= load_one(L.ffn_down_exps, expert_id, dir, "down");
    return ok;
}
