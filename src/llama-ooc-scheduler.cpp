#include "llama-ooc-scheduler.h"

#include <atomic>

// Global singleton pointer (non-owning). Users manage lifetime externally.
static std::atomic<llama_ooc_scheduler_i *> g_llama_ooc_sched{nullptr};

void llama_set_ooc_scheduler(llama_ooc_scheduler_i * sched) {
    g_llama_ooc_sched.store(sched, std::memory_order_release);
}

llama_ooc_scheduler_i * llama_get_ooc_scheduler() {
    return g_llama_ooc_sched.load(std::memory_order_acquire);
}

