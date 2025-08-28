#pragma once

// Per-expert on-demand load/evict helpers (testing/prototyping)
// These operate at expert granularity for the MoE weight tensors in a given layer.

struct llama_model;

// Load expert `expert_id` for layer `il` into the model's current weight buffers (CPU/GPU as allocated).
// Returns true on success.
bool llama_expert_load(struct llama_model * model, int il, int expert_id);

// Evict expert `expert_id` for layer `il` by stashing its current bytes to a cache file and zeroing its slice in-place.
// Returns true on success.
bool llama_expert_evict(struct llama_model * model, int il, int expert_id);

