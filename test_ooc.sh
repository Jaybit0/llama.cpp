export LLAMA_OOC_SCHED=1
export LLAMA_OOC_DISABLE_UNUSED=1
export LLAMA_OOC_WARMUP_TOKENS=1
export LLAMA_GRAPH_REUSE_DISABLE=1
export LLAMA_OOC_EXPERTS=1
./build/bin/llama-cli -m /Users/janniklindemann/Dev/llms/qwen1.5-moe/qwen1.5-moe-a2.7b-chat-q3_k_s.gguf -p "Hello world!" -n 32