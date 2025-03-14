unset http_proxy
unset https_proxy
model_id=/mnt/data/renyuming/model-qwen2-7b-instruct-awq
model_name_ext=llm
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server  --model ${model_id}  --served-model-name ${model_name_ext} --trust-remote-code --port 4128 --api-key 123456 --max-num-seqs 10 --max-model-len 2048 --gpu-memory-utilization 0.9 --disable-log-requests