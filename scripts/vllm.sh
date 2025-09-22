#!/usr/bin/env bash

############## vLLM Config ##############

MODEL_NAME="ljt019/Qwen2.5-Coder-1.5B-Instruct-rust"
CUDA_VISIBLE_DEVICES="1"

##############################################

main() {
    # Count number of processes (number of GPUs)
    DATA_PARALLEL_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    
    echo "Running vLLM with ${DATA_PARALLEL_SIZE} GPUs on devices: ${CUDA_VISIBLE_DEVICES}"
    
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run vf-vllm \
        --model ${MODEL_NAME} \
        --data-parallel-size ${DATA_PARALLEL_SIZE} \
        --enforce-eager \
        --disable-log-requests
}

main "$@"
