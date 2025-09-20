#!/usr/bin/env bash

############## vLLM Config ##############

MODEL_NAME="willcb/Qwen3-4B"
CUDA_VISIBLE_DEVICES="0,1"

##############################################

main() {
    # Count number of processes (number of GPUs)
    DATA_PARALLEL_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    
    echo "Running vLLM with ${DATA_PARALLEL_SIZE} GPUs on devices: ${CUDA_VISIBLE_DEVICES}"
    
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run vf-vllm \
        --model ${MODEL_NAME} \
        --data-parallel-size ${DATA_PARALLEL_SIZE} \
        --max-model-len 65536 \
        --enforce-eager \
        --disable-log-requests
}

main "$@"
