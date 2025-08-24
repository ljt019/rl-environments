#!/usr/bin/env bash

############## Training Config ##############

CUDA_VISIBLE_DEVICES="1"

#############################################

main() {
    # Count number of processes (number of GPUs)
    NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    
    echo "Running training with ${NUM_PROCESSES} processes on devices: ${CUDA_VISIBLE_DEVICES}"
    
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
        --config-file configs/zero3.yaml \
        --num-processes ${NUM_PROCESSES} \
        scripts/grpo.py
}

main "$@"
