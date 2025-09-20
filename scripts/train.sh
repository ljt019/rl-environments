#!/usr/bin/env bash

############## Training Config ##############

CUDA_VISIBLE_DEVICES="2,3"

#############################################

main() {
    # Get script path from argument, default to scripts/grpo.py
    SCRIPT_PATH=${1:-"scripts/grpo.py"}
    
    # Check if script exists
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script '$SCRIPT_PATH' not found!"
        echo "Usage: $0 [script_path]"
        echo "Examples:"
        echo "  $0                                    # Uses default scripts/grpo.py"
        echo "  $0 environments/battleship/scripts/grpo.py"
        echo "  $0 my_custom_training.py"
        exit 1
    fi
    
    # Count number of processes (number of GPUs)
    NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    
    echo "Running training with ${NUM_PROCESSES} processes on devices: ${CUDA_VISIBLE_DEVICES}"
    echo "Script: ${SCRIPT_PATH}"
    
    # Add the project root to PYTHONPATH so imports work
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run accelerate launch \
        --config-file configs/zero3.yaml \
        --num-processes ${NUM_PROCESSES} \
        "${SCRIPT_PATH}"
}

main "$@"