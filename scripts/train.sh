#!/usr/bin/env bash

############## Training Config ##############

CUDA_VISIBLE_DEVICES="1"

#############################################

main() {
    # Get environment name from argument
    ENV_NAME=${1:-""}
    
    if [[ -z "$ENV_NAME" ]]; then
        echo "Error: Environment name required!"
    fi
    
    # Construct the script path
    SCRIPT_PATH="environments/${ENV_NAME}/scripts/grpo.py"
    
    # Check if script exists
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script '$SCRIPT_PATH' not found!"
        echo "Make sure the environment '${ENV_NAME}' exists and has a scripts/grpo.py file."
        echo "Available environments:"
        ls -1 environments/ 2>/dev/null | grep -v __pycache__ || echo "  (none found)"
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