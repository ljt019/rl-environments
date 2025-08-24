#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    echo "Usage: $0 <environment> [--key value] [--key value] ..."
    echo ""
    echo "Arguments:"
    echo "  environment        Environment name (required)"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  --key value        Environment arguments (passed as JSON to vf-eval)"
    echo ""
    echo "Examples:"
    echo "  $0 battleship"
    echo "  $0 battleship --max_turns 10 --num_games 100"
    exit 1
}

main() {
    local environment=""
    local env_args=()
    
    # Check for help first
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
    fi
    
    # First argument must be environment name
    if [[ $# -lt 1 ]]; then
        log_error "Environment name is required"
        usage
    fi
    
    environment="$1"
    shift
    
    # Parse remaining arguments as key-value pairs
    while [[ $# -gt 0 ]]; do
        if [[ "$1" =~ ^-- ]]; then
            # Remove -- prefix
            key="${1#--}"
            if [[ $# -lt 2 ]]; then
                log_error "Missing value for --$key"
                usage
            fi
            value="$2"
            
            # Try to detect if value is numeric
            if [[ "$value" =~ ^[0-9]+$ ]]; then
                env_args+=("\"$key\": $value")
            else
                env_args+=("\"$key\": \"$value\"")
            fi
            shift 2
        else
            log_error "Unknown argument: $1"
            usage
        fi
    done
    
    # Build the command
    cmd="uv run vf-eval ${environment} -b 'https://openrouter.ai/api/v1' -m 'qwen/qwen3-235b-a22b-2507'"
    
    # Add env-args if any
    if [[ ${#env_args[@]} -gt 0 ]]; then
        # Join array elements with commas
        env_args_json="{$(IFS=', '; echo "${env_args[*]}")}"
        cmd="$cmd -a '$env_args_json'"
        log_info "Running eval for environment: $environment with args: $env_args_json"
    else
        log_info "Running eval for environment: $environment"
    fi
    
    # Execute the command
    eval "$cmd"
    
    # Keep window open to view results
    log_info "Evaluation completed. Press any key to close..."
    read -n 1 -s
}

main "$@"