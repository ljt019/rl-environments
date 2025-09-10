#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Installing verifiers and flash-attn..."
    uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation

    log_info "Installing dependencies in virtual environment..."
    uv sync

    log_info "Logging in to HuggingFace & WandB..."
    uv run hf auth login
    uv run wandb login

    log_info "Setup completed!"
}

main