#!/bin/bash
# Entrypoint for LLM Red-Teaming test runner container
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [ "$(id -u)" = "0" ]; then
    log_error "Container should not run as root"
    exit 1
fi

export PYTHONPATH="/app/src:/app/libs/rag-target/src:/app/libs/llm-provider/src:/app/libs/guards/src:/app/libs/test-harness/src:/app/apps/reference-rag/src:/app/apps/existing-rag-adapter/src:${PYTHONPATH:-}"

validate_environment() {
    log_info "Validating environment..."
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 1
    fi

    python -c "
import sys
try:
    from llm_redteam_cli import app
    from test_harness import TestHarness
    from guards import create_default_pipelines
    print('All imports OK')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
" || exit 1

    log_success "Environment validation passed"
}

init_database() {
    if [ "${INIT_DB:-false}" = "true" ]; then
        log_info "Initializing database..."
        python -c "
import asyncio
from reference_rag.database import init_db
asyncio.run(init_db())
" && log_success "Database initialized" || log_warning "Database init skipped/failed"
    fi
}

wait_for_dependencies() {
    log_info "Waiting for dependencies..."

    if [ -n "${DATABASE_URL:-}" ]; then
        log_info "Waiting for PostgreSQL..."
        for i in {1..30}; do
            if python -c "
import os, sys, asyncio, asyncpg
async def check():
    try:
        dsn = os.environ['DATABASE_URL'].replace('postgresql+asyncpg://', 'postgresql://')
        conn = await asyncpg.connect(dsn)
        await conn.close()
        return True
    except Exception:
        return False
sys.exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; then
                log_success "PostgreSQL is ready"
                break
            fi
            sleep 2
        done
    fi

    if [ -n "${OLLAMA_BASE_URL:-}" ]; then
        log_info "Waiting for Ollama..."
        for i in {1..30}; do
            if curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
                log_success "Ollama is ready"
                break
            fi
            sleep 2
        done
    fi
}

print_banner() {
    echo ""
    echo "================================================================"
    echo "         LLM Red-Teaming Framework - Test Runner"
    echo "         Version: 0.1.0"
    echo "================================================================"
    echo ""
}

main() {
    print_banner
    validate_environment
    wait_for_dependencies
    init_database

    if [ $# -eq 0 ]; then
        python -m llm_redteam_cli --help
        exit 0
    fi

    log_info "Executing: $*"
    exec python -m llm_redteam_cli "$@"
}

main "$@"
