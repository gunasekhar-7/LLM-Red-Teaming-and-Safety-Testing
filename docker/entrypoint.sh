#!/bin/bash
# Entrypoint for LLM Red-Teaming test runner container
# Handles initialization, environment validation, and command routing

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if running as root (should not happen)
if [ "$(id -u)" = "0" ]; then
    log_error "Container should not run as root"
    exit 1
fi

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check Python availability
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 1
    fi
    
    # Check critical imports
    python -c "
import sys
sys.path.insert(0, '/app/src')
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

# Initialize database if needed
init_database() {
    if [ "${INIT_DB:-false}" = "true" ]; then
        log_info "Initializing database..."
        python -c "
import sys
sys.path.insert(0, '/app/apps/reference_rag')
from reference_rag.database import init_db
import asyncio
asyncio.run(init_db())
" && log_success "Database initialized" || log_warning "Database init skipped/failed"
    fi
}

# Wait for dependencies
wait_for_dependencies() {
    log_info "Waiting for dependencies..."
    
    # Wait for PostgreSQL
    if [ -n "${DATABASE_URL:-}" ]; then
        log_info "Waiting for PostgreSQL..."
        for i in {1..30}; do
            if python -c "
import sys
import os
import asyncpg
import asyncio
async def check():
    try:
        conn = await asyncpg.connect(os.environ['DATABASE_URL'])
        await conn.close()
        return True
    except:
        return False
result = asyncio.run(check())
sys.exit(0 if result else 1)
" 2>/dev/null; then
                log_success "PostgreSQL is ready"
                break
            fi
            sleep 2
        done
    fi
    
    # Wait for Ollama
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

# Print banner
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         LLM Red-Teaming Framework - Test Runner             ║"
    echo "║         Version: 0.1.0                                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

# Main execution
main() {
    print_banner
    validate_environment
    wait_for_dependencies
    init_database
    
    # If no arguments provided, show help
    if [ $# -eq 0 ]; then
        python -m llm_redteam_cli --help
        exit 0
    fi
    
    # Execute command
    log_info "Executing: $*"
    exec python -m llm_redteam_cli "$@"
}

# Run main
main "$@"