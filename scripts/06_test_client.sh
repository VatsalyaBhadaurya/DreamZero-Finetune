#!/bin/bash
# ============================================================================
# Step 6: Hit the running inference server with the test client.
#   source scripts/config.env && bash scripts/06_test_client.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
cd "$DREAMZERO_ROOT"

PORT="${PORT:-5000}"
python test_client_AR.py --port "$PORT"
