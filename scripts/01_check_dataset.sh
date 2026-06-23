#!/bin/bash
# ============================================================================
# Step 1: Sanity-check that your LeRobot v2 dataset has the expected layout
# before conversion. Read-only.
#   source scripts/config.env && bash scripts/01_check_dataset.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

fail=0
echo ">> Checking dataset at: $DATA_ROOT"
[ -d "$DATA_ROOT/data" ]   || { echo "  MISSING: data/";   fail=1; }
[ -d "$DATA_ROOT/videos" ] || { echo "  MISSING: videos/"; fail=1; }
[ -f "$DATA_ROOT/meta/info.json" ] || { echo "  MISSING: meta/info.json"; fail=1; }

if [ -f "$DATA_ROOT/meta/info.json" ]; then
    echo ">> info.json keys (need: features, total_episodes, fps):"
    $PY - "$DATA_ROOT/meta/info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
for k in ("features", "total_episodes", "fps"):
    print(f"   {k}: {'OK' if k in info else 'MISSING'}")
feats = info.get("features", {})
cams = [k for k in feats if k.startswith("observation.images.")]
print("   camera streams found:", cams)
print("   total_episodes:", info.get("total_episodes"))
print("   fps:", info.get("fps"))
PY
fi

if [ "$fail" -ne 0 ]; then
    echo ">> FAILED: dataset is not a valid LeRobot v2 layout. See README for the expected tree."
    exit 1
fi
echo ">> Dataset layout looks OK."
