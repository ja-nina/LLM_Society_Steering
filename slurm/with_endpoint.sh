#!/bin/bash
# Run a command from this repo against the vLLM server that `swmp submit` (Social_World_Models_Personas) starts.
# swmp writes the server URL to $SWMP_ENDPOINTS; the OpenAI client of lib/datagen reads OPENAI_BASE_URL, so the
# generation code runs unchanged.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL=$(python3 -c "import json,os;print(json.load(open(os.environ['SWMP_ENDPOINTS']))['urls'][0])")
export OPENAI_BASE_URL="$URL" OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" PYTHONUNBUFFERED=1
cd "$HERE"
echo "[with_endpoint] OPENAI_BASE_URL=$OPENAI_BASE_URL cwd=$HERE cmd=$*"
exec "$@"
