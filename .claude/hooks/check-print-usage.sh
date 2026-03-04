#!/bin/bash
# Hook: Discourage print() in source code, suggest logger instead
# Runs as PreToolUse hook on Edit/Write

INPUT=$(cat <<'ENDINPUT'
$TOOL_INPUT
ENDINPUT
)

FILE=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)
CONTENT=$(echo "$TOOL_INPUT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(d.get('content','') + d.get('new_string',''))
" 2>/dev/null)

# Only check .py files in src/ or packages/ (not tests, not notebooks)
if echo "$FILE" | grep -qE '\.(py)$' && \
   echo "$FILE" | grep -qE '(src/|packages/)' && \
   ! echo "$FILE" | grep -qE '(tests/|test_|notebook)' && \
   echo "$CONTENT" | grep -qE 'print\('; then
  echo "[Hook] Avoid print() in source code. Use logger instead: from src.utils.logger import setup_logger"
fi
