#!/bin/bash
# Hook: Warn when modifying test files without approval
# Runs as PreToolUse hook on Edit/Write

FILE=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)

if echo "$FILE" | grep -qE 'tests/|test_'; then
  echo "[Hook] Modifying test file: $(basename "$FILE"). Make sure the user approved this change."
fi
