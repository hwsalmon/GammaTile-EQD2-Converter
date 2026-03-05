#!/usr/bin/env bash
# launch.sh — GammaTile EQD2 Converter launcher
#
# Ensures PySide6's bundled Qt libraries take priority over any older system Qt6,
# preventing the "Qt_6_PRIVATE_API not found" shared-library version conflict.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the PySide6 bundled Qt libs (works for both venv and user installs)
PYSIDE6_LIB="$(python3 -c "
import importlib.util, os, sys
spec = importlib.util.find_spec('PySide6')
if spec:
    print(os.path.join(os.path.dirname(spec.origin), 'Qt', 'lib'))
" 2>/dev/null)"

if [ -n "$PYSIDE6_LIB" ] && [ -d "$PYSIDE6_LIB" ]; then
    export LD_LIBRARY_PATH="$PYSIDE6_LIB:${LD_LIBRARY_PATH:-}"
fi

cd "$SCRIPT_DIR"
exec python3 viewer.py "$@"
