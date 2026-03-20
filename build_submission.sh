#!/usr/bin/env bash
#
# Build and validate a submission.zip for the NorgesGruppen competition.
#
# Usage: ./build_submission.sh [--weights best.pt] [--output submission.zip]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS="${1:-$SCRIPT_DIR/best.pt}"
OUTPUT="$(cd "$SCRIPT_DIR" && pwd)/${2:-submission.zip}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

fail() { echo -e "${RED}FAIL:${NC} $1"; exit 1; }
pass() { echo -e "${GREEN}PASS:${NC} $1"; }
warn() { echo -e "${YELLOW}WARN:${NC} $1"; }

echo "============================================"
echo "  NorgesGruppen Submission Builder"
echo "============================================"
echo ""

# --- 1. Check required files exist ---
echo "--- Checking required files ---"
[ -f "$SCRIPT_DIR/run.py" ] && pass "run.py exists" || fail "run.py not found"
[ -f "$WEIGHTS" ] && pass "Weights file exists: $WEIGHTS" || fail "Weights not found: $WEIGHTS"

# --- 2. Validate run.py constraints ---
echo ""
echo "--- Validating run.py ---"

# --- Security scan (blocked imports) ---
BLOCKED_IMPORTS="os|sys|subprocess|socket|ctypes|builtins|importlib|pickle|marshal|shelve|shutil|yaml|requests|urllib|http\.client|multiprocessing|threading|signal|gc|code|codeop|pty"
SECURITY_FAIL=0
for pyfile in "$SCRIPT_DIR"/*.py; do
    fname=$(basename "$pyfile")
    [[ "$fname" == "train.py" || "$fname" == "evaluate.py" || "$fname" == "train_experiment.py" || "$fname" == "eval_quick.py" ]] && continue
    # Check for blocked imports
    if grep -nE "^import ($BLOCKED_IMPORTS)$|^from ($BLOCKED_IMPORTS)" "$pyfile" 2>/dev/null; then
        fail "Security: $fname contains blocked import (see above)"
        SECURITY_FAIL=1
    fi
    # Check for blocked calls
    if grep -nE "\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\b__import__\s*\(" "$pyfile" 2>/dev/null; then
        fail "Security: $fname contains blocked call (eval/exec/compile/__import__)"
        SECURITY_FAIL=1
    fi
done
if [ "$SECURITY_FAIL" -eq 0 ]; then
    pass "Security scan: no blocked imports or calls"
fi

# Has --input and --output args
if grep -q "\-\-input" "$SCRIPT_DIR/run.py" && grep -q "\-\-output" "$SCRIPT_DIR/run.py"; then
    pass "Has --input and --output arguments"
else
    fail "Missing --input or --output arguments"
fi

# Loads model relative to script (not hardcoded path)
if grep -q '__file__' "$SCRIPT_DIR/run.py" || grep -qE 'YOLO\("best.pt"\)' "$SCRIPT_DIR/run.py"; then
    pass "Model loaded relative to script or by name"
else
    warn "Check model loading path — should be relative to run.py"
fi

# Output format: writes JSON
if grep -q "json.dump" "$SCRIPT_DIR/run.py"; then
    pass "Writes JSON output"
else
    fail "No json.dump found — output format incorrect"
fi

# COCO bbox format [x, y, w, h]
if grep -qE "x2\s*-\s*x1|x2-x1" "$SCRIPT_DIR/run.py"; then
    pass "COCO bbox format (x, y, w, h)"
else
    warn "Could not verify COCO bbox conversion"
fi

# --- 3. Count .py files ---
echo ""
echo "--- Checking file counts ---"
PY_COUNT=$(find "$SCRIPT_DIR" -maxdepth 1 -name "*.py" -not -name "train.py" -not -name "evaluate.py" -not -name "train_experiment.py" -not -name "eval_quick.py" | wc -l | tr -d ' ')
if [ "$PY_COUNT" -le 10 ]; then
    pass ".py file count: $PY_COUNT (max 10)"
else
    fail "Too many .py files: $PY_COUNT (max 10)"
fi

# --- 4. Build the zip ---
echo ""
echo "--- Building submission.zip ---"
TMPDIR=$(mktemp -d)
cp "$SCRIPT_DIR/run.py" "$TMPDIR/"
cp "$WEIGHTS" "$TMPDIR/best.pt"

# Copy any additional .py files (not train/evaluate/build scripts)
for f in "$SCRIPT_DIR"/*.py; do
    fname=$(basename "$f")
    if [[ "$fname" != "run.py" && "$fname" != "train.py" && "$fname" != "evaluate.py" && "$fname" != "train_experiment.py" && "$fname" != "eval_quick.py" ]]; then
        cp "$f" "$TMPDIR/"
    fi
done

cd "$TMPDIR"
rm -f "$OUTPUT"
zip -q -j "$OUTPUT" *
cd "$SCRIPT_DIR"

# --- 5. Validate zip ---
echo ""
echo "--- Validating submission.zip ---"
ZIP_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat --printf="%s" "$OUTPUT" 2>/dev/null)
ZIP_SIZE_MB=$((ZIP_SIZE / 1024 / 1024))

if [ "$ZIP_SIZE_MB" -le 420 ]; then
    pass "Zip size: ${ZIP_SIZE_MB}MB (max 420MB)"
else
    fail "Zip too large: ${ZIP_SIZE_MB}MB (max 420MB)"
fi

# Check zip contains required files
if zipinfo -1 "$OUTPUT" | grep -q "run.py"; then
    pass "run.py in zip"
else
    fail "run.py missing from zip"
fi

if zipinfo -1 "$OUTPUT" | grep -q "best.pt"; then
    pass "best.pt in zip"
else
    fail "best.pt missing from zip"
fi

ZIP_PY_COUNT=$(zipinfo -1 "$OUTPUT" | grep -c "\.py$" || true)
if [ "$ZIP_PY_COUNT" -le 10 ]; then
    pass ".py files in zip: $ZIP_PY_COUNT (max 10)"
else
    fail "Too many .py files in zip: $ZIP_PY_COUNT"
fi

# --- 6. Dry-run test (unzip to temp, verify structure) ---
echo ""
echo "--- Dry-run extraction test ---"
TESTDIR=$(mktemp -d)
unzip -q "$OUTPUT" -d "$TESTDIR"
if [ -f "$TESTDIR/run.py" ] && [ -f "$TESTDIR/best.pt" ]; then
    pass "Extracted structure valid"
else
    fail "Extraction missing required files"
fi

# Quick syntax check
if python3 -c "import py_compile; py_compile.compile('$TESTDIR/run.py', doraise=True)" 2>/dev/null; then
    pass "run.py passes syntax check"
else
    fail "run.py has syntax errors"
fi

# Cleanup
rm -rf "$TMPDIR" "$TESTDIR"

echo ""
echo "============================================"
echo -e "${GREEN}  Submission ready: $OUTPUT${NC}"
echo -e "  Size: ${ZIP_SIZE_MB}MB"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Upload to competition platform"
echo "  2. Max 3 submissions/day"
echo ""
