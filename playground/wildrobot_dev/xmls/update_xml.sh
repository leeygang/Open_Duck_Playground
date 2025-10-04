#!/usr/bin/env bash
# Save as: update_xml.sh

set -e  # Exit on error

echo "=============================="
echo "onshape-to-robot Pipeline"
echo "Platform: $(uname -s)"
echo "=============================="

# Check if commands exist
command -v onshape-to-robot >/dev/null 2>&1 || { 
    echo "Error: onshape-to-robot not found. Install with: pip install onshape-to-robot"
    exit 1
}

# Detect Python command
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Step 1: Run onshape-to-robot
echo ""
echo "Running onshape-to-robot..."
onshape-to-robot .

# Step 2: Run post-process
echo ""
echo "Running post_process.py..."
$PYTHON_CMD post_process.py

echo ""
echo "✓ Pipeline completed successfully!"