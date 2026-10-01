#!/usr/bin/env bash
# ============================================================
# PDF Chat - macOS Startup Script
# Double-click this file in Finder to set up and launch the app.
# ============================================================

set -uo pipefail

# Move to the folder this script lives in, so it works no matter
# where the user double-clicks it from.
cd "$(dirname "$0")"

echo "================================================================"
echo "  PDF Chat - macOS Startup Script"
echo "================================================================"
echo ""
echo "[INFO] Working directory: $(pwd)"
echo ""

# ---------------------------------------------------------------
# Helper: keep the Terminal window open so the user can read any
# error messages, whether we exit successfully or not.
# ---------------------------------------------------------------
pause_on_exit() {
    echo ""
    echo "================================================================"
    echo "  Press Enter to close this window..."
    echo "================================================================"
    read -r
}
trap pause_on_exit EXIT

# ---------------------------------------------------------------
# 1. Verify Python is installed
# ---------------------------------------------------------------
echo "[STEP 1/6] Checking for Python..."
PYTHON_BIN=""
# The pinned dependencies need Python 3.12; prefer it when installed.
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "[ERROR] Python was not found on your system."
    echo "        Please install Python from https://www.python.org/downloads/"
    echo "        then double-click this file again."
    exit 1
fi
echo "[OK] Found $($PYTHON_BIN --version)"
echo ""

# ---------------------------------------------------------------
# 2. Create a virtual environment if one doesn't already exist
# ---------------------------------------------------------------
echo "[STEP 2/6] Checking for virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] No virtual environment found. Creating one now..."
    "$PYTHON_BIN" -m venv venv
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to create the virtual environment."
        exit 1
    fi
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi
echo ""

# ---------------------------------------------------------------
# 3. Activate the virtual environment
# ---------------------------------------------------------------
echo "[STEP 3/6] Activating virtual environment..."
# shellcheck disable=SC1091
source "venv/bin/activate"
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to activate the virtual environment."
    exit 1
fi
echo "[OK] Virtual environment activated."
echo ""

# ---------------------------------------------------------------
# 4. Install missing dependencies
# ---------------------------------------------------------------
echo "[STEP 4/6] Checking / installing dependencies (this may take a minute)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install dependencies from requirements.txt."
    echo "        Check your internet connection and try again."
    exit 1
fi
echo "[OK] Dependencies are up to date."
echo ""

# ---------------------------------------------------------------
# 5. Verify the .env file exists
# ---------------------------------------------------------------
echo "[STEP 5/6] Checking for .env configuration file..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[WARN] No .env file found. Creating one from .env.example..."
        cp ".env.example" ".env"
        echo "[WARN] IMPORTANT: Open the new .env file and set your OPENAI_API_KEY"
        echo "       before using chat/upload features. See INSTRUCTION.md for help."
    else
        echo "[ERROR] Neither .env nor .env.example was found. Cannot continue safely."
        exit 1
    fi
else
    echo "[OK] .env file found."
    if grep -q "OPENAI_API_KEY=sk-your-openai-api-key-here" ".env" 2>/dev/null; then
        echo "[WARN] Your .env file still has the placeholder OPENAI_API_KEY."
        echo "       Chat and upload features will fail until you set a real key."
        echo "       See INSTRUCTION.md, Section 11, for how to get one."
    fi
fi
echo ""

# ---------------------------------------------------------------
# 6. Launch the application
# ---------------------------------------------------------------
echo "[STEP 6/6] Starting PDF Chat..."
echo "================================================================"
echo "  The app will open at:  http://127.0.0.1:8000"
echo "  API docs available at: http://127.0.0.1:8000/docs"
echo "  Press CTRL+C in this window to stop the server."
echo "================================================================"
echo ""

# Open the app in the default browser shortly after the server starts.
( sleep 2 && open "http://127.0.0.1:8000" ) &

uvicorn main:app --host 127.0.0.1 --port 8000 --reload

echo ""
echo "================================================================"
echo "  The server has stopped."
echo "================================================================"
