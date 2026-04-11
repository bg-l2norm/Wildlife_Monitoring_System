#!/usr/bin/env bash
set -e

echo "======================================"
echo " Starting setup for Wildlife Monitoring"
echo "======================================"

# 1. Check for Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 could not be found. Please install Python 3."
    exit 1
fi

$PYTHON_CMD -c 'import sys; exit(1) if sys.version_info < (3, 10) else exit(0)' || {
    echo "Warning: Python 3.10+ is recommended."
}

# 2. Create/Find Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# 3. Activate it
echo "Activating virtual environment..."
source venv/bin/activate

# 4. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create default .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cat <<EOF > .env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ROBOFLOW_API_KEY=your_api_key
EOF
else
    echo ".env file already exists."
fi

# 6. Doctor checkup
echo ""
echo "======================================"
echo " Running Doctor Checkup"
echo "======================================"

DOCTOR_PASSED=true

# Check Python version
PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[OK] Python version: $PY_VERSION"

# Check Venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "[OK] Virtual environment active: $VIRTUAL_ENV"
else
    echo "[FAIL] Virtual environment is not active!"
    DOCTOR_PASSED=false
fi

# Check requirements
echo "Checking core dependencies..."
python -c 'import flask' 2>/dev/null && echo "[OK] Flask is installed" || { echo "[FAIL] Flask is missing"; DOCTOR_PASSED=false; }
python -c 'import torch' 2>/dev/null && echo "[OK] PyTorch is installed" || { echo "[FAIL] PyTorch is missing"; DOCTOR_PASSED=false; }
python -c 'import ultralytics' 2>/dev/null && echo "[OK] Ultralytics is installed" || { echo "[FAIL] Ultralytics is missing"; DOCTOR_PASSED=false; }

if [ "$DOCTOR_PASSED" = true ]; then
    echo "======================================"
    echo " Setup complete! Everything looks good."
    echo " To start the app, run:"
    echo "   source venv/bin/activate"
    echo "   python app.py"
    echo "======================================"
else
    echo "======================================"
    echo " Doctor checkup found issues. Please review above."
    echo "======================================"
    exit 1
fi
