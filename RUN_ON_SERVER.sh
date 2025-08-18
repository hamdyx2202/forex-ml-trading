#!/bin/bash
#################################################
# 🚀 Run Forex ML Server with venv_pro
# 📊 For server with existing venv_pro
#################################################

echo "============================================"
echo "🚀 Starting Forex ML Server"
echo "📊 Using existing venv_pro"
echo "============================================"

# Navigate to the correct directory
# Adjust this path based on where your files are
cd /home/forex-ml-trading || cd /root/forex-ml-trading || cd .

# Activate venv_pro
echo "🔧 Activating venv_pro..."
source venv_pro/bin/activate

# Check if activation worked
if [ $? -eq 0 ]; then
    echo "✅ venv_pro activated successfully"
    echo "📦 Python path: $(which python3)"
else
    echo "❌ Failed to activate venv_pro"
    echo "Trying alternative paths..."
    
    # Try different locations
    for dir in /root /home /opt; do
        if [ -f "$dir/forex-ml-trading/venv_pro/bin/activate" ]; then
            echo "Found venv_pro in $dir/forex-ml-trading"
            cd "$dir/forex-ml-trading"
            source venv_pro/bin/activate
            break
        fi
    done
fi

# Run the server
echo ""
echo "🚀 Starting server..."
echo "============================================"

# Check which server file exists and run it
if [ -f "run_forex_ml_server.py" ]; then
    python3 run_forex_ml_server.py
elif [ -f "run_complete_system.py" ]; then
    python3 run_complete_system.py
elif [ -f "unified_prediction_server.py" ]; then
    python3 unified_prediction_server.py
else
    echo "❌ No server file found!"
    echo "Available Python files:"
    ls -la *.py | head -10
fi