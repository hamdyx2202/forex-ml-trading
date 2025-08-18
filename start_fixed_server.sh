#!/bin/bash
# Start the fixed Forex ML Server

echo "🚀 Starting Fixed Forex ML Server..."
echo "📊 With improved JSON error handling"

# Check if we're on the server or local
if [ -d "/root/forex-ml-trading" ]; then
    cd /root/forex-ml-trading
    echo "📍 Server location: /root/forex-ml-trading"
elif [ -d "/home/forex-ml-trading" ]; then
    cd /home/forex-ml-trading
    echo "📍 Server location: /home/forex-ml-trading"
else
    echo "📍 Current location: $(pwd)"
fi

# Try to find and activate venv_pro
if [ -f "venv_pro/bin/activate" ]; then
    source venv_pro/bin/activate
    echo "✅ Activated venv_pro"
elif [ -f "../venv_pro/bin/activate" ]; then
    source ../venv_pro/bin/activate
    echo "✅ Activated ../venv_pro"
else
    echo "⚠️  Could not find venv_pro, using system Python"
fi

# Run the fixed server
echo "🌐 Starting server on port 5000..."
python3 run_forex_ml_server_fixed.py