#!/bin/bash
echo "🚀 Starting Complete Advanced Forex ML System"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ -d "venv_pro" ]; then
    echo "✅ Using existing virtual environment"
    source venv_pro/bin/activate
elif [ -d "venv" ]; then
    echo "✅ Using existing virtual environment"
    source venv/bin/activate
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_pro
    source venv_pro/bin/activate
fi

echo ""
echo "📦 Installing/Updating packages..."

# Install basic requirements
pip install --upgrade pip
pip install pandas numpy scikit-learn joblib scipy

# Install ML frameworks
pip install lightgbm xgboost

# Try to install TA-Lib (may fail on some systems)
pip install TA-Lib 2>/dev/null || echo "⚠️ TA-Lib installation failed, will use basic indicators"

echo ""
echo "🚀 Starting training..."
echo ""

# Run the complete system
python3 complete_advanced_system.py

echo ""
echo "✅ Training complete!"