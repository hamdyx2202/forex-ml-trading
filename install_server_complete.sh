#!/bin/bash
#################################################
# 🚀 Complete Forex ML System Installation
# 📊 Full system with all features
# 🤖 For Linux Server (69.62.121.53)
#################################################

echo "=================================================="
echo "🚀 Installing Complete Forex ML Trading System"
echo "📊 Full Features: Training + Learning + API"
echo "🌐 Server: 69.62.121.53:5000"
echo "=================================================="

# Update system
echo -e "\n📦 Updating system packages..."
apt-get update -y

# Install Python and dependencies
echo -e "\n🐍 Installing Python and build tools..."
apt-get install -y python3 python3-dev python3-pip python3-venv
apt-get install -y build-essential gcc g++ make
apt-get install -y libgomp1

# Create virtual environment
echo -e "\n🔧 Creating Python virtual environment..."
python3 -m venv venv_forex

# Activate venv and install packages
echo -e "\n📚 Installing Python packages..."
source venv_forex/bin/activate

# Install all required packages
pip install --upgrade pip
pip install pandas numpy scikit-learn
pip install lightgbm xgboost
pip install flask flask-cors
pip install joblib
pip install scipy

# Install TA-Lib (optional but recommended)
echo -e "\n📈 Installing TA-Lib (optional)..."
apt-get install -y libta-lib0-dev
pip install TA-Lib || echo "TA-Lib installation failed - continuing without it"

# Create necessary directories
echo -e "\n📁 Creating directories..."
mkdir -p unified_models
mkdir -p logs
mkdir -p data

# Set permissions
chmod -R 755 .

echo -e "\n✅ Installation complete!"
echo "📌 Next steps:"
echo "   1. Upload all system files"
echo "   2. Run: source venv_forex/bin/activate"
echo "   3. Run: python3 start_forex_server.py"
echo "=================================================="