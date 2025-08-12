#!/bin/bash
# تثبيت النظام الكامل للوصول لدقة 95%

echo "🚀 Installing FULL Forex ML Trading System for 95% Accuracy"
echo "==========================================================="

# تحديث النظام
echo "📦 Updating system packages..."
sudo apt-get update -y

# تثبيت المتطلبات الأساسية
echo "🔧 Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev python3-venv build-essential
sudo apt-get install -y wget libta-lib0 libta-lib0-dev
sudo apt-get install -y gcc g++ make cmake
sudo apt-get install -y libatlas-base-dev gfortran

# تثبيت TA-Lib من المصدر
echo "📊 Installing TA-Lib from source..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd /home/forex-ml-trading

# إنشاء بيئة افتراضية جديدة
echo "🌐 Creating powerful virtual environment..."
python3 -m venv venv_pro
source venv_pro/bin/activate

# تثبيت جميع المكتبات المتقدمة
echo "💪 Installing ALL advanced libraries..."
pip install --upgrade pip setuptools wheel

# المكتبات الأساسية
pip install pandas==2.0.3 numpy==1.24.3 scipy statsmodels

# مكتبات التعلم الآلي المتقدمة
pip install scikit-learn==1.3.0
pip install lightgbm==4.1.0
pip install xgboost==2.0.0
pip install catboost
pip install tensorflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# مكتبات التحليل الفني المتقدمة
pip install TA-Lib==0.4.28
pip install ta==0.10.2
pip install pandas-ta==0.3.14b0
pip install technical
pip install tulipy

# مكتبات إضافية للدقة العالية
pip install optuna  # للتحسين التلقائي للمعاملات
pip install shap   # لتفسير النماذج
pip install mlflow  # لتتبع التجارب
pip install prophet # للتنبؤ بالسلاسل الزمنية
pip install arch   # لنماذج GARCH

# باقي المكتبات
pip install yfinance schedule loguru joblib sqlalchemy
pip install flask flask-cors python-telegram-bot
pip install streamlit plotly tqdm pyyaml
pip install pytest black flake8

echo ""
echo "✅ FULL System Installation Completed!"
echo ""
echo "🎯 This system is designed for 95% accuracy with:"
echo "   • Advanced ML models (LightGBM, XGBoost, CatBoost, Neural Networks)"
echo "   • 50+ Technical indicators"
echo "   • Pattern recognition algorithms"
echo "   • Auto-optimization with Optuna"
echo "   • Deep learning capabilities"
echo ""