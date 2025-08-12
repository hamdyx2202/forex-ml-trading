#!/bin/bash
# تثبيت المتطلبات الأساسية للنظام

echo "🔧 Installing basic requirements for Forex ML Trading System"
echo "=========================================================="

# تحديث النظام
echo "📦 Updating system packages..."
sudo apt-get update -y

# تثبيت Python pip
echo "🐍 Installing Python pip..."
sudo apt-get install -y python3-pip python3-venv

# إنشاء بيئة افتراضية
echo "🌐 Creating virtual environment..."
python3 -m venv venv

# تفعيل البيئة الافتراضية
source venv/bin/activate

# تثبيت المكتبات الأساسية فقط
echo "📚 Installing essential libraries..."
pip install --upgrade pip
pip install pandas numpy scikit-learn joblib loguru

# تثبيت مكتبات الخادم
echo "🌐 Installing server libraries..."
pip install flask flask-cors

# تثبيت مكتبة قاعدة البيانات
echo "💾 Installing database library..."
pip install sqlalchemy

# المكتبات الاختيارية (بدون TA-Lib)
echo "📊 Installing optional libraries..."
pip install yfinance schedule tqdm pyyaml

echo ""
echo "✅ Basic installation completed!"
echo ""
echo "📝 Note: TA-Lib not installed (requires complex setup)"
echo "   The system will work with simple versions of scripts"
echo ""
echo "🚀 To start the system:"
echo "   1. source venv/bin/activate"
echo "   2. python3 src/mt5_bridge_server_linux.py"
echo ""
echo "🧠 To run advanced learning (simple version):"
echo "   python3 src/advanced_learner_simple.py"
echo ""
echo "🔄 To run continuous learning (simple version):"
echo "   python3 src/continuous_learner_simple.py"