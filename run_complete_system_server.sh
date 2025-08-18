#!/bin/bash
#################################################
# 🚀 تشغيل النظام الكامل مع venv_pro
# 📊 يحتوي على كل الميزات بدون تبسيط
#################################################

echo "============================================"
echo "🚀 تشغيل النظام الكامل للتداول بالذكاء الاصطناعي"
echo "📊 6 نماذج ML | 200+ ميزة | تعلم مستمر"
echo "🌐 السيرفر: 69.62.121.53:5000"
echo "============================================"

# تحديد المسار
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# البحث عن venv_pro
VENV_PATH=""
if [ -d "venv_pro" ]; then
    VENV_PATH="venv_pro"
elif [ -d "../venv_pro" ]; then
    VENV_PATH="../venv_pro"
elif [ -d "/root/forex-ml-trading/venv_pro" ]; then
    VENV_PATH="/root/forex-ml-trading/venv_pro"
elif [ -d "/home/forex-ml-trading/venv_pro" ]; then
    VENV_PATH="/home/forex-ml-trading/venv_pro"
fi

if [ -n "$VENV_PATH" ]; then
    echo "✅ تم العثور على venv_pro في: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "✅ تم تفعيل البيئة الافتراضية"
else
    echo "⚠️  لم يتم العثور على venv_pro"
    echo "📦 استخدام Python النظام..."
fi

# التحقق من المكتبات
echo ""
echo "📦 فحص المكتبات المطلوبة..."
python3 -c "
import sys
try:
    import flask
    print('✅ Flask متوفر')
except:
    print('❌ Flask غير متوفر')
    
try:
    import pandas
    print('✅ Pandas متوفر')
except:
    print('❌ Pandas غير متوفر')
    
try:
    import sklearn
    print('✅ Scikit-learn متوفر')
except:
    print('❌ Scikit-learn غير متوفر')
    
try:
    import lightgbm
    print('✅ LightGBM متوفر')
except:
    print('⚠️  LightGBM غير متوفر (اختياري)')
    
try:
    import xgboost
    print('✅ XGBoost متوفر')
except:
    print('⚠️  XGBoost غير متوفر (اختياري)')
"

# إيقاف أي سيرفر قديم
echo ""
echo "🛑 إيقاف أي سيرفر قديم..."
pkill -f "forex_ml_server" || true
pkill -f "complete_forex_ml" || true

# اختيار السيرفر المناسب
SERVER_FILE=""
if [ -f "complete_forex_ml_server.py" ]; then
    SERVER_FILE="complete_forex_ml_server.py"
elif [ -f "run_forex_ml_server_fixed.py" ]; then
    SERVER_FILE="run_forex_ml_server_fixed.py"
elif [ -f "run_forex_ml_server.py" ]; then
    SERVER_FILE="run_forex_ml_server.py"
else
    echo "❌ لم يتم العثور على ملف السيرفر!"
    exit 1
fi

echo ""
echo "🚀 تشغيل السيرفر: $SERVER_FILE"
echo "📊 السجلات في: complete_forex_ml_server.log"
echo ""

# تشغيل السيرفر
if [ "$1" == "background" ]; then
    # تشغيل في الخلفية
    nohup python3 "$SERVER_FILE" > server_output.log 2>&1 &
    SERVER_PID=$!
    echo "✅ السيرفر يعمل في الخلفية (PID: $SERVER_PID)"
    echo "📊 لمشاهدة السجلات: tail -f server_output.log"
else
    # تشغيل تفاعلي
    python3 "$SERVER_FILE"
fi