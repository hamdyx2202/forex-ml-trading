#!/bin/bash
#################################################
# 🚀 نشر النظام الكامل على السيرفر
# 📊 مع كل الميزات المتقدمة
#################################################

echo "============================================"
echo "🚀 نشر النظام الكامل للتداول بالذكاء الاصطناعي"
echo "📊 6 نماذج ML + 200+ ميزة + تعلم مستمر"
echo "============================================"

# الملفات المطلوبة
FILES=(
    "complete_forex_ml_server.py"
    "train_with_real_data.py"
    "unified_trading_learning_system.py"
    "ForexMLBot_Advanced_V3_Unified.mq5"
)

# معلومات السيرفر
SERVER="root@69.62.121.53"
REMOTE_DIR="/home/forex-ml-trading"

echo "📤 رفع الملفات إلى السيرفر..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   - $file"
        scp "$file" "$SERVER:$REMOTE_DIR/"
    else
        echo "   ⚠️  $file غير موجود"
    fi
done

echo ""
echo "🔧 تنفيذ الأوامر على السيرفر..."
ssh "$SERVER" << 'EOF'
cd /home/forex-ml-trading

echo "📍 المسار الحالي: $(pwd)"
echo "📂 الملفات الموجودة:"
ls -la *.py | head -10

# تفعيل البيئة الافتراضية
echo ""
echo "🐍 تفعيل البيئة الافتراضية..."
if [ -f "venv_pro/bin/activate" ]; then
    source venv_pro/bin/activate
    echo "✅ تم تفعيل venv_pro"
else
    echo "❌ لم يتم العثور على venv_pro"
fi

# إيقاف أي سيرفر قديم
echo ""
echo "🛑 إيقاف السيرفر القديم..."
pkill -f "run_forex_ml_server" || true
pkill -f "complete_forex_ml_server" || true

# تشغيل السيرفر الجديد
echo ""
echo "🚀 تشغيل السيرفر الكامل..."
nohup python3 complete_forex_ml_server.py > complete_server.log 2>&1 &

# الانتظار قليلاً
sleep 3

# التحقق من حالة السيرفر
echo ""
echo "🔍 التحقق من حالة السيرفر..."
if pgrep -f "complete_forex_ml_server" > /dev/null; then
    echo "✅ السيرفر يعمل!"
    echo ""
    echo "📊 اختبار السيرفر:"
    curl -s http://localhost:5000/status | python3 -m json.tool
else
    echo "❌ فشل تشغيل السيرفر"
    echo "📋 آخر 20 سطر من السجل:"
    tail -20 complete_server.log
fi

echo ""
echo "✅ تم الانتهاء!"
echo "🌐 السيرفر: http://69.62.121.53:5000"
echo "📊 للمراقبة: tail -f complete_server.log"
EOF

echo ""
echo "============================================"
echo "✅ تم نشر النظام الكامل"
echo "🌐 يمكنك الآن اختبار السيرفر من أي مكان:"
echo "   curl http://69.62.121.53:5000/status"
echo "============================================"