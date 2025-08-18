#!/bin/bash
#################################################
# 🚀 الإعداد والتشغيل النهائي
# 📊 يعمل مع البيانات الحقيقية
#################################################

echo "============================================"
echo "🚀 إعداد وتشغيل النظام النهائي"
echo "📊 العمل مع 7.8 مليون سجل"
echo "============================================"

# الانتقال للمجلد الصحيح
cd /home/forex-ml-trading || cd /root/forex-ml-trading || cd .

# تفعيل البيئة الافتراضية
echo ""
echo "🐍 تفعيل البيئة الافتراضية..."
source venv_pro/bin/activate

# فحص البيانات
echo ""
echo "🔍 فحص قاعدة البيانات..."
python3 -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('./data/forex_ml.db')

# فحص الجداول
cursor = conn.cursor()
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
tables = cursor.fetchall()
print(f'📋 الجداول: {[t[0] for t in tables]}')

# فحص price_data
cursor.execute(\"SELECT COUNT(*) FROM price_data\")
count = cursor.fetchone()[0]
print(f'📊 إجمالي السجلات: {count:,}')

# عينة من الأزواج
cursor.execute(\"SELECT symbol, COUNT(*) as cnt FROM price_data GROUP BY symbol ORDER BY cnt DESC LIMIT 10\")
pairs = cursor.fetchall()
print(f'🎯 أكثر الأزواج:')
for pair, cnt in pairs:
    print(f'   - {pair}: {cnt:,}')

conn.close()
"

# فحص وتدريب
echo ""
echo "🤖 فحص وتدريب النماذج..."
if [ -f "inspect_and_train_price_data.py" ]; then
    python3 inspect_and_train_price_data.py
else
    echo "⚠️  ملف الفحص غير موجود"
fi

# إيقاف السيرفرات القديمة
echo ""
echo "🛑 إيقاف السيرفرات القديمة..."
pkill -f "forex_ml_server" || true
pkill -f "complete_forex" || true
pkill -f "optimized_forex" || true

# اختيار وتشغيل السيرفر
echo ""
echo "🚀 تشغيل السيرفر المحسن..."

if [ -f "optimized_forex_server.py" ]; then
    SERVER_FILE="optimized_forex_server.py"
elif [ -f "complete_forex_ml_server.py" ]; then
    SERVER_FILE="complete_forex_ml_server.py"
else
    echo "❌ لا يوجد ملف سيرفر!"
    exit 1
fi

echo "📊 تشغيل: $SERVER_FILE"

# التشغيل حسب المعامل
if [ "$1" == "background" ]; then
    nohup python3 "$SERVER_FILE" > server.log 2>&1 &
    PID=$!
    echo "✅ السيرفر يعمل في الخلفية (PID: $PID)"
    
    # الانتظار والتحقق
    sleep 5
    
    # اختبار السيرفر
    echo ""
    echo "🔍 اختبار السيرفر..."
    curl -s http://localhost:5000/status | python3 -m json.tool || echo "⚠️  السيرفر لم يبدأ بعد"
    
    echo ""
    echo "📊 لمشاهدة السجلات: tail -f server.log"
else
    # تشغيل تفاعلي
    python3 "$SERVER_FILE"
fi

echo ""
echo "============================================"
echo "📋 التعليمات:"
echo ""
echo "1. في MT5 استخدم:"
echo "   - ForexMLBot_MultiPair_Scanner_Fixed.mq5"
echo "   - ServerURL: http://69.62.121.53:5000"
echo ""
echo "2. لتدريب جميع النماذج:"
echo "   curl -X POST http://localhost:5000/train_all"
echo ""
echo "3. لمشاهدة النماذج:"
echo "   curl http://localhost:5000/models"
echo ""
echo "4. لمراقبة السجلات:"
echo "   tail -f optimized_forex_server.log"
echo "============================================"