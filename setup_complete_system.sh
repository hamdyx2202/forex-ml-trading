#!/bin/bash
#################################################
# 🚀 إعداد النظام الكامل على السيرفر
# 📊 تدريب جميع البيانات + تشغيل السيرفر
#################################################

echo "============================================"
echo "🚀 إعداد النظام الكامل للتداول بالذكاء الاصطناعي"
echo "📊 البحث عن البيانات وتدريب النماذج"
echo "============================================"

# 1. البحث عن قواعد البيانات
echo ""
echo "🔍 البحث عن قواعد البيانات..."
find / -name "*.db" -size +10M 2>/dev/null | head -20

# 2. البحث في المسارات المحتملة
echo ""
echo "📂 فحص المسارات المعروفة..."
POSSIBLE_PATHS=(
    "/home/forex-ml-trading/data"
    "/root/forex-ml-trading/data"
    "/var/lib/mysql"
    "/var/lib/postgresql"
    "/opt/forex"
    "/home/data"
    "/root/data"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ موجود: $path"
        ls -lh "$path"/*.db 2>/dev/null | head -5
    fi
done

# 3. تشغيل سكريبت البحث والتدريب
echo ""
echo "🤖 تشغيل البحث والتدريب التلقائي..."
if [ -f "find_and_train_data.py" ]; then
    python3 find_and_train_data.py
else
    echo "❌ ملف find_and_train_data.py غير موجود!"
fi

# 4. تحديث إعدادات السيرفر
echo ""
echo "🔧 تحديث إعدادات السيرفر..."
if [ -f "database_config.txt" ]; then
    source database_config.txt
    echo "📊 أفضل قاعدة بيانات: $BEST_DATABASE"
    echo "📈 إجمالي السجلات: $TOTAL_RECORDS"
    
    # تحديث ملف السيرفر
    if [ -n "$BEST_DATABASE" ]; then
        sed -i "s|self.historical_db = './data/forex_ml.db'|self.historical_db = '$BEST_DATABASE'|g" complete_forex_ml_server.py
        echo "✅ تم تحديث مسار قاعدة البيانات"
    fi
fi

# 5. إيقاف السيرفر القديم
echo ""
echo "🛑 إيقاف أي سيرفر قديم..."
pkill -f "forex_ml_server" || true
pkill -f "complete_forex_ml" || true

# 6. تشغيل السيرفر الجديد
echo ""
echo "🚀 تشغيل السيرفر الكامل..."
nohup python3 complete_forex_ml_server.py > complete_server.log 2>&1 &
SERVER_PID=$!

echo "✅ السيرفر يعمل (PID: $SERVER_PID)"

# 7. الانتظار والتحقق
sleep 5

# 8. اختبار السيرفر
echo ""
echo "🔍 اختبار السيرفر..."
curl -s http://localhost:5000/status | python3 -m json.tool

# 9. عرض النماذج المدربة
echo ""
echo "📊 النماذج المدربة:"
curl -s http://localhost:5000/models | python3 -m json.tool

# 10. عرض التعليمات
echo ""
echo "============================================"
echo "✅ النظام جاهز للعمل!"
echo ""
echo "📊 معلومات السيرفر:"
echo "   - العنوان: http://69.62.121.53:5000"
echo "   - السجلات: tail -f complete_server.log"
echo "   - النماذج: ./trained_models/"
echo ""
echo "🎯 في MT5:"
echo "   - استخدم: ForexMLBot_MultiPair_Scanner.mq5"
echo "   - يفحص جميع الأزواج والفريمات"
echo "   - يفتح صفقات متعددة"
echo ""
echo "💡 للمراقبة:"
echo "   watch 'tail -20 complete_server.log'"
echo "============================================"