#!/bin/bash

# أوامر سريعة للتشخيص والإصلاح

echo "🔍 فحص سريع لنظام Forex ML..."
echo ""

# 1. فحص السيرفر
echo "1️⃣ فحص السيرفر:"
if pgrep -f "mt5_bridge_server" > /dev/null; then
    echo "   ✅ السيرفر يعمل"
    # اختبار سريع
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/test 2>/dev/null)
    if [ "$RESPONSE" == "200" ] || [ "$RESPONSE" == "404" ]; then
        echo "   ✅ السيرفر يستجيب"
    else
        echo "   ❌ السيرفر لا يستجيب (HTTP: $RESPONSE)"
    fi
else
    echo "   ❌ السيرفر متوقف"
    echo ""
    echo "   🔧 لبدء السيرفر:"
    echo "   python src/mt5_bridge_server_advanced.py"
fi

echo ""

# 2. فحص قاعدة البيانات
echo "2️⃣ فحص قاعدة البيانات:"
if [ -f "trading_data.db" ]; then
    SIZE=$(du -h trading_data.db | cut -f1)
    echo "   ✅ قاعدة البيانات موجودة (الحجم: $SIZE)"
else
    echo "   ❌ قاعدة البيانات غير موجودة"
fi

echo ""

# 3. فحص النماذج
echo "3️⃣ فحص النماذج:"
if [ -d "models/unified_sltp" ]; then
    COUNT=$(find models/unified_sltp -name "*.pkl" 2>/dev/null | wc -l)
    echo "   📊 عدد النماذج: $COUNT"
else
    echo "   ❌ مجلد النماذج غير موجود"
    mkdir -p models/unified_sltp
    echo "   ✅ تم إنشاء مجلد النماذج"
fi

echo ""

# 4. فحص السجلات
echo "4️⃣ فحص السجلات:"
if [ -d "logs" ]; then
    echo "   ✅ مجلد السجلات موجود"
    # آخر خطأ
    LAST_ERROR=$(grep -i "error" logs/*.log 2>/dev/null | tail -1)
    if [ -n "$LAST_ERROR" ]; then
        echo "   ⚠️ آخر خطأ: ${LAST_ERROR:0:100}..."
    fi
else
    echo "   ❌ مجلد السجلات غير موجود"
    mkdir -p logs
    echo "   ✅ تم إنشاء مجلد السجلات"
fi

echo ""

# 5. فحص البيئة الافتراضية
echo "5️⃣ فحص البيئة الافتراضية:"
if [ -d "venv_pro" ]; then
    echo "   ✅ البيئة الافتراضية موجودة"
    # فحص الحزم المطلوبة
    source venv_pro/bin/activate 2>/dev/null
    MISSING=""
    for pkg in pandas numpy sklearn xgboost joblib loguru flask; do
        python -c "import $pkg" 2>/dev/null || MISSING="$MISSING $pkg"
    done
    if [ -n "$MISSING" ]; then
        echo "   ⚠️ حزم ناقصة:$MISSING"
        echo "   🔧 لتثبيتها: pip install$MISSING"
    else
        echo "   ✅ جميع الحزم المطلوبة مثبتة"
    fi
else
    echo "   ❌ البيئة الافتراضية غير موجودة"
fi

echo ""

# 6. اقتراحات الإصلاح
echo "📝 اقتراحات:"

if ! pgrep -f "mt5_bridge_server" > /dev/null; then
    echo "   1. بدء السيرفر:"
    echo "      source venv_pro/bin/activate"
    echo "      nohup python src/mt5_bridge_server_advanced.py > logs/server.log 2>&1 &"
fi

if [ ! -f "performance_tracker.py" ]; then
    echo "   2. ملف performance_tracker.py مفقود!"
    echo "      قم بنسخه من الجهاز المحلي"
fi

echo ""
echo "✅ انتهى الفحص السريع"
echo ""
echo "للمزيد من التفاصيل استخدم:"
echo "  ./server_diagnostics.sh check-all"