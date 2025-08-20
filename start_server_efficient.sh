#!/bin/bash
# تشغيل السيرفر بطريقة أكثر كفاءة لتقليل استهلاك CPU

echo "🚀 تشغيل السيرفر بإعدادات موفرة للموارد..."
echo "========================================="

# 1. إيقاف أي سيرفر قديم
echo "⏹️ إيقاف السيرفر القديم..."
pkill -f "gunicorn.*enhanced_ml_server"
sleep 2

# 2. تحديد إعدادات أقل استهلاكاً
export PYTHONOPTIMIZE=1  # تقليل استهلاك الذاكرة
export OMP_NUM_THREADS=1  # تحديد عدد الخيوط

# 3. تشغيل السيرفر بإعدادات محسّنة
echo "🔧 الإعدادات:"
echo "   - Workers: 1 (بدلاً من أكثر)"
echo "   - Timeout: 120 ثانية"
echo "   - Max requests: 1000 (لإعادة تشغيل العامل)"
echo "   - CPU affinity: core 0 (لتحديد نواة واحدة)"

# تشغيل مع nice لتقليل الأولوية
nice -n 10 nohup gunicorn \
    -b 0.0.0.0:5000 \
    enhanced_ml_server:app \
    --workers 1 \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --worker-class sync \
    --preload \
    > server.log 2>&1 &

SERVER_PID=$!
echo "✅ السيرفر يعمل الآن (PID: $SERVER_PID)"

# 4. حفظ PID للإيقاف لاحقاً
echo $SERVER_PID > server.pid

echo ""
echo "📊 لمراقبة استهلاك CPU:"
echo "   top -p $SERVER_PID"
echo ""
echo "🛑 لإيقاف السيرفر:"
echo "   kill \$(cat server.pid)"
echo ""
echo "📝 لمشاهدة السجلات:"
echo "   tail -f server.log"