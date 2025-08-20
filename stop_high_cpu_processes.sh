#!/bin/bash
# إيقاف العمليات التي تستهلك CPU عالي

echo "🛑 إيقاف العمليات التي تستهلك CPU عالي..."
echo "================================"

# 1. إيقاف التدريب المحسن
echo "⏹️ إيقاف train_all_pairs_enhanced.py..."
pkill -f "train_all_pairs_enhanced.py"

# 2. إيقاف run_complete_system.py القديم
echo "⏹️ إيقاف run_complete_system.py..."
pkill -f "run_complete_system.py"

# 3. إيقاف السيرفر gunicorn (مؤقتاً)
echo "⏹️ إيقاف gunicorn server..."
pkill -f "gunicorn.*enhanced_ml_server"

# 4. إيقاف أي عمليات Python قديمة
echo "⏹️ البحث عن عمليات Python معلقة..."
ps aux | grep python | grep -E "(train|run_complete|forex)" | grep -v grep

# انتظار قليلاً
sleep 2

echo ""
echo "✅ تم إيقاف العمليات"
echo ""
echo "📊 استخدام CPU الحالي:"
top -bn1 | head -20

echo ""
echo "💡 لإعادة تشغيل السيرفر فقط (بدون التدريب):"
echo "   nohup gunicorn -b 0.0.0.0:5000 enhanced_ml_server:app --workers 1 --timeout 120 > server.log 2>&1 &"