#!/bin/bash
# إيقاف العمليات غير المستخدمة لتقليل استهلاك CPU

echo "🛑 إيقاف العمليات غير المستخدمة..."
echo "================================"

# 1. إيقاف run_complete_system.py من المسار القديم
echo "⏹️ إيقاف run_complete_system.py من /opt/forex-ml-trading..."
sudo pkill -f "/opt/forex-ml-trading/venv_forex/bin/python run_complete_system.py" || pkill -f "run_complete_system.py"

# 2. إيقاف أي عمليات تدريب قد تكون معلقة
echo "⏹️ إيقاف عمليات التدريب..."
pkill -f "train_all_pairs_enhanced.py"
pkill -f "train_with_available_data.py"
pkill -f "training" 

# 3. عرض العمليات التي تستهلك CPU عالي
echo ""
echo "📊 العمليات التي تستهلك أعلى CPU:"
ps aux --sort=-%cpu | head -10

# 4. عرض عمليات Python النشطة
echo ""
echo "🐍 عمليات Python النشطة:"
ps aux | grep python | grep -v grep

# 5. إيقاف العمليات من المجلد القديم /opt
echo ""
echo "🗑️ إيقاف أي عمليات من /opt/forex-ml-trading..."
ps aux | grep "/opt/forex-ml-trading" | grep -v grep | awk '{print $2}' | xargs -r kill -9

echo ""
echo "✅ تم إيقاف العمليات غير المستخدمة"

# اختياري: حذف المجلد القديم إذا لم تعد تحتاجه
echo ""
echo "❓ المجلد /opt/forex-ml-trading لا يُستخدم. هل تريد حذفه؟ (يحتاج صلاحيات sudo)"
echo "   إذا أردت حذفه، شغل: sudo rm -rf /opt/forex-ml-trading"