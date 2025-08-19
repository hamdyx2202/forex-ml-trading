#!/bin/bash
#################################################
# 🧹 تنظيف النماذج القديمة والضعيفة
# يُنصح بتشغيله يومياً أو أسبوعياً
#################################################

echo "============================================"
echo "🧹 Model Cleanup Script"
echo "📅 $(date)"
echo "============================================"

cd /home/forex-ml-trading || cd /root/forex-ml-trading || cd .

# تفعيل البيئة الافتراضية
source venv_pro/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# تشغيل تنظيف النماذج
echo ""
echo "🔍 Analyzing models..."
python3 model_manager.py

# إضافة لـ crontab للتشغيل اليومي:
# 0 3 * * * /home/forex-ml-trading/cleanup_models.sh > /home/forex-ml-trading/cleanup.log 2>&1

echo ""
echo "✅ Cleanup complete!"
echo "============================================"