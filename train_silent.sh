#!/bin/bash
# 🔇 تشغيل التدريب بدون تحذيرات مزعجة

echo "🚀 Starting training without warnings..."
echo "="*60

# إخفاء كل التحذيرات
export PYTHONWARNINGS='ignore'

# تشغيل التدريب مع تصفية التحذيرات
python3 -W ignore train_all_pairs_enhanced.py 2>&1 | grep -v "PerformanceWarning" | grep -v "DataFrame is highly fragmented"

echo "✅ Training completed!"