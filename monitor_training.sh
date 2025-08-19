#!/bin/bash
# 📊 مراقبة تقدم التدريب

echo "📊 Training Progress Monitor"
echo "="*60

# متابعة السجلات الحية
echo "🔍 Following training logs..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# عرض آخر 20 سطر ثم متابعة الجديد
tail -n 20 -f enhanced_ml_server.log | grep -E "(Training|✅|❌|Success|Failed|Completed|Progress|models trained|[0-9]+/[0-9]+)"