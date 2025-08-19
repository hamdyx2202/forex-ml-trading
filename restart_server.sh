#!/bin/bash
# 🔄 إعادة تشغيل السيرفر مع التعديلات الجديدة

echo "🔄 Restarting Enhanced ML Server..."

# 1. إيقاف السيرفر الحالي
echo "⏹️ Stopping current server..."
pkill -f "enhanced_ml_server"
sleep 2

# 2. حفظ السجلات القديمة
if [ -f enhanced_ml_server.log ]; then
    mv enhanced_ml_server.log "enhanced_ml_server_$(date +%Y%m%d_%H%M%S).log"
    echo "📁 Old logs saved"
fi

# 3. تشغيل السيرفر الجديد
echo "🚀 Starting server with new settings..."
nohup gunicorn -b 0.0.0.0:5000 enhanced_ml_server:app --workers 1 --timeout 120 > server.log 2>&1 &

echo "✅ Server restarted!"
echo ""
echo "🔧 New settings:"
echo "   - Buy threshold: ratio > 0.55 & score > 5 (was 0.6 & 20)"
echo "   - Sell threshold: ratio < 0.45 & score < -5 (was 0.4 & -20)"
echo "   - Min confidence: 0.55 (was 0.65)"
echo "   - News impact: 0.9x (was 0.6x)"
echo ""
echo "📊 Monitor with:"
echo "   ./monitor_server.sh"
echo "   python3 analyze_server_logs.py"