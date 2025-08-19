#!/bin/bash
# 📊 مراقبة السيرفر المحسن

echo "📊 Enhanced ML Server Monitor"
echo "================================"

# متابعة السجلات الحية
echo "🔍 Following server logs..."
echo "Press Ctrl+C to stop"
echo ""

# متابعة سجل السيرفر الرئيسي
tail -f enhanced_ml_server.log | grep -E "(Request:|Market Analysis:|confidence:|action:|HOLD|BUY|SELL|News|volatility|Risk|Balance|Auto-training|Saved|Score=|رفض|rejected|⚠️|❌|✅|💰|📊|🤖)"