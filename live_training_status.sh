#!/bin/bash
# 📊 حالة التدريب المباشرة

# ألوان للتوضيح
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${YELLOW}📊 Live Training Status${NC}"
    echo "================================"
    echo -e "🕐 $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # عدد النماذج المدربة
    if [ -d "./trained_models" ]; then
        MODEL_COUNT=$(ls ./trained_models/*.pkl 2>/dev/null | grep -v scaler | wc -l)
        echo -e "${GREEN}🤖 Models Trained: $MODEL_COUNT${NC}"
        
        # آخر نموذج مدرب
        LAST_MODEL=$(ls -t ./trained_models/*.pkl 2>/dev/null | grep -v scaler | head -1)
        if [ ! -z "$LAST_MODEL" ]; then
            LAST_MODEL_NAME=$(basename "$LAST_MODEL")
            echo -e "📍 Latest: ${GREEN}$LAST_MODEL_NAME${NC}"
        fi
    fi
    
    echo ""
    echo "📝 Recent Activity:"
    echo "--------------------------------"
    
    # آخر 5 أسطر من السجل
    if [ -f "enhanced_ml_server.log" ]; then
        tail -5 enhanced_ml_server.log | grep -E "(Training|✅|❌|Completed)" | tail -3
    fi
    
    # حجم قاعدة البيانات
    echo ""
    echo "💾 Database Size:"
    if [ -f "./data/forex_ml.db" ]; then
        DB_SIZE=$(du -h ./data/forex_ml.db | cut -f1)
        echo "   forex_ml.db: $DB_SIZE"
    fi
    
    # استخدام الذاكرة
    echo ""
    echo "🖥️ System Resources:"
    FREE_MEM=$(free -h | grep Mem | awk '{print $4}')
    echo "   Free Memory: $FREE_MEM"
    
    # تحديث كل 10 ثواني
    echo ""
    echo "--------------------------------"
    echo "Press Ctrl+C to exit"
    sleep 10
done