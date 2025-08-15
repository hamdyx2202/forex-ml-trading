#!/bin/bash

# أوامر بسيطة لإدارة نظام Forex ML Trading

# الألوان
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# دالة للطباعة الملونة
print_colored() {
    echo -e "${2}${1}${NC}"
}

case "$1" in
    "stop")
        print_colored "🛑 إيقاف جميع العمليات..." "$RED"
        pkill -f "mt5_bridge_server"
        pkill -f "learner_unified"
        pkill -f "training_sltp"
        print_colored "✅ تم إيقاف جميع العمليات" "$GREEN"
        ;;
        
    "start")
        print_colored "🚀 بدء النظام..." "$GREEN"
        source venv_pro/bin/activate
        nohup python src/mt5_bridge_server_advanced.py > logs/server.log 2>&1 &
        sleep 3
        nohup python automated_training_sltp.py > logs/training.log 2>&1 &
        print_colored "✅ النظام يعمل" "$GREEN"
        ;;
        
    "server")
        print_colored "🌐 بدء السيرفر فقط..." "$YELLOW"
        source venv_pro/bin/activate
        python src/mt5_bridge_server_advanced.py
        ;;
        
    "train")
        print_colored "🎯 بدء التدريب..." "$YELLOW"
        source venv_pro/bin/activate
        python integrated_training_sltp.py
        ;;
        
    "train-auto")
        print_colored "🤖 بدء التدريب الآلي..." "$YELLOW"
        source venv_pro/bin/activate
        python automated_training_sltp.py
        ;;
        
    "status")
        print_colored "📊 حالة النظام:" "$YELLOW"
        echo ""
        
        # فحص السيرفر
        if pgrep -f "mt5_bridge_server" > /dev/null; then
            print_colored "✅ السيرفر: يعمل" "$GREEN"
        else
            print_colored "❌ السيرفر: متوقف" "$RED"
        fi
        
        # فحص التدريب
        if pgrep -f "training_sltp" > /dev/null; then
            print_colored "✅ التدريب: يعمل" "$GREEN"
        else
            print_colored "❌ التدريب: متوقف" "$RED"
        fi
        
        # عدد النماذج
        MODEL_COUNT=$(find models/unified_sltp -name "*.pkl" 2>/dev/null | wc -l)
        print_colored "📈 عدد النماذج: $MODEL_COUNT" "$YELLOW"
        ;;
        
    "logs")
        print_colored "📜 عرض السجلات..." "$YELLOW"
        tail -f logs/*.log
        ;;
        
    "clean")
        print_colored "🧹 تنظيف الملفات المؤقتة..." "$YELLOW"
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -exec rm -rf {} +
        print_colored "✅ تم التنظيف" "$GREEN"
        ;;
        
    *)
        print_colored "📌 الأوامر المتاحة:" "$YELLOW"
        echo ""
        echo "  ./simple_commands.sh stop       - إيقاف جميع العمليات"
        echo "  ./simple_commands.sh start      - بدء النظام الكامل"
        echo "  ./simple_commands.sh server     - بدء السيرفر فقط"
        echo "  ./simple_commands.sh train      - تدريب النماذج"
        echo "  ./simple_commands.sh train-auto - التدريب الآلي المستمر"
        echo "  ./simple_commands.sh status     - عرض حالة النظام"
        echo "  ./simple_commands.sh logs       - عرض السجلات"
        echo "  ./simple_commands.sh clean      - تنظيف الملفات المؤقتة"
        echo ""
        ;;
esac