#!/bin/bash

# أوامر تشخيص شاملة لنظام Forex ML Trading
# يعمل على السيرفر 69.62.121.53

# الألوان
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# دالة للطباعة الملونة
print_colored() {
    echo -e "${2}${1}${NC}"
}

# دالة لطباعة القسم
print_section() {
    echo ""
    echo -e "${BLUE}========== $1 ==========${NC}"
}

case "$1" in
    # ============== أوامر فحص النظام ==============
    "check-all")
        print_colored "🔍 فحص شامل للنظام..." "$YELLOW"
        
        print_section "1. فحص العمليات"
        ps aux | grep -E "python.*(mt5|learner|training)" | grep -v grep
        
        print_section "2. فحص المنافذ"
        ss -tulpn | grep :5000 2>/dev/null || echo "المنفذ 5000 غير مستخدم"
        
        print_section "3. فحص استخدام الموارد"
        echo "CPU والذاكرة:"
        top -b -n 1 | head -20
        
        print_section "4. فحص مساحة القرص"
        df -h /home
        
        print_section "5. فحص السجلات"
        if [ -d "logs" ]; then
            echo "آخر 10 أسطر من السجلات:"
            tail -n 10 logs/*.log 2>/dev/null || echo "لا توجد سجلات"
        fi
        
        print_section "6. فحص النماذج"
        if [ -d "models/unified_sltp" ]; then
            MODEL_COUNT=$(find models/unified_sltp -name "*.pkl" 2>/dev/null | wc -l)
            echo "عدد النماذج: $MODEL_COUNT"
            echo "أحدث 5 نماذج:"
            ls -lt models/unified_sltp/*.pkl 2>/dev/null | head -5
        fi
        ;;
        
    # ============== أوامر فحص السيرفر ==============
    "check-server")
        print_section "فحص سيرفر MT5"
        
        # فحص العملية
        if pgrep -f "mt5_bridge_server" > /dev/null; then
            PID=$(pgrep -f "mt5_bridge_server")
            print_colored "✅ السيرفر يعمل (PID: $PID)" "$GREEN"
            
            # معلومات العملية
            ps -p $PID -o pid,vsz,rss,comm,args
        else
            print_colored "❌ السيرفر متوقف" "$RED"
        fi
        
        # فحص المنفذ
        if ss -tulpn 2>/dev/null | grep -q ":5000"; then
            print_colored "✅ المنفذ 5000 مفتوح" "$GREEN"
        else
            print_colored "❌ المنفذ 5000 مغلق" "$RED"
        fi
        
        # اختبار الاتصال
        echo ""
        echo "اختبار الاتصال:"
        curl -X POST http://localhost:5000/api/test \
             -H "Content-Type: application/json" \
             -d '{"test": true}' \
             -w "\nHTTP Code: %{http_code}\nTime: %{time_total}s\n" \
             2>/dev/null || print_colored "❌ فشل الاتصال" "$RED"
        ;;
        
    # ============== أوامر فحص قاعدة البيانات ==============
    "check-db")
        print_section "فحص قاعدة البيانات"
        
        if [ -f "trading_data.db" ]; then
            print_colored "✅ قاعدة البيانات موجودة" "$GREEN"
            echo "الحجم: $(du -h trading_data.db | cut -f1)"
            
            # عدد السجلات
            echo ""
            echo "إحصائيات البيانات:"
            sqlite3 trading_data.db "SELECT symbol, timeframe, COUNT(*) as count FROM ohlcv_data GROUP BY symbol, timeframe ORDER BY count DESC LIMIT 10;" 2>/dev/null || echo "خطأ في قراءة البيانات"
        else
            print_colored "❌ قاعدة البيانات غير موجودة" "$RED"
        fi
        ;;
        
    # ============== أوامر فحص الأخطاء ==============
    "check-errors")
        print_section "فحص الأخطاء"
        
        # أخطاء Python
        echo "أخطاء Python الأخيرة:"
        grep -i "error\|exception\|traceback" logs/*.log 2>/dev/null | tail -20 || echo "لا توجد أخطاء مسجلة"
        
        # أخطاء النظام
        echo ""
        echo "أخطاء النظام:"
        dmesg | grep -i "error\|fail" | tail -10
        
        # أخطاء الذاكرة
        echo ""
        echo "استخدام الذاكرة:"
        free -h
        ;;
        
    # ============== أوامر فحص التدريب ==============
    "check-training")
        print_section "فحص التدريب"
        
        # العمليات النشطة
        echo "عمليات التدريب:"
        ps aux | grep -E "learner|training" | grep -v grep
        
        # آخر تدريب
        echo ""
        echo "آخر النماذج المدربة:"
        ls -lt models/unified_sltp/*.pkl 2>/dev/null | head -10 || echo "لا توجد نماذج"
        
        # تقارير التدريب
        echo ""
        echo "آخر تقارير التدريب:"
        ls -lt reports/*.json 2>/dev/null | head -5 || echo "لا توجد تقارير"
        ;;
        
    # ============== أوامر فحص الشبكة ==============
    "check-network")
        print_section "فحص الشبكة"
        
        # فحص الاتصال بالإنترنت
        echo "فحص الاتصال بالإنترنت:"
        ping -c 2 8.8.8.8 > /dev/null 2>&1 && print_colored "✅ متصل بالإنترنت" "$GREEN" || print_colored "❌ غير متصل" "$RED"
        
        # فحص DNS
        echo ""
        echo "فحص DNS:"
        nslookup google.com > /dev/null 2>&1 && print_colored "✅ DNS يعمل" "$GREEN" || print_colored "❌ مشكلة DNS" "$RED"
        
        # الاتصالات النشطة
        echo ""
        echo "الاتصالات النشطة على المنفذ 5000:"
        ss -an | grep :5000
        
        # جدار الحماية
        echo ""
        echo "قواعد جدار الحماية:"
        iptables -L -n | grep 5000 2>/dev/null || echo "لا توجد قواعد خاصة"
        ;;
        
    # ============== أوامر تنظيف وإصلاح ==============
    "fix-permissions")
        print_section "إصلاح الصلاحيات"
        
        chmod +x *.sh
        chmod 755 src/
        chmod 644 *.py
        chmod 755 models/
        chmod 755 logs/
        
        print_colored "✅ تم إصلاح الصلاحيات" "$GREEN"
        ;;
        
    "clean-logs")
        print_section "تنظيف السجلات"
        
        # حفظ نسخة احتياطية
        if [ -d "logs" ]; then
            tar -czf logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz logs/
            rm -f logs/*.log
            print_colored "✅ تم تنظيف السجلات" "$GREEN"
        fi
        ;;
        
    "reset-db")
        print_section "إعادة تعيين قاعدة البيانات"
        
        read -p "⚠️ هل أنت متأكد؟ سيتم حذف جميع البيانات! (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            mv trading_data.db trading_data_backup_$(date +%Y%m%d_%H%M%S).db
            print_colored "✅ تم إعادة تعيين قاعدة البيانات" "$GREEN"
        fi
        ;;
        
    # ============== أوامر المراقبة المباشرة ==============
    "monitor")
        print_section "مراقبة مباشرة"
        
        while true; do
            clear
            print_colored "🔍 مراقبة النظام - $(date)" "$YELLOW"
            
            echo ""
            echo "العمليات:"
            ps aux | grep -E "python.*(mt5|learner|training)" | grep -v grep | awk '{print $2, $11}'
            
            echo ""
            echo "استخدام الموارد:"
            top -b -n 1 | grep python | head -5
            
            echo ""
            echo "المنفذ 5000:"
            ss -an | grep :5000 | grep ESTABLISHED | wc -l | xargs echo "اتصالات نشطة:"
            
            echo ""
            echo "آخر السجلات:"
            tail -n 5 logs/*.log 2>/dev/null | grep -v "^$"
            
            sleep 5
        done
        ;;
        
    # ============== أوامر الاختبار ==============
    "test-connection")
        print_section "اختبار الاتصال بالسيرفر"
        
        # اختبار بسيط
        echo "1. اختبار GET:"
        curl -X GET http://localhost:5000/ -w "\nTime: %{time_total}s\n" 2>/dev/null
        
        echo ""
        echo "2. اختبار POST:"
        curl -X POST http://localhost:5000/api/test \
             -H "Content-Type: application/json" \
             -d '{"test": true, "timestamp": "'$(date +%Y-%m-%d\ %H:%M:%S)'"}' \
             -w "\nHTTP Code: %{http_code}\nTime: %{time_total}s\n" 2>/dev/null
        
        echo ""
        echo "3. اختبار البيانات:"
        curl -X POST http://localhost:5000/api/historical_data \
             -H "Content-Type: application/json" \
             -d '{"symbol": "EURUSD", "timeframe": "H1", "bars_count": 10, "data": []}' \
             -w "\nHTTP Code: %{http_code}\nTime: %{time_total}s\n" 2>/dev/null
        ;;
        
    # ============== أوامر بدء السيرفر ==============
    "start-server")
        print_section "بدء سيرفر MT5"
        
        # فحص البيئة الافتراضية
        if [ -d "venv_pro" ]; then
            print_colored "✅ تم العثور على البيئة الافتراضية venv_pro" "$GREEN"
            source venv_pro/bin/activate
        elif [ -d "venv" ]; then
            print_colored "✅ تم العثور على البيئة الافتراضية venv" "$GREEN"
            source venv/bin/activate
        else
            print_colored "⚠️ لا توجد بيئة افتراضية - استخدام Python النظام" "$YELLOW"
        fi
        
        # التحقق من وجود ملف السيرفر
        if [ -f "start_bridge_server.py" ]; then
            print_colored "🚀 بدء السيرفر..." "$GREEN"
            python start_bridge_server.py
        elif [ -f "src/mt5_bridge_server_linux.py" ]; then
            print_colored "🚀 بدء السيرفر مباشرة..." "$GREEN"
            python -m src.mt5_bridge_server_linux
        else
            print_colored "❌ لم يتم العثور على ملف السيرفر" "$RED"
        fi
        ;;
        
    "stop-server")
        print_section "إيقاف سيرفر MT5"
        
        # إيقاف العملية
        if pgrep -f "mt5_bridge_server" > /dev/null; then
            PID=$(pgrep -f "mt5_bridge_server")
            kill $PID
            print_colored "✅ تم إيقاف السيرفر (PID: $PID)" "$GREEN"
        else
            print_colored "❌ السيرفر غير مشغل" "$RED"
        fi
        ;;
        
    # ============== مساعدة ==============
    *)
        print_colored "📌 أوامر التشخيص المتاحة:" "$PURPLE"
        echo ""
        print_colored "🔍 أوامر الفحص:" "$YELLOW"
        echo "  ./server_diagnostics.sh check-all      - فحص شامل للنظام"
        echo "  ./server_diagnostics.sh check-server   - فحص السيرفر"
        echo "  ./server_diagnostics.sh check-db       - فحص قاعدة البيانات"
        echo "  ./server_diagnostics.sh check-errors   - فحص الأخطاء"
        echo "  ./server_diagnostics.sh check-training - فحص التدريب"
        echo "  ./server_diagnostics.sh check-network  - فحص الشبكة"
        echo ""
        print_colored "🛠️ أوامر الإصلاح:" "$YELLOW"
        echo "  ./server_diagnostics.sh fix-permissions - إصلاح الصلاحيات"
        echo "  ./server_diagnostics.sh clean-logs      - تنظيف السجلات"
        echo "  ./server_diagnostics.sh reset-db        - إعادة تعيين قاعدة البيانات"
        echo ""
        print_colored "📊 أوامر المراقبة:" "$YELLOW"
        echo "  ./server_diagnostics.sh monitor         - مراقبة مباشرة"
        echo "  ./server_diagnostics.sh test-connection - اختبار الاتصال"
        echo ""
        print_colored "🚀 أوامر التحكم:" "$YELLOW"
        echo "  ./server_diagnostics.sh start-server    - بدء السيرفر"
        echo "  ./server_diagnostics.sh stop-server     - إيقاف السيرفر"
        echo ""
        ;;
esac