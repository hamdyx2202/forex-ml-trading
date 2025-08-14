#!/bin/bash
# Daily Maintenance Script
# سكريبت الصيانة اليومية

# الألوان
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# المتغيرات
BASE_DIR="/home/forex-ml-trading"
BACKUP_DIR="${BASE_DIR}/backups"
LOG_FILE="${BASE_DIR}/logs/maintenance.log"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_${DATE}"

# دالة للتسجيل
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

# دالة للتحقق من النجاح
check_status() {
    if [ $? -eq 0 ]; then
        log "${GREEN}✓ ${1} succeeded${NC}"
    else
        log "${RED}✗ ${1} failed${NC}"
        return 1
    fi
}

# بدء الصيانة
log "
========================================
Daily Maintenance Started: $(date)
========================================"

# 1. إنشاء مجلد النسخ الاحتياطية
mkdir -p "${BACKUP_DIR}/daily"
mkdir -p "${BACKUP_DIR}/weekly"
mkdir -p "${BACKUP_DIR}/models"
mkdir -p "${BACKUP_DIR}/database"

# 2. نسخ احتياطي للنماذج
log "\n${YELLOW}1. Backing up models...${NC}"
if [ -d "${BASE_DIR}/models" ]; then
    tar -czf "${BACKUP_DIR}/models/${BACKUP_NAME}_models.tar.gz" \
        -C "${BASE_DIR}" models/ 2>/dev/null
    check_status "Models backup"
    
    # حجم النسخة الاحتياطية
    size=$(du -h "${BACKUP_DIR}/models/${BACKUP_NAME}_models.tar.gz" | cut -f1)
    log "   Backup size: ${size}"
else
    log "${YELLOW}   No models directory found${NC}"
fi

# 3. نسخ احتياطي لقاعدة البيانات
log "\n${YELLOW}2. Backing up database...${NC}"
if [ -f "${BASE_DIR}/data/forex_data.db" ]; then
    cp "${BASE_DIR}/data/forex_data.db" \
       "${BACKUP_DIR}/database/${BACKUP_NAME}_forex_data.db"
    check_status "Database backup"
    
    # ضغط قاعدة البيانات
    gzip "${BACKUP_DIR}/database/${BACKUP_NAME}_forex_data.db"
    check_status "Database compression"
else
    log "${YELLOW}   No database found${NC}"
fi

# 4. نسخ احتياطي للإعدادات والسكريبتات
log "\n${YELLOW}3. Backing up configurations...${NC}"
tar -czf "${BACKUP_DIR}/daily/${BACKUP_NAME}_configs.tar.gz" \
    -C "${BASE_DIR}" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv*' \
    --exclude='logs' \
    --exclude='backups' \
    src/*.py \
    *.py \
    *.sh \
    *.md \
    2>/dev/null
check_status "Configuration backup"

# 5. تنظيف السجلات القديمة
log "\n${YELLOW}4. Cleaning old logs...${NC}"
# حذف السجلات الأقدم من 30 يوم
find "${BASE_DIR}/logs" -name "*.log" -type f -mtime +30 -delete 2>/dev/null
check_status "Old logs cleanup"

# حذف السجلات الكبيرة جداً (أكبر من 100MB)
find "${BASE_DIR}/logs" -name "*.log" -type f -size +100M -exec truncate -s 0 {} \;
check_status "Large logs truncation"

# 6. تنظيف النسخ الاحتياطية القديمة
log "\n${YELLOW}5. Cleaning old backups...${NC}"
# حذف النسخ اليومية الأقدم من 7 أيام
find "${BACKUP_DIR}/daily" -name "*.tar.gz" -type f -mtime +7 -delete 2>/dev/null
check_status "Daily backups cleanup (>7 days)"

# حذف نسخ النماذج الأقدم من 30 يوم
find "${BACKUP_DIR}/models" -name "*.tar.gz" -type f -mtime +30 -delete 2>/dev/null
check_status "Model backups cleanup (>30 days)"

# حذف نسخ قواعد البيانات الأقدم من 14 يوم
find "${BACKUP_DIR}/database" -name "*.db.gz" -type f -mtime +14 -delete 2>/dev/null
check_status "Database backups cleanup (>14 days)"

# 7. التحقق من المساحة المتاحة
log "\n${YELLOW}6. Checking disk space...${NC}"
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
DISK_FREE=$(df -h / | awk 'NR==2 {print $4}')

log "   Disk usage: ${DISK_USAGE}%"
log "   Free space: ${DISK_FREE}"

if [ ${DISK_USAGE} -gt 90 ]; then
    log "${RED}   WARNING: Low disk space!${NC}"
    
    # محاولة تحرير مساحة إضافية
    log "   Attempting to free up space..."
    
    # حذف ملفات temp
    find /tmp -type f -atime +2 -delete 2>/dev/null
    
    # حذف cache
    if [ -d "${BASE_DIR}/__pycache__" ]; then
        find "${BASE_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    fi
    
    # إعادة فحص المساحة
    NEW_DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    log "   New disk usage: ${NEW_DISK_USAGE}%"
fi

# 8. التحقق من صحة النماذج
log "\n${YELLOW}7. Verifying models integrity...${NC}"
MODEL_COUNT=$(find "${BASE_DIR}/models" -name "*.pkl" -type f 2>/dev/null | wc -l)
log "   Total models: ${MODEL_COUNT}"

if [ ${MODEL_COUNT} -eq 0 ]; then
    log "${RED}   WARNING: No models found!${NC}"
fi

# 9. إنشاء تقرير يومي
log "\n${YELLOW}8. Creating daily report...${NC}"
REPORT_FILE="${BASE_DIR}/logs/daily_report_${DATE}.txt"

cat > "${REPORT_FILE}" << EOF
DAILY MAINTENANCE REPORT
========================
Date: $(date)
Hostname: $(hostname)
Uptime: $(uptime)

SYSTEM STATUS:
--------------
CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%
Memory Usage: $(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')
Disk Usage: ${DISK_USAGE}%
Free Space: ${DISK_FREE}

BACKUPS:
--------
Models: $(ls -1 ${BACKUP_DIR}/models/*.tar.gz 2>/dev/null | wc -l) backups
Database: $(ls -1 ${BACKUP_DIR}/database/*.db.gz 2>/dev/null | wc -l) backups
Configs: $(ls -1 ${BACKUP_DIR}/daily/*.tar.gz 2>/dev/null | wc -l) backups

PROCESSES:
----------
Main Server: $(pgrep -f "mt5_bridge_server" >/dev/null && echo "Running" || echo "Stopped")
Advanced Learner: $(screen -ls | grep -q "advanced_unified" && echo "Running" || echo "Stopped")
Continuous Learner: $(screen -ls | grep -q "continuous_unified" && echo "Running" || echo "Stopped")
Dashboard: $(pgrep -f "web_dashboard/app.py" >/dev/null && echo "Running" || echo "Stopped")

MODELS:
-------
Total Models: ${MODEL_COUNT}
Latest Model: $(ls -t ${BASE_DIR}/models/unified/*.pkl 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "None")

RECENT ERRORS:
--------------
$(grep -i "error" ${BASE_DIR}/logs/server.log 2>/dev/null | tail -5 || echo "No recent errors")
EOF

check_status "Daily report creation"

# 10. نسخة احتياطية أسبوعية (يوم الأحد)
if [ $(date +%u) -eq 7 ]; then
    log "\n${YELLOW}9. Creating weekly backup...${NC}"
    
    # نسخة كاملة
    tar -czf "${BACKUP_DIR}/weekly/full_backup_${DATE}.tar.gz" \
        -C "${BASE_DIR}/.." \
        --exclude='forex-ml-trading/venv*' \
        --exclude='forex-ml-trading/backups' \
        --exclude='forex-ml-trading/logs/*.log' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        forex-ml-trading/ 2>/dev/null
    
    check_status "Weekly full backup"
    
    # حذف النسخ الأسبوعية الأقدم من 60 يوم
    find "${BACKUP_DIR}/weekly" -name "*.tar.gz" -type f -mtime +60 -delete 2>/dev/null
    check_status "Weekly backups cleanup (>60 days)"
fi

# 11. إرسال تنبيهات إذا لزم الأمر
if [ ${DISK_USAGE} -gt 90 ] || [ ${MODEL_COUNT} -eq 0 ]; then
    echo "ALERT: System needs attention!" >> "${BASE_DIR}/logs/alerts.log"
    echo "- Disk usage: ${DISK_USAGE}%" >> "${BASE_DIR}/logs/alerts.log"
    echo "- Model count: ${MODEL_COUNT}" >> "${BASE_DIR}/logs/alerts.log"
fi

# النهاية
log "\n========================================
Daily Maintenance Completed: $(date)
========================================"

# حساب الوقت المستغرق
END_TIME=$(date +%s)
START_TIME=$(date -d "$(head -1 ${LOG_FILE} | grep 'Started:' | cut -d':' -f2-)" +%s 2>/dev/null || echo $((END_TIME - 60)))
DURATION=$((END_TIME - START_TIME))

log "Duration: ${DURATION} seconds"
log "\n${GREEN}✓ Maintenance completed successfully!${NC}"

# إرسال النتيجة للـ dashboard
if [ -f "${BASE_DIR}/web_dashboard/system_state.json" ]; then
    jq --arg date "$(date -Iseconds)" \
       --arg status "completed" \
       --arg duration "${DURATION}" \
       '.maintenance = {"last_run": $date, "status": $status, "duration": $duration}' \
       "${BASE_DIR}/web_dashboard/system_state.json" > /tmp/state.json && \
    mv /tmp/state.json "${BASE_DIR}/web_dashboard/system_state.json"
fi

exit 0