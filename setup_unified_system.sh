#!/bin/bash
# Setup Unified System Script
# سكريبت إعداد النظام الموحد

echo "🚀 Setting up Unified ML Trading System..."
echo "========================================"

# ألوان للتوضيح
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. إنشاء المجلدات المطلوبة
echo -e "\n${YELLOW}1. Creating directories...${NC}"
mkdir -p models/unified
mkdir -p models/backup
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"

# 2. التحقق من البيئة الافتراضية
echo -e "\n${YELLOW}2. Checking virtual environment...${NC}"
if [ -d "venv_pro" ]; then
    echo -e "${GREEN}✅ venv_pro found${NC}"
    source venv_pro/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}✅ venv found${NC}"
    source venv/bin/activate
else
    echo -e "${RED}❌ No virtual environment found!${NC}"
    exit 1
fi

# 3. التحقق من المكتبات المطلوبة
echo -e "\n${YELLOW}3. Checking required packages...${NC}"
python -c "import pandas, numpy, sklearn, lightgbm, xgboost, catboost, joblib, loguru" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All packages installed${NC}"
else
    echo -e "${RED}❌ Some packages missing!${NC}"
    echo "Installing missing packages..."
    pip install pandas numpy scikit-learn lightgbm xgboost catboost joblib loguru
fi

# 4. نسخ احتياطية للنماذج الحالية
echo -e "\n${YELLOW}4. Backing up existing models...${NC}"
if [ -d "models/advanced" ] && [ "$(ls -A models/advanced)" ]; then
    backup_dir="models/backup/backup_$(date +%Y%m%d_%H%M%S)"
    cp -r models/advanced "$backup_dir"
    echo -e "${GREEN}✅ Backed up to $backup_dir${NC}"
else
    echo -e "${YELLOW}⚠️ No models to backup${NC}"
fi

# 5. التحقق من النماذج الحالية
echo -e "\n${YELLOW}5. Validating current models...${NC}"
if [ -f "model_validator.py" ]; then
    python model_validator.py
else
    echo -e "${YELLOW}⚠️ model_validator.py not found${NC}"
fi

# 6. إيقاف الأنظمة القديمة
echo -e "\n${YELLOW}6. Stopping old learning systems...${NC}"
pkill -f "advanced_learner_simple.py" 2>/dev/null
pkill -f "continuous_learner_simple.py" 2>/dev/null
echo -e "${GREEN}✅ Old systems stopped${NC}"

# 7. إنشاء ملفات السجلات الأولية
echo -e "\n${YELLOW}7. Creating initial log files...${NC}"
touch models/unified/performance_log.json
touch models/unified/continuous_learning_log.json
touch models/unified/trades_log.json
echo '{"models": {}, "last_update": null}' > models/unified/performance_log.json
echo '{"updates": [], "model_performance": {}}' > models/unified/continuous_learning_log.json
echo '{"trades": [], "last_id": 0}' > models/unified/trades_log.json
echo -e "${GREEN}✅ Log files created${NC}"

# 8. التحقق من قاعدة البيانات
echo -e "\n${YELLOW}8. Checking database...${NC}"
if [ -f "data/forex_data.db" ]; then
    echo -e "${GREEN}✅ Database found${NC}"
    # عرض عدد السجلات
    echo "Database statistics:"
    python -c "
import sqlite3
conn = sqlite3.connect('data/forex_data.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM forex_data')
count = cursor.fetchone()[0]
print(f'  Total records: {count}')
cursor.execute('SELECT DISTINCT symbol FROM forex_data')
symbols = cursor.fetchall()
print(f'  Symbols: {len(symbols)}')
conn.close()
" 2>/dev/null || echo -e "${YELLOW}  Could not read database stats${NC}"
else
    echo -e "${RED}❌ Database not found at data/forex_data.db${NC}"
fi

# 9. إعداد screens للتشغيل
echo -e "\n${YELLOW}9. Setting up screen sessions...${NC}"
echo -e "${GREEN}To start learning systems:${NC}"
echo "  1. Advanced Learner:"
echo "     screen -S advanced_unified"
echo "     python src/advanced_learner_unified.py"
echo ""
echo "  2. Continuous Learner:"
echo "     screen -S continuous_unified"
echo "     python src/continuous_learner_unified.py"
echo ""
echo "  3. To detach: Ctrl+A then D"
echo "  4. To reattach: screen -r [name]"

# 10. التحقق النهائي
echo -e "\n${YELLOW}10. Final checks...${NC}"
echo -e "${GREEN}✅ System files:${NC}"
[ -f "unified_standards.py" ] && echo "  ✓ unified_standards.py" || echo "  ✗ unified_standards.py"
[ -f "src/advanced_learner_unified.py" ] && echo "  ✓ advanced_learner_unified.py" || echo "  ✗ advanced_learner_unified.py"
[ -f "src/continuous_learner_unified.py" ] && echo "  ✓ continuous_learner_unified.py" || echo "  ✗ continuous_learner_unified.py"
[ -f "model_validator.py" ] && echo "  ✓ model_validator.py" || echo "  ✗ model_validator.py"

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\n📋 Next steps:"
echo "1. Review the validation report if any"
echo "2. Start the unified learning systems"
echo "3. Monitor logs/server.log for any issues"
echo ""
echo -e "${YELLOW}⚠️ Important:${NC}"
echo "- Don't run old and new systems together"
echo "- Monitor system performance regularly"
echo "- Check disk space periodically"

# إنشاء ملف status للتحقق السريع
cat > check_status.sh << 'EOF'
#!/bin/bash
echo "🔍 Unified System Status Check"
echo "=============================="
echo ""
echo "📊 Running processes:"
ps aux | grep -E "(advanced|continuous)_learner_unified" | grep -v grep
echo ""
echo "📁 Model files:"
ls -la models/unified/*.pkl 2>/dev/null | wc -l | xargs echo "Total models:"
echo ""
echo "📈 Latest updates:"
if [ -f "models/unified/performance_log.json" ]; then
    echo "Performance log:"
    tail -1 models/unified/performance_log.json | grep -o '"last_update":"[^"]*"'
fi
if [ -f "models/unified/continuous_learning_log.json" ]; then
    echo "Learning log:"
    tail -1 models/unified/continuous_learning_log.json | grep -o '"timestamp":"[^"]*"' | head -1
fi
echo ""
echo "💾 Disk usage:"
du -sh models/unified 2>/dev/null
EOF

chmod +x check_status.sh
echo -e "\n${GREEN}✅ Created check_status.sh for quick status checks${NC}"

echo -e "\n${GREEN}🎉 Unified system setup completed successfully!${NC}"