#!/bin/bash
# Quick Fix for Missing Import
# إصلاح سريع لـ import المفقود

echo "🔧 Fixing missing import time..."

# إصلاح advanced_learner_unified.py
if grep -q "import time" src/advanced_learner_unified.py; then
    echo "✅ import time already exists in advanced_learner_unified.py"
else
    echo "➕ Adding import time to advanced_learner_unified.py"
    # إضافة import time بعد import json
    sed -i '/import json/a import time' src/advanced_learner_unified.py
    echo "✅ Fixed advanced_learner_unified.py"
fi

# التحقق من continuous_learner_unified.py
if grep -q "import time" src/continuous_learner_unified.py; then
    echo "✅ import time already exists in continuous_learner_unified.py"
else
    echo "➕ Adding import time to continuous_learner_unified.py"
    sed -i '/import json/a import time' src/continuous_learner_unified.py
    echo "✅ Fixed continuous_learner_unified.py"
fi

echo ""
echo "🎉 Fix completed!"
echo ""
echo "📋 To apply on VPS:"
echo "1. Copy this fix:"
echo "   scp quick_fix_import.sh root@69.62.121.53:/home/forex-ml-trading/"
echo ""
echo "2. Run on VPS:"
echo "   ssh root@69.62.121.53"
echo "   cd /home/forex-ml-trading"
echo "   chmod +x quick_fix_import.sh"
echo "   ./quick_fix_import.sh"
echo ""
echo "3. Start the learning systems:"
echo "   screen -S advanced_unified"
echo "   python src/advanced_learner_unified.py"