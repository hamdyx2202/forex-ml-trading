#!/bin/bash
# Quick Fix and Train Script
# سكريبت سريع للإصلاح والتدريب

echo "🚀 Quick Fix and Train"
echo "===================="

# 1. تطبيق إصلاح الأعمدة الفئوية
echo "1️⃣ Fixing categorical columns..."
python fix_categorical_training.py

# 2. محاولة التدريب بطرق مختلفة
echo -e "\n2️⃣ Attempting training..."

# الطريقة 1: التدريب المبسط
if [ -f "train_simple.py" ]; then
    echo "Running simple training..."
    python train_simple.py
    if [ $? -eq 0 ]; then
        echo "✅ Simple training successful!"
    fi
fi

# الطريقة 2: التدريب مع الإصلاح
if [ -f "train_with_fix.py" ]; then
    echo -e "\nRunning training with fix..."
    python train_with_fix.py
    if [ $? -eq 0 ]; then
        echo "✅ Training with fix successful!"
    fi
fi

# 3. التحقق من النماذج
echo -e "\n3️⃣ Checking models..."
model_count=$(find models/advanced -name "*.pkl" 2>/dev/null | wc -l)
echo "Found $model_count models in models/advanced/"

if [ $model_count -gt 0 ]; then
    echo "✅ Models are ready!"
    echo -e "\n🎉 You can now restart the server:"
    echo "   python src/mt5_bridge_server_advanced.py"
else
    echo "❌ No models found yet"
    echo -e "\nTry manual training:"
    echo "   python train_simple.py"
fi