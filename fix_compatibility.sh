#!/bin/bash
# إصلاح مشكلة التوافق بين numpy و scikit-learn

echo "🔧 Fixing compatibility issues..."
echo "================================"

# تفعيل البيئة الافتراضية
source venv_pro/bin/activate

# إلغاء تثبيت النسخ الحالية
echo "📦 Uninstalling current versions..."
pip uninstall -y numpy scikit-learn scipy

# تثبيت نسخ متوافقة
echo "📦 Installing compatible versions..."
pip install numpy==1.26.4
pip install scipy==1.12.0
pip install scikit-learn==1.4.2

# إعادة تثبيت المكتبات التي تعتمد عليها
echo "📦 Reinstalling dependent libraries..."
pip install --upgrade lightgbm xgboost pandas

echo ""
echo "✅ Compatibility fix completed!"
echo ""
echo "Now you can run:"
echo "python train_advanced_95_percent.py"