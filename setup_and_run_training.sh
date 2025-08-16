#!/bin/bash
# سكريبت لإعداد البيئة وتشغيل التدريب

echo "🔧 إعداد البيئة الافتراضية..."

# التحقق من البيئة الافتراضية
if [ -d "venv_pro" ]; then
    echo "✅ البيئة الافتراضية موجودة"
    source venv_pro/bin/activate
elif [ -d "../venv_pro" ]; then
    echo "✅ البيئة الافتراضية في المجلد الأعلى"
    source ../venv_pro/bin/activate
else
    echo "❌ البيئة الافتراضية غير موجودة!"
    echo "يرجى إنشاء البيئة أولاً:"
    echo "python3 -m venv venv_pro"
    echo "source venv_pro/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

echo -e "\n📦 المكتبات المثبتة:"
pip list | grep -E "numpy|pandas|scikit-learn|lightgbm|xgboost"

echo -e "\n🚀 تشغيل التدريب..."

# اختبار بسيط أولاً
echo -e "\n1️⃣ اختبار التدريب البسيط..."
python test_simple_training.py

if [ $? -eq 0 ]; then
    echo -e "\n2️⃣ تشغيل التدريب الكامل..."
    echo "هل تريد تشغيل التدريب الكامل؟ (y/n)"
    read -r response
    
    if [ "$response" = "y" ]; then
        python train_full_advanced.py
    fi
else
    echo "❌ فشل الاختبار البسيط"
fi