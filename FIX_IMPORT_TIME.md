# إصلاح خطأ Import Time

## المشكلة
```python
NameError: name 'time' is not defined
```

## الحل السريع

### على VPS مباشرة:
```bash
# إضافة import time
sed -i '/import json/a import time' src/advanced_learner_unified.py

# أو يدوياً
nano src/advanced_learner_unified.py
# أضف: import time
# بعد: import json
```

### أو انسخ الملف المُصلح:
```bash
# من جهازك المحلي
scp src/advanced_learner_unified.py root@69.62.121.53:/home/forex-ml-trading/src/
```

## التحقق من الإصلاح
```bash
# تحقق من وجود import
grep "import time" src/advanced_learner_unified.py

# يجب أن يظهر:
# import time
```

## تشغيل النظام بعد الإصلاح
```bash
# Advanced Learner
screen -S advanced_unified
cd /home/forex-ml-trading
source venv_pro/bin/activate
python src/advanced_learner_unified.py

# Ctrl+A ثم D للخروج من screen
```

## ملاحظة
الملف `continuous_learner_unified.py` يحتوي بالفعل على `import time` ولا يحتاج إصلاح.