# تعليمات إصلاح أسماء النماذج
# Instructions to Fix Model Names Mismatch

## المشكلة
الخادم يبحث عن نماذج بأسماء بسيطة، لكن النماذج المحفوظة لها أسماء مع timestamps

```
❌ الخادم يبحث عن: models/advanced/EURJPYm_PERIOD_H1.pkl
✅ الموجود فعلياً: models/advanced/EURJPYm_PERIOD_H1_ensemble_20250814_152901.pkl
```

## الحل السريع

### على VPS:

1. **انسخ ملف الإصلاح:**
```bash
scp fix_model_names_mismatch.py root@69.62.121.53:/home/forex-ml-trading/
scp server_fixed_models.py root@69.62.121.53:/home/forex-ml-trading/
```

2. **على VPS - شغل الإصلاح:**
```bash
ssh root@69.62.121.53
cd /home/forex-ml-trading
source venv_pro/bin/activate

# شغل إصلاح الأسماء
python fix_model_names_mismatch.py
```

3. **إعادة تشغيل الخادم:**
```bash
# إيقاف الخادم الحالي
pkill -f "python.*mt5_bridge_server"

# تشغيل الخادم المحدث
python server_fixed_models.py &
```

## الحلول البديلة

### الحل 1: إنشاء نسخ بأسماء بسيطة
```bash
cd models/advanced
for file in *_ensemble_*.pkl; do
    # استخراج الاسم الأساسي
    base_name=$(echo $file | sed 's/_ensemble_.*//')
    # إنشاء نسخة بالاسم البسيط
    cp "$file" "${base_name}.pkl"
    echo "Created: ${base_name}.pkl"
done
```

### الحل 2: تحديث الخادم للبحث عن النماذج مع timestamps
الملف `server_fixed_models.py` يحتوي على هذا التحديث

### الحل 3: استخدام روابط رمزية
```bash
cd models/advanced
for file in *_ensemble_*.pkl; do
    base_name=$(echo $file | sed 's/_ensemble_.*//')
    ln -s "$file" "${base_name}.pkl"
done
```

## التحقق من نجاح الإصلاح

1. **تحقق من الملفات:**
```bash
ls -la models/advanced/*.pkl | grep -E "(EURJPYm_PERIOD_H1\.pkl|GBPUSDm_PERIOD_M5\.pkl)"
```

2. **تحقق من سجلات الخادم:**
```bash
tail -f logs/server.log
```

يجب أن ترى:
- "✅ Loaded model: EURJPYm_PERIOD_H1"
- عدم ظهور "No model found for EURJPYm_PERIOD_H1"

3. **اختبر من MT5:**
- شغل EA
- يجب أن تحصل على إشارات تداول بدلاً من "NO_TRADE"

## معلومات إضافية

### أسماء النماذج المتوقعة:
```
EURUSDm_PERIOD_M5.pkl
EURUSDm_PERIOD_M15.pkl
EURUSDm_PERIOD_H1.pkl
EURUSDm_PERIOD_H4.pkl
GBPUSDm_PERIOD_M5.pkl
... (وهكذا لجميع الأزواج والأطر الزمنية)
```

### الأزواج المدعومة:
- EURUSDm, GBPUSDm, USDJPYm, USDCHFm
- AUDUSDm, USDCADm, NZDUSDm, EURJPYm

### الأطر الزمنية:
- PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4

## ملاحظات
- الإصلاح لا يؤثر على النماذج نفسها، فقط يحل مشكلة الأسماء
- يمكن حذف النسخ/الروابط القديمة بعد إعادة التدريب بأسماء صحيحة