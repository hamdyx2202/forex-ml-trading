# تعليمات تطبيق إصلاح عدد الميزات
# Instructions to Apply Feature Count Fix

## المشكلة
النماذج تتوقع 70 ميزة لكن الخادم ينتج 49 ميزة فقط

## الحل
تم إنشاء التحديثات التالية لحل المشكلة:

### على VPS (الخادم):

1. **انسخ الملفات المحدثة إلى VPS:**
```bash
# من جهازك المحلي
scp fix_feature_count.py root@69.62.121.53:/home/forex-ml-trading/
scp feature_engineer_adaptive.py root@69.62.121.53:/home/forex-ml-trading/
scp server_70_features.py root@69.62.121.53:/home/forex-ml-trading/
```

2. **على VPS - قم بتفعيل البيئة الافتراضية:**
```bash
ssh root@69.62.121.53
cd /home/forex-ml-trading
source venv_pro/bin/activate  # أو venv حسب اسم بيئتك
```

3. **تشغيل إصلاح عدد الميزات:**
```bash
python fix_feature_count.py
```

4. **إعادة تشغيل الخادم:**
```bash
# إيقاف الخادم الحالي
pkill -f "python.*mt5_bridge_server"

# تشغيل الخادم المحدث
python src/mt5_bridge_server_advanced.py &
```

## التحقق من نجاح الإصلاح

1. **تحقق من سجلات الخادم:**
```bash
tail -f logs/server.log
```

يجب أن ترى:
- "Adding padding features: 49 -> 70" عند معالجة الطلبات
- عدم ظهور خطأ "X has 49 features, but RobustScaler is expecting 70"

2. **اختبر من MT5:**
- شغل EA على الرسم البياني
- انتظر ظهور إشارات التداول

## إذا استمرت المشكلة

جرب الخادم البديل:
```bash
python server_70_features.py &
```

أو أعد تدريب نماذج الطوارئ:
```bash
python emergency_training.py
```

## ملاحظات مهمة

- التحديث يضيف ميزات padding (قيمتها 0) للوصول إلى 70 ميزة
- هذا حل مؤقت حتى يتم إعادة تدريب النماذج بشكل صحيح
- النماذج ستعمل لكن الدقة قد تكون أقل قليلاً

## للمساعدة

إذا واجهت أي مشاكل:
1. تحقق من أن البيئة الافتراضية مفعلة
2. تأكد من أن جميع المكتبات مثبتة
3. راجع سجلات الخادم للأخطاء