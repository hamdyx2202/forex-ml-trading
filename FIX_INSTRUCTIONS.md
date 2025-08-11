# 🔧 تعليمات إصلاح النظام على Linux VPS

## المشكلة المحلولة:
كان النظام يحاول استخدام MT5 الحقيقي على Linux مما يسبب خطأ في جمع البيانات.

## الحل المطبق:

### 1. **إنشاء data_collector_linux.py**
- يستخدم Yahoo Finance لجمع البيانات على Linux
- يستخدم MT5 على Windows
- يكتشف نظام التشغيل تلقائياً

### 2. **التحديثات المطلوبة:**

```bash
# تحديث المتطلبات
pip install yfinance

# أو
pip install -r requirements.txt
```

### 3. **اختبار النظام:**

```bash
# فحص صحة النظام الكامل
python test_system_health.py

# إعداد قاعدة البيانات (إذا لم تكن موجودة)
python main_linux.py setup

# جمع البيانات
python main.py collect

# تدريب النماذج
python main.py train

# تشغيل الخادم
python main_linux.py server
```

## 📊 ما يحدث الآن:

1. **على Linux VPS:**
   - يستخدم Yahoo Finance لجمع بيانات حقيقية
   - يحفظ البيانات في قاعدة البيانات
   - يدرب النماذج على البيانات الحقيقية
   - يشغل خادم API لاستقبال طلبات EA

2. **على Windows (مع MT5):**
   - EA يرسل البيانات للخادم على Linux
   - يستقبل إشارات التداول من الخادم
   - ينفذ الصفقات على MT5

## ✅ التحقق من العمل:

عند تشغيل `python test_system_health.py` يجب أن ترى:

```
✓ Data collector module loaded
✓ Data collector initialized
✓ Connected to data source
✓ Fetched XXX records
✓ Saved XXX records to database
```

## 🚀 البدء السريع:

```bash
# 1. تفعيل البيئة الافتراضية
source venv/bin/activate

# 2. تثبيت yfinance
pip install yfinance

# 3. إعداد النظام
python main_linux.py setup

# 4. جمع البيانات
python main.py collect

# 5. تدريب النماذج
python main.py train

# 6. تشغيل الخادم
python main_linux.py server
```

## 📌 ملاحظات مهمة:

1. **Yahoo Finance** يوفر بيانات مجانية لكن قد تكون متأخرة 15 دقيقة
2. **حدود المعدل:** لا تجمع البيانات بكثرة لتجنب الحظر
3. **الرموز المدعومة:**
   - EURUSD → EURUSD=X
   - GBPUSD → GBPUSD=X
   - XAUUSD → GC=F (Gold futures)

## 🎯 النتيجة:

النظام الآن يعمل بالكامل على Linux VPS:
- ✅ يجمع بيانات حقيقية
- ✅ يتعلم من التاريخ
- ✅ يدرب النماذج
- ✅ يستقبل طلبات من EA
- ✅ يرسل إشارات التداول

**النظام جاهز للتعلم والتداول! 🤖📈**