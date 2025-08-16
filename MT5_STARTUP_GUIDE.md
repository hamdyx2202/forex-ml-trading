# دليل تشغيل نظام MT5 المتكامل
# MT5 Integration System Startup Guide

## 🚀 خطوات البدء السريع (Quick Start)

### 1️⃣ تشغيل خادم التنبؤات (بدون dependencies)
```bash
# Terminal 1 - خادم اختباري بسيط
python3 start_mt5_server_simple.py
```

### 2️⃣ اختبار الاتصال
```bash
# Terminal 2 - اختبار الخادم
python3 test_mt5_simple.py
```

### 3️⃣ إعداد MT5
1. افتح MetaTrader 5
2. انسخ `ForexMLBot_Advanced_V2.mq5` إلى: `MQL5/Experts/`
3. اضغط F7 في MetaEditor للتجميع
4. في MT5: Tools > Options > Expert Advisors
   - ✅ Allow WebRequest for listed URL
   - أضف: `http://localhost:5000`

## 📊 النظام الكامل (مع النماذج المدربة)

### المتطلبات:
- Python 3.8+
- مكتبات ML (numpy, pandas, scikit-learn, etc.)
- نماذج مدربة في مجلد `models/`

### تشغيل الخادم الكامل:
```bash
# إنشاء بيئة افتراضية (مرة واحدة فقط)
python3 -m venv venv_pro
source venv_pro/bin/activate  # Linux/Mac
# أو
venv_pro\Scripts\activate  # Windows

# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل الخادم الكامل
python mt5_prediction_server.py
```

## 🔧 حل المشاكل الشائعة

### مشكلة: "No module named 'numpy'"
```bash
# حل سريع - استخدم الخادم البسيط
python3 start_mt5_server_simple.py
```

### مشكلة: "WebRequest failed" في MT5
1. تأكد من إضافة URL في إعدادات MT5
2. جرب `http://127.0.0.1:5000` بدلاً من `localhost`
3. تأكد من تشغيل الخادم

### مشكلة: "No models found"
- استخدم الخادم البسيط للاختبار أولاً
- الخادم البسيط يرسل تنبؤات وهمية للتأكد من عمل النظام

## 📈 مراقبة النظام

### في MT5 Journal:
```
🚀 بدء تشغيل ForexMLBot Advanced V2
📊 السيرفر: http://localhost:5000/api/predict_advanced
✅ عدد الأزواج المتاحة: 19 من 19
🎯 إشارة جديدة! EURUSD M5
```

### في Terminal (Python):
```
🚀 Starting MT5 Prediction Server
📊 Server URL: http://localhost:5000
📊 Received prediction request: EURUSD M5
✅ Received trade result: Ticket #12345678
```

## 💡 نصائح مهمة

1. **ابدأ بالخادم البسيط** للتأكد من عمل الاتصال
2. **راقب Journal في MT5** لمعرفة أي أخطاء
3. **استخدم فترة زمنية صغيرة** (M5) للاختبار السريع
4. **ابدأ بزوج واحد** (EURUSD) قبل تفعيل جميع الأزواج

## 📋 الملفات المهمة

- **EA**: `ForexMLBot_Advanced_V2.mq5`
- **خادم بسيط**: `start_mt5_server_simple.py`
- **خادم كامل**: `mt5_prediction_server.py`
- **اختبار**: `test_mt5_simple.py`
- **دليل الإعداد**: `mt5_ea_setup_guide.md`

## 🎯 الخطوات التالية

بعد نجاح الاختبار البسيط:
1. درب النماذج باستخدام `train_full_advanced.py`
2. شغل الخادم الكامل `mt5_prediction_server.py`
3. راقب الأداء وحسّن الإعدادات