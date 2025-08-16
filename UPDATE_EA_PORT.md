# تحديث EA لاستخدام المنفذ 5001

## المشكلة:
المنفذ 5000 محجوز على السيرفر

## الحل:
استخدم المنفذ 5001 بدلاً من 5000

## خطوات التحديث في MT5:

### 1. في EA (ForexMLBot_Advanced_V2.mq5):
غير السطر 16:
```mql5
// من:
input string   InpServerURL = "http://localhost:5000/api/predict_advanced";

// إلى:
input string   InpServerURL = "http://localhost:5001/api/predict_advanced";
```

### 2. في إعدادات MT5:
Tools > Options > Expert Advisors > Allow WebRequest:
- أضف: `http://localhost:5001`

### 3. تشغيل الخادم:
```bash
# على السيرفر
python3 mt5_server_port5001.py

# أو في حالة استخدام الخادم الكامل
python3 mt5_prediction_server.py --port 5001
```

### 4. اختبار الاتصال:
```bash
# اختبار الخادم
curl http://localhost:5001/api/health

# أو استخدم
python3 test_minimal_port5001.py
```

## ملاحظة مهمة:
تأكد من تحديث جميع الروابط في EA من 5000 إلى 5001