# 📤 دليل محمل البيانات - ForexMLDataLoader

## 🎯 الهدف:
إرسال البيانات التاريخية من MT5 إلى خادم Linux ليتعلم منها النظام

## ✅ المميزات:

1. **يستخدم نفس طريقة الاتصال الناجحة** من ForexMLBot
2. **يرسل البيانات عبر `/get_signal`** الذي يعمل بالفعل
3. **يحفظ البيانات في قاعدة البيانات** تلقائياً
4. **سهل الاستخدام** - واجهة بسيطة

## 🚀 خطوات الاستخدام:

### 1. على Linux VPS:
```bash
# أعد تشغيل الخادم مع التحديثات
ctrl+c  # لإيقاف الخادم
python src/mt5_bridge_server_linux.py
```

### 2. في MT5:
1. احفظ `ForexMLDataLoader.mq5` في مجلد Experts
2. اجمعه في MetaEditor (F7)
3. اسحبه على أي chart
4. الإعدادات:
   ```
   Server URL: http://69.62.121.53:5000
   History Days: 365  // سنة من البيانات
   Send On Init: true  // إرسال تلقائي
   ```

### 3. ماذا سيحدث:
- EA سيكتشف نهاية الرموز تلقائياً (m, pro, إلخ)
- سيختبر الاتصال
- سيبدأ إرسال البيانات التاريخية
- كل شمعة ترسل كطلب منفصل (مثل EA الأصلي)

## 📊 البيانات المرسلة:

### الأزواج:
- EURUSD, GBPUSD, USDJPY, XAUUSD
- AUDUSD, USDCAD, EURJPY, GBPJPY

### الأطر الزمنية:
- M5, M15, H1, H4, D1

### حجم البيانات:
- سنة كاملة من البيانات
- 8 أزواج × 5 أطر × 365 يوم = آلاف الشموع

## 🔍 مراقبة التقدم:

### في EA:
- شاشة تعرض عدد الشموع المرسلة
- رسائل في Experts log

### على الخادم:
```bash
# مشاهدة السجلات
tail -f logs/mt5_bridge_linux.log

# فحص قاعدة البيانات
sqlite3 data/forex_ml.db "SELECT COUNT(*) FROM price_data;"
```

## 💡 نصائح:

1. **ابدأ بزوج واحد للاختبار:**
   - عدّل الكود وأبقِ فقط EURUSD

2. **لتسريع الإرسال:**
   - قلل InpBatchSize إلى 50
   - أزل Sleep(10)

3. **لإرسال بيانات حديثة:**
   - EA يرسل آخر 10 شموع كل 5 دقائق تلقائياً

## ⚠️ ملاحظات مهمة:

1. **استهلاك النطاق:**
   - كل شمعة = طلب HTTP
   - قد يستغرق وقتاً طويلاً

2. **حجم قاعدة البيانات:**
   - ستكبر بسرعة
   - تأكد من وجود مساحة كافية

3. **الخادم يجب أن يعمل:**
   - تأكد من تشغيل الخادم أولاً
   - أعد تشغيله بعد التحديثات

## ✅ التحقق من النجاح:

### في MT5:
```
✅ Connection successful!
📊 Sending EURUSD M5 - 1000 bars
✅ Sent 100 requests successfully
```

### على Linux:
```bash
# عدد السجلات
sqlite3 data/forex_ml.db "SELECT symbol, COUNT(*) FROM price_data GROUP BY symbol;"

# آخر البيانات
sqlite3 data/forex_ml.db "SELECT * FROM price_data ORDER BY id DESC LIMIT 5;"
```

## 🎯 الخطوة التالية:

بعد تحميل البيانات:
```bash
# تدريب النماذج
python main.py train

# بدء التعلم المستمر
python src/continuous_learner.py
```

**النظام جاهز للتعلم من البيانات!** 🚀