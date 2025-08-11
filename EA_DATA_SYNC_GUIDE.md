# 📡 دليل مزامنة البيانات من MT5 إلى Linux VPS

## 🎯 الحل النهائي: EA يرسل البيانات مباشرة لـ Linux

### المميزات:
- ✅ بيانات حقيقية 100% من MT5
- ✅ تحديث تلقائي مستمر
- ✅ النظام يتعلم من نفس بيانات المنصة
- ✅ لا حاجة لنقل ملفات يدوياً

## 🚀 خطوات التشغيل:

### 1️⃣ على Linux VPS:

```bash
# تشغيل خادم استقبال البيانات
python start_data_sync_server.py

# أو مع تحديد المنفذ
python start_data_sync_server.py 0.0.0.0 5000
```

سترى:
```
🚀 FOREX ML DATA SYNC SERVER
====================================
Starting server on 0.0.0.0:5000
Ready to receive data from MT5 EA...
```

### 2️⃣ على Windows (MT5):

1. **نسخ EA إلى MT5:**
   - انسخ `ForexMLDataSync.mq5` إلى:
   - `C:\Users\[YourName]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts\`

2. **تجميع EA:**
   - افتح MetaEditor
   - افتح الملف وأضغط F7 للتجميع

3. **إعدادات EA:**
   ```
   ServerURL: http://YOUR_VPS_IP:5000
   APIKey: your_secure_api_key
   HistoryDays: 1095 (3 سنوات)
   UpdateIntervalSeconds: 300 (5 دقائق)
   SendHistoricalData: true
   SendLiveData: true
   AutoStart: true
   ```

4. **السماح بـ WebRequest:**
   - Tools → Options → Expert Advisors
   - ✅ Allow WebRequest for listed URL
   - أضف: `http://YOUR_VPS_IP:5000`

5. **تشغيل EA:**
   - اسحب EA على أي chart
   - سيبدأ إرسال البيانات تلقائياً

## 📊 ماذا يحدث:

### عند التشغيل الأول:
1. EA يرسل 3 سنوات من البيانات التاريخية
2. جميع الأزواج والأطر الزمنية المحددة
3. البيانات تُحفظ في قاعدة بيانات Linux

### التحديث المستمر:
- كل 5 دقائق يرسل آخر البيانات
- النظام يتعلم من البيانات الجديدة
- لا تفوت أي حركة في السوق

## 🔧 التحكم في EA:

### الأزرار على الشارت:
- **Start Sync**: بدء المزامنة
- **Stop Sync**: إيقاف المزامنة
- **Send History**: إرسال التاريخ مرة أخرى
- **Test Connection**: اختبار الاتصال

### مراقبة الحالة:
- يعرض عدد الشموع المرسلة
- يعرض حالة الاتصال
- ينبهك عند أي مشكلة

## 📈 التحقق من البيانات على Linux:

```bash
# عرض إحصائيات البيانات
curl http://localhost:5000/api/stats

# عرض بيانات زوج معين
curl http://localhost:5000/api/data/EURUSD/H1?limit=100
```

## 🔒 الأمان:

1. **تغيير API Key:**
   ```bash
   # على Linux
   export FOREXTML_API_KEY="your_new_secure_key"
   
   # في EA
   APIKey: "your_new_secure_key"
   ```

2. **تقييد الوصول (اختياري):**
   ```bash
   # استخدم firewall
   sudo ufw allow from YOUR_MT5_IP to any port 5000
   ```

## 💡 نصائح مهمة:

1. **الأزواج المدعومة:**
   - يمكن تعديلها في كود EA
   - الافتراضي: EURUSD, GBPUSD, USDJPY, XAUUSD, GBPJPY, AUDUSD

2. **الأطر الزمنية:**
   - M5, M15, H1, H4, D1
   - يمكن إضافة المزيد

3. **حجم البيانات:**
   - 3 سنوات × 6 أزواج × 5 أطر = الكثير من البيانات
   - قد يستغرق الإرسال الأول 10-30 دقيقة

4. **استهلاك الإنترنت:**
   - الإرسال الأول: ~50-200 MB
   - التحديثات: ~1-5 MB كل 5 دقائق

## ✅ التحقق من النجاح:

### على MT5:
- Status: Running (أخضر)
- Sent: XXXXX bars

### على Linux:
```bash
# فحص قاعدة البيانات
python test_system_health.py

# تدريب النماذج بالبيانات الجديدة
python main.py train
```

## 🎯 النتيجة النهائية:

- ✅ بيانات MT5 الحقيقية على Linux
- ✅ تحديث تلقائي مستمر
- ✅ النظام يتعلم من نفس بيانات التداول
- ✅ جاهز للتداول الحقيقي

**مبروك! النظام الآن يعمل بالبيانات الحقيقية! 🚀**