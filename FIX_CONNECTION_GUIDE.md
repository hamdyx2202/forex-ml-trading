# 🔧 دليل حل مشاكل الاتصال والبيانات

## المشكلة 1: ❌ No data for Symbol

### الأسباب:
1. الرمز غير مفعّل في Market Watch
2. لا توجد بيانات تاريخية للرمز
3. الإطار الزمني غير متاح

### الحلول:
```
1. في MT5: View → Symbols → ابحث عن الرمز → Show Symbol
2. انقر بزر الماوس الأيمن على الرمز → Bars → اختر الإطار الزمني
3. انتظر حتى يتم تحميل البيانات
```

## المشكلة 2: ❌ Server error: 1003, Error: 5203

### السبب الرئيسي:
**لم يتم السماح بـ WebRequest في MT5**

### الحل خطوة بخطوة:

### 1️⃣ **السماح بـ WebRequest:**
```
1. في MT5: Tools → Options → Expert Advisors
2. ✅ Allow WebRequest for listed URL addresses
3. أضف عنوان الخادم:
   - http://YOUR_VPS_IP:5000
   - مثال: http://185.224.137.200:5000
4. OK
```

### 2️⃣ **التأكد من تشغيل الخادم على Linux:**
```bash
# على Linux VPS
python start_data_sync_server.py

# يجب أن ترى:
🚀 FOREX ML DATA SYNC SERVER
Starting server on 0.0.0.0:5000
```

### 3️⃣ **اختبار الاتصال من المتصفح:**
```
افتح المتصفح واذهب إلى:
http://YOUR_VPS_IP:5000/health

يجب أن ترى:
{"status": "healthy", ...}
```

### 4️⃣ **فحص جدار الحماية:**
```bash
# على Linux VPS
sudo ufw status

# إذا كان مفعلاً:
sudo ufw allow 5000/tcp
```

## استخدام النسخة المحسنة: ForexMLDataSyncFixed.mq5

### المميزات:
1. ✅ يختبر الاتصال أولاً
2. ✅ يحدد الأزواج يدوياً
3. ✅ يتجاهل الأزواج بدون بيانات
4. ✅ إعادة محاولة عند الفشل
5. ✅ رسائل خطأ واضحة

### الإعدادات الموصى بها:
```
ServerURL: http://YOUR_VPS_IP:5000
APIKey: your_secure_api_key
SymbolsToSync: EURUSD,GBPUSD,USDJPY,XAUUSD
HistoryDays: 365               // سنة واحدة فقط
BatchSize: 500                  // دفعات أصغر
RequestTimeout: 10000           // 10 ثواني
MaxRetries: 3                   // 3 محاولات
SkipMissingData: true          // تجاهل الأزواج الناقصة
```

## خطوات التشغيل الصحيحة:

### 1. على Linux VPS:
```bash
# تشغيل الخادم
cd /path/to/forex-ml-trading
source venv/bin/activate
python start_data_sync_server.py
```

### 2. في MT5:
```
1. Tools → Options → Expert Advisors
2. ✅ Allow WebRequest
3. أضف: http://YOUR_VPS_IP:5000
4. OK
```

### 3. تشغيل EA:
```
1. اسحب ForexMLDataSyncFixed.mq5 على أي chart
2. أدخل IP الخادم الصحيح
3. اضغط "Test Connection" أولاً
4. إذا نجح، اضغط "Start"
```

## نصائح مهمة:

### 1. ابدأ بأزواج قليلة:
```
SymbolsToSync: EURUSD,GBPUSD
```

### 2. استخدم أطر زمنية كبيرة فقط:
النسخة المحسنة تستخدم H1, H4, D1 فقط

### 3. راقب السجلات:
```
✅ Connection successful
✅ Added: EURUSDm
✅ Sent EURUSDm H1 batch 1/5
```

### 4. إذا فشل الاتصال:
```
1. تأكد من IP الصحيح
2. تأكد من تشغيل الخادم
3. تأكد من WebRequest
4. جرب من المتصفح أولاً
```

## الخطأ الأكثر شيوعاً:

**نسيان السماح بـ WebRequest في MT5!**

هذا يسبب Error 5203 دائماً.

## الحل السريع:

استخدم `ForexMLDataSyncFixed.mq5` مع:
- أزواج محددة فقط
- اختبار اتصال أولاً
- تجاهل الأزواج الناقصة

**النظام سيعمل! 🚀**