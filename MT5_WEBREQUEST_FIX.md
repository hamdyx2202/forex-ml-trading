# 🚨 حل مشكلة WebRequest في MT5 - خطوة بخطوة

## المشكلة:
رغم وضع عنوان الخادم الصحيح، MT5 يرفض الاتصال!

## ✅ الحل الكامل:

### 1️⃣ **افتح إعدادات MT5:**
```
Tools → Options
أو
أدوات → خيارات
```

### 2️⃣ **اذهب إلى تبويب Expert Advisors:**
![Expert Advisors Tab]

### 3️⃣ **ضع علامة ✅ على الخيارات التالية:**
```
✅ Allow automated trading
✅ Allow DLL imports
✅ Allow WebRequest for listed URL addresses
```

### 4️⃣ **أضف عنوان الخادم في القائمة:**

#### الطريقة الصحيحة:
1. في مربع "Add new URL..." اكتب:
   ```
   http://69.62.121.53:5000
   ```
2. اضغط زر "Add" أو Enter
3. تأكد أن العنوان ظهر في القائمة

#### ⚠️ أخطاء شائعة:
```
❌ خطأ: http://69.62.121.53:5000/
✅ صحيح: http://69.62.121.53:5000

❌ خطأ: 69.62.121.53:5000
✅ صحيح: http://69.62.121.53:5000

❌ خطأ: https://69.62.121.53:5000
✅ صحيح: http://69.62.121.53:5000
```

### 5️⃣ **اضغط OK وأعد تشغيل EA:**
1. OK للحفظ
2. أزل EA من الشارت
3. أضفه مرة أخرى

## 🔍 للتأكد من الإعدادات:

### في MT5:
1. View → Terminal → Experts
2. يجب أن ترى:
   ```
   ForexMLDataSyncFixed: initialized
   ForexMLDataSyncFixed: WebRequest allowed
   ```

### اختبار سريع من المتصفح:
افتح المتصفح واذهب إلى:
```
http://69.62.121.53:5000/health
```

إذا ظهر:
```json
{"status": "healthy", "service": "ForexML Data Sync Server"}
```
= الخادم يعمل ✅

## 📝 قائمة تحقق نهائية:

### على Linux VPS:
```bash
# تأكد أن الخادم يعمل
curl http://localhost:5000/health

# تأكد من المنفذ مفتوح
sudo netstat -tlnp | grep 5000

# تأكد من جدار الحماية
sudo ufw status
sudo ufw allow 5000/tcp
```

### في MT5:
- [ ] Tools → Options → Expert Advisors
- [ ] ✅ Allow WebRequest for listed URL addresses
- [ ] العنوان موجود في القائمة: http://69.62.121.53:5000
- [ ] اضغط OK
- [ ] أعد تشغيل EA

## 🎯 حل بديل إذا استمرت المشكلة:

### 1. جرب بدون منفذ:
أضف هذه العناوين أيضاً:
```
http://69.62.121.53
http://69.62.121.53/
69.62.121.53
```

### 2. تأكد من إصدار MT5:
```
Help → About
يجب أن يكون Build 3000 أو أحدث
```

### 3. أعد تشغيل MT5 بالكامل:
1. أغلق MT5
2. افتحه مرة أخرى
3. جرب EA

## 💡 نصيحة مهمة:

**بعد إضافة العنوان في WebRequest، يجب:**
1. الضغط OK (ليس Apply)
2. إزالة EA من الشارت
3. إضافته مرة أخرى

## 🔴 إذا ظهر الخطأ مرة أخرى:

تحقق من Experts log:
```
View → Terminal → Experts

ابحث عن:
"WebRequest is not allowed"
"Error 4060"
"Error 5203"
```

إذا رأيت هذه الأخطاء = لم تضف العنوان بشكل صحيح في WebRequest

## ✅ عند النجاح ستري:

```
ForexMLDataSyncFixed: Testing connection...
ForexMLDataSyncFixed: ✅ Connection successful!
ForexMLDataSyncFixed: Ready to sync data
```

**لا تنسى: يجب إضافة http:// في بداية العنوان!**