# 🚨 الحل النهائي لمشكلة الاتصال

## الخطوات بالترتيب:

### 1️⃣ **أعد تشغيل خادم البيانات على Linux:**

```bash
# أوقف الخادم الحالي (Ctrl+C)
# ثم شغله مرة أخرى
cd /path/to/forex-ml-trading
source venv/bin/activate
python start_data_sync_server.py
```

### 2️⃣ **في MT5 - استخدم EA الاختبار أولاً:**

1. احفظ `TestConnection.mq5` في مجلد Experts
2. اجمعه (F7 في MetaEditor)
3. اسحبه على أي chart

### 3️⃣ **في MT5 - الإعدادات الصحيحة:**

#### افتح: Tools → Options → Expert Advisors

ضع علامة ✅ على:
- ✅ Allow automated trading
- ✅ Allow DLL imports (مهم!)
- ✅ Allow WebRequest for listed URL addresses

#### في قسم URLs أضف كل هذه العناوين:

```
http://69.62.121.53:5000
http://69.62.121.53
69.62.121.53
http://69.62.121.53:5000/
```

#### اضغط OK (ليس Apply)

### 4️⃣ **أغلق MT5 وافتحه مرة أخرى**

هذه خطوة مهمة جداً!

### 5️⃣ **شغل TestConnection EA:**

يجب أن ترى في Experts log:
```
Testing connection to: http://69.62.121.53:5000
✅ SUCCESS! Response: {"status":"healthy"...}
```

## 🔴 إذا ظهر Error 4060:

هذا يعني أن WebRequest غير مسموح. الحل:

### في ملف تكوين MT5:
1. أغلق MT5
2. اذهب إلى مجلد MT5:
   ```
   C:\Users\[YourName]\AppData\Roaming\MetaQuotes\Terminal\[ID]\
   ```
3. افتح `terminal.ini` بـ Notepad
4. أضف في النهاية:
   ```
   [Experts]
   AllowWebRequest=1
   WebRequestURL=http://69.62.121.53:5000
   ```
5. احفظ وشغل MT5

## 🟡 حل بديل - استخدام Windows:

إذا فشلت كل المحاولات، شغل الخادم على Windows:

### على Windows (مع MT5):
```cmd
cd C:\path\to\forex-ml-trading
python start_data_sync_server.py
```

### في EA غير ServerURL إلى:
```
ServerURL: http://127.0.0.1:5000
```

## 📝 قائمة تحقق نهائية:

### على Linux VPS:
- [ ] الخادم يعمل على port 5000
- [ ] جدار الحماية يسمح بـ 5000
- [ ] يمكن الوصول من المتصفح

### في MT5:
- [ ] Allow WebRequest مفعل
- [ ] العناوين مضافة في القائمة
- [ ] MT5 تم إعادة تشغيله
- [ ] EA تم إزالته وإضافته

## 🎯 الحل الأكيد 100%:

### استخدم PowerShell كـ Administrator:

```powershell
# أضف استثناء في Windows Firewall
New-NetFirewallRule -DisplayName "MT5 WebRequest" -Direction Outbound -Action Allow -Protocol TCP -RemotePort 5000

# تحقق من الاتصال
Test-NetConnection -ComputerName 69.62.121.53 -Port 5000
```

## 💡 نصيحة أخيرة:

إذا كنت تستخدم VPN أو Proxy:
- أوقفه مؤقتاً
- جرب الاتصال مباشرة

## ✅ عندما ينجح الاتصال:

استخدم `ForexMLDataSyncFixed.mq5` مع:
```
ServerURL: http://69.62.121.53:5000
SymbolsToSync: EURUSD,GBPUSD
```

**يجب أن يعمل الآن!** 🚀