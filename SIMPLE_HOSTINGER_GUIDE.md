# 🚀 دليل Hostinger السريع - 5 خطوات فقط!

## 🎯 نعم، يمكنك تشغيل النظام مباشرة على الاستضافة!

### ✅ ما تحتاجه فقط:
1. **اشتراك Hostinger VPS** (خطة VPS 2 كافية)
2. **حساب MetaTrader 5 تجريبي** (مجاني)
3. **30 دقيقة من وقتك**

---

## 📋 الخطوات (انسخ والصق فقط!)

### الخطوة 1️⃣: اشترِ VPS من Hostinger

1. اذهب إلى [Hostinger.com](https://www.hostinger.com)
2. اختر **VPS Hosting** → **VPS 2**
3. نظام التشغيل: **Ubuntu 22.04**
4. ادفع واحصل على:
   - **IP Address**: xxx.xxx.xxx.xxx
   - **Password**: xxxxxxxxx

### الخطوة 2️⃣: احصل على حساب MT5 تجريبي

1. حمل [MetaTrader 5](https://www.metatrader5.com/en/download) على جهازك
2. افتحه → File → Open an Account
3. اختر **Demo Account**
4. احفظ هذه البيانات:
   ```
   Login: 12345678
   Password: yourpassword
   Server: MetaQuotes-Demo
   ```

### الخطوة 3️⃣: اتصل بـ VPS الخاص بك

**على Windows:**
1. اضغط `Win + R`
2. اكتب: `cmd`
3. انسخ والصق:
```bash
ssh root@YOUR_IP_HERE
```
4. اكتب كلمة المرور (لن تظهر وأنت تكتبها)

### الخطوة 4️⃣: انسخ هذه الأوامر (واحد تلو الآخر)

```bash
# 1. تحديث النظام (انتظر 2-3 دقائق)
apt update && apt upgrade -y

# 2. تثبيت الأدوات (انتظر 5 دقائق)
apt install -y python3.9 python3-pip python3-venv git screen

# 3. تحميل النظام
cd /root
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading

# 4. إعداد Python (انتظر 5-10 دقائق)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. إعداد البيانات
cp .env.example .env
nano .env
```

**في محرر nano:**
- غيّر هذه السطور فقط:
```
MT5_LOGIN=ضع_رقم_حسابك_هنا
MT5_PASSWORD=ضع_كلمة_المرور_هنا
MT5_SERVER=ضع_اسم_الخادم_هنا
```
- اضغط `Ctrl+X` ثم `Y` ثم `Enter`

### الخطوة 5️⃣: شغّل النظام!

```bash
# 1. اختبر أولاً
python main.py test

# 2. اجمع البيانات (انتظر 20 دقيقة)
python main.py collect

# 3. تعلم من التاريخ (انتظر 30 دقيقة)
python learn_from_history.py

# 4. ابدأ التداول!
screen -S trading
python main.py trade
```

**للخروج من screen:** اضغط `Ctrl+A` ثم `D`

---

## ✨ النظام الآن يعمل 24/7!

### للمراقبة:
```bash
# شاهد ماذا يحدث
screen -r trading

# شاهد السجلات
tail -f logs/trader.log
```

### التحليل اليومي:
```bash
python run_daily_analysis.py
```

---

## 🆘 إذا واجهت مشكلة:

### مشكلة: "command not found"
```bash
source venv/bin/activate
```

### مشكلة: "MT5 connection failed"
- تأكد من بيانات الحساب في `.env`
- جرب خادم آخر

### مشكلة: "No module named..."
```bash
pip install -r requirements.txt
```

---

## 💡 نصائح مهمة:

1. **استخدم حساب Demo أولاً** (شهر على الأقل)
2. **راقب يومياً في البداية**
3. **النظام يتعلم ويتحسن تلقائياً**
4. **الصبر مفتاح النجاح**

---

## 🎯 ماذا يحدث في الخلفية؟

النظام يقوم بـ:
1. **يتعلم من 3 سنوات بيانات**
2. **يحلل 50+ مؤشر فني**
3. **يبحث عن أفضل الفرص**
4. **يدخل صفقات تلقائياً**
5. **يتعلم من كل صفقة ويتحسن**

---

## 📱 إضافة Telegram (اختياري):

1. ابحث عن `@BotFather` في Telegram
2. أرسل `/newbot`
3. أضف Token في `.env`
4. ستصلك تنبيهات الصفقات!

---

## 🚀 خلاص! النظام شغال!

**تذكر:** 
- النظام يتحسن مع الوقت
- كل صفقة يتعلم منها
- بعد شهر ستلاحظ الفرق

**سؤال؟** الأوامر أعلاه كافية 100% للتشغيل!