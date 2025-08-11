# 📚 دليل الإعداد الكامل للمبتدئين

دليل شامل خطوة بخطوة لتشغيل نظام التداول الآلي من الصفر.

## 📌 المتطلبات الأساسية

### 1. **جهاز كمبيوتر** بمواصفات:
- Windows 10/11 أو Mac أو Linux
- ذاكرة RAM: 4 GB على الأقل
- مساحة تخزين: 10 GB فارغة

### 2. **حساب Hostinger VPS**:
- اذهب إلى [Hostinger](https://www.hostinger.com)
- اختر خطة VPS (الحد الأدنى VPS 2)

### 3. **حساب تداول تجريبي**:
- سنستخدم حساب تجريبي للأمان في البداية

---

## 🚀 الجزء الأول: الإعداد على جهازك المحلي

### الخطوة 1: تثبيت Python

1. **تحميل Python:**
   - اذهب إلى [python.org](https://www.python.org/downloads/)
   - اضغط على "Download Python 3.9" أو أحدث
   - **مهم جداً**: عند التثبيت، ضع علامة ✓ على "Add Python to PATH"

2. **التحقق من التثبيت:**
   - افتح Command Prompt (اضغط Win+R واكتب cmd)
   - اكتب:
   ```bash
   python --version
   ```
   - يجب أن ترى: Python 3.9.x

### الخطوة 2: تثبيت MetaTrader 5

1. **تحميل MT5:**
   - اذهب إلى [MetaTrader 5](https://www.metatrader5.com/en/download)
   - حمل وثبت البرنامج

2. **فتح حساب تجريبي:**
   - افتح MT5
   - File → Open an Account
   - ابحث عن "MetaQuotes" واختره
   - اختر "Demo account"
   - املأ البيانات (أي بيانات)
   - **احفظ هذه المعلومات:**
     - Login: (رقم الحساب)
     - Password: (كلمة المرور)
     - Server: (اسم الخادم)

### الخطوة 3: تحميل المشروع

1. **إنشاء مجلد للمشروع:**
   - على سطح المكتب، أنشئ مجلد جديد اسمه "trading"

2. **تحميل الملفات:**
   - حمل ملف ZIP للمشروع
   - فك الضغط في مجلد "trading"

3. **فتح Command Prompt في المجلد:**
   - افتح مجلد "trading/forex-ml-trading"
   - اضغط Shift + Right Click
   - اختر "Open PowerShell window here"

### الخطوة 4: إعداد البيئة

1. **إنشاء البيئة الافتراضية:**
   ```bash
   python -m venv venv
   ```

2. **تفعيل البيئة:**
   ```bash
   # على Windows:
   venv\Scripts\activate
   
   # يجب أن ترى (venv) في بداية السطر
   ```

3. **تثبيت المكتبات:**
   ```bash
   pip install -r requirements.txt
   ```
   (سيستغرق 5-10 دقائق)

### الخطوة 5: إعداد بيانات الحساب

1. **نسخ ملف الإعدادات:**
   ```bash
   copy .env.example .env
   ```

2. **تعديل الملف:**
   - افتح الملف .env بـ Notepad
   - عدّل القيم:
   ```
   MT5_LOGIN=ضع رقم حسابك هنا
   MT5_PASSWORD=ضع كلمة المرور هنا
   MT5_SERVER=ضع اسم الخادم هنا
   ```
   - احفظ الملف

### الخطوة 6: اختبار النظام

1. **اختبر الاتصال:**
   ```bash
   python main.py test
   ```
   
   يجب أن ترى:
   ```
   ✅ Successfully connected to MT5
   Account: 12345678
   Balance: $10000.00
   ```

2. **اجمع البيانات التاريخية:**
   ```bash
   python main.py collect
   ```
   (سيستغرق 10-20 دقيقة)

3. **تعلم من التاريخ:**
   ```bash
   python learn_from_history.py
   ```
   (سيستغرق 30-45 دقيقة)

4. **درب النماذج:**
   ```bash
   python train_models.py
   ```
   (سيستغرق 30-60 دقيقة)

---

## 🌐 الجزء الثاني: النشر على Hostinger VPS

### الخطوة 1: شراء VPS من Hostinger

1. **اذهب إلى Hostinger:**
   - [hostinger.com](https://www.hostinger.com)
   - اختر "VPS Hosting"

2. **اختر الخطة:**
   - اختر "VPS 2" على الأقل
   - نظام التشغيل: Ubuntu 22.04 64-bit
   - الموقع: اختر الأقرب لك

3. **أكمل الشراء:**
   - ستحصل على إيميل به:
     - IP Address: xxx.xxx.xxx.xxx
     - Username: root
     - Password: xxxxxxxxx

### الخطوة 2: الاتصال بـ VPS

1. **على Windows - استخدم PuTTY:**
   - حمل [PuTTY](https://www.putty.org/)
   - افتح PuTTY
   - ضع IP في "Host Name"
   - اضغط "Open"
   - Username: root
   - Password: (الذي حصلت عليه)

2. **أو استخدم PowerShell:**
   ```bash
   ssh root@xxx.xxx.xxx.xxx
   ```

### الخطوة 3: إعداد VPS

1. **نسخ أوامر الإعداد:**
   - افتح الملف `deployment/vps_setup_commands.txt`
   - انسخ الأوامر واحدة تلو الأخرى

2. **تنفيذ الأوامر الأساسية:**
   ```bash
   # تحديث النظام
   apt update && apt upgrade -y
   
   # تثبيت Python
   apt install python3.9 python3-pip python3-venv -y
   
   # إنشاء مستخدم جديد
   adduser trader
   usermod -aG sudo trader
   su - trader
   ```

3. **نسخ المشروع:**
   ```bash
   # استنساخ المشروع
   cd ~
   git clone [رابط_المشروع]
   cd forex-ml-trading
   
   # إنشاء البيئة
   python3 -m venv venv
   source venv/bin/activate
   
   # تثبيت المكتبات
   pip install -r requirements.txt
   ```

4. **نسخ البيانات من جهازك:**
   - ستحتاج لنسخ:
     - ملف .env
     - مجلد data/ (البيانات والنماذج)
   
   استخدم FileZilla أو WinSCP لنقل الملفات

### الخطوة 4: تشغيل النظام

1. **إنشاء خدمة للتشغيل المستمر:**
   ```bash
   sudo nano /etc/systemd/system/trading-bot.service
   ```

2. **الصق هذا المحتوى:**
   ```ini
   [Unit]
   Description=Forex Trading Bot
   After=network.target

   [Service]
   Type=simple
   User=trader
   WorkingDirectory=/home/trader/forex-ml-trading
   ExecStart=/home/trader/forex-ml-trading/venv/bin/python main.py trade
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **احفظ بـ Ctrl+X ثم Y**

4. **شغل الخدمة:**
   ```bash
   sudo systemctl enable trading-bot
   sudo systemctl start trading-bot
   ```

5. **تحقق من الحالة:**
   ```bash
   sudo systemctl status trading-bot
   ```

---

## 📊 الجزء الثالث: المراقبة والمتابعة

### إعداد Telegram للتنبيهات

1. **إنشاء Bot:**
   - افتح Telegram
   - ابحث عن @BotFather
   - أرسل `/newbot`
   - اختر اسم (مثل: MyTradingBot)
   - احفظ الـ Token

2. **الحصول على Chat ID:**
   - ابدأ محادثة مع البوت
   - اذهب إلى: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - ابحث عن "chat_id"

3. **أضف للملف .env:**
   ```
   TELEGRAM_BOT_TOKEN=ضع_التوكن_هنا
   TELEGRAM_CHAT_ID=ضع_الشات_اي_دي_هنا
   ```

### مراقبة الأداء

1. **عبر SSH:**
   ```bash
   # سجلات النظام
   tail -f logs/trader.log
   
   # حالة الخدمة
   sudo systemctl status trading-bot
   ```

2. **عبر Telegram:**
   - ستصلك تنبيهات عن:
     - فتح/إغلاق الصفقات
     - الأرباح والخسائر
     - أخطاء النظام

---

## ⚠️ نصائح مهمة للمبتدئين

### 1. **ابدأ بحساب تجريبي:**
- استخدم حساب Demo لمدة شهر على الأقل
- راقب الأداء يومياً
- لا تنتقل للحساب الحقيقي حتى ترى نتائج إيجابية مستقرة

### 2. **إعدادات الأمان:**
- لا تشارك ملف .env أبداً
- استخدم كلمات مرور قوية
- فعّل المصادقة الثنائية على VPS

### 3. **الصيانة الدورية:**
- حدث البيانات أسبوعياً: `python main.py collect`
- أعد تدريب النماذج شهرياً: `python train_models.py`
- راجع السجلات يومياً

### 4. **حل المشاكل الشائعة:**

**مشكلة: "MT5 connection failed"**
- تأكد من تشغيل MT5
- تحقق من بيانات الحساب في .env
- جرب خادم آخر

**مشكلة: "No data available"**
- شغل: `python main.py collect`
- انتظر حتى اكتمال جمع البيانات

**مشكلة: "Low accuracy"**
- اجمع بيانات أكثر (سنة على الأقل)
- جرب إطارات زمنية أكبر (H4, D1)

---

## 🎯 الخطوات النهائية

1. **تأكد من عمل كل شيء:**
   ```bash
   python main.py test    # اختبار الاتصال
   python main.py collect # جمع البيانات
   python learn_from_history.py # التعلم
   python train_models.py # التدريب
   ```

2. **ابدأ التداول التجريبي:**
   ```bash
   python main.py trade
   ```

3. **راقب لمدة أسبوع:**
   - تحقق من السجلات يومياً
   - راقب الصفقات والنتائج
   - سجل الملاحظات

4. **بعد شهر من النتائج الإيجابية:**
   - يمكنك التفكير في حساب حقيقي صغير
   - ابدأ بـ $500-1000 فقط
   - لا تخاطر بما لا تستطيع خسارته

---

## 📞 الحصول على المساعدة

إذا واجهت أي مشكلة:
1. راجع هذا الدليل مرة أخرى
2. اقرأ السجلات في مجلد logs/
3. ابحث عن رسالة الخطأ في Google
4. اسأل في منتديات Python أو MetaTrader

تذكر: الصبر والتعلم المستمر هما مفتاح النجاح! 🚀