# 🚀 ابدأ من هنا - دليل البداية السريعة

## 📋 الخطوات الأساسية للبدء

### 1️⃣ **التحضير الأولي (30 دقيقة)**

```bash
# 1. افتح Command Prompt في مجلد المشروع
cd forex-ml-trading

# 2. أنشئ البيئة الافتراضية
python -m venv venv

# 3. فعّل البيئة
venv\Scripts\activate

# 4. ثبت المكتبات
pip install -r requirements.txt
```

### 2️⃣ **إعداد حساب التداول (10 دقائق)**

1. افتح MetaTrader 5
2. File → Open an Account → Demo Account
3. احفظ البيانات:
   - Login: XXXXXXXX
   - Password: XXXXXXXX
   - Server: MetaQuotes-Demo

4. عدّل ملف `.env`:
```bash
copy .env.example .env
notepad .env
```

### 3️⃣ **اختبار النظام (5 دقائق)**

```bash
python main.py test
```

يجب أن ترى: ✅ Successfully connected to MT5

### 4️⃣ **جمع البيانات والتعلم (ساعة واحدة)**

```bash
# جمع البيانات التاريخية
python main.py collect

# التعلم من التاريخ
python learn_from_history.py

# تدريب النماذج
python train_models.py
```

### 5️⃣ **البحث عن فرص اليوم (5 دقائق)**

```bash
python run_daily_analysis.py
```

### 6️⃣ **بدء التداول التجريبي**

```bash
python main.py trade
```

---

## 📱 إعداد التنبيهات (اختياري)

1. **Telegram Bot:**
   - ابحث عن @BotFather
   - `/newbot` → اختر اسم
   - احفظ Token
   - أضف في `.env`

---

## 🔍 مراقبة النظام

### لوحة التحكم:
```bash
streamlit run dashboard.py
```
افتح: http://localhost:8501

### السجلات:
```bash
# في نافذة جديدة
tail -f logs/trader.log
```

---

## ⚡ أوامر سريعة

| الأمر | الوظيفة |
|------|---------|
| `python main.py test` | اختبار الاتصال |
| `python main.py collect` | جمع البيانات |
| `python learn_from_history.py` | التعلم من التاريخ |
| `python train_models.py` | تدريب النماذج |
| `python run_daily_analysis.py` | تحليل يومي |
| `python main.py trade` | بدء التداول |
| `streamlit run dashboard.py` | لوحة التحكم |

---

## 🚨 حل المشاكل السريع

### "MT5 connection failed"
- تأكد من تشغيل MT5
- تحقق من بيانات `.env`

### "No data available"
- شغّل: `python main.py collect`

### "Low accuracy"
- شغّل: `python learn_from_history.py`
- ثم: `python train_models.py`

---

## 📞 خطوات النشر على VPS

1. اشترِ VPS من Hostinger
2. اتبع: `docs/BEGINNER_SETUP_GUIDE.md`
3. استخدم: `deployment/vps_setup_commands.txt`

---

## ⏰ الروتين اليومي الموصى به

1. **الصباح (9:00):**
   - `python run_daily_analysis.py`
   - راجع الفرص في `data/daily_opportunities.json`

2. **بدء التداول:**
   - `python main.py trade`

3. **المساء (18:00):**
   - راجع `logs/trader.log`
   - تحقق من الأداء في Dashboard

4. **نهاية الأسبوع:**
   - `python main.py collect`
   - `python train_models.py`

---

## 💡 نصيحة اليوم

> "ابدأ صغيراً، تعلم كثيراً، وكن صبوراً. النجاح في التداول يحتاج وقت وخبرة."

**تذكر:** استخدم حساب Demo لمدة شهر على الأقل قبل التفكير في حساب حقيقي!

---

🎉 **مبروك! أنت جاهز للبدء!**