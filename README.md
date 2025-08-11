# 🤖 Forex ML Trading Bot

نظام تداول آلي متطور يستخدم التعلم الآلي للتنبؤ بحركة أزواج الفوركس والتداول تلقائياً عبر MetaTrader 5.

## 📋 المحتويات

- [المميزات](#المميزات)
- [المتطلبات](#المتطلبات)
- [التثبيت](#التثبيت)
- [الإعداد](#الإعداد)
- [الاستخدام](#الاستخدام)
- [النشر على VPS](#النشر-على-vps)
- [هيكل المشروع](#هيكل-المشروع)
- [الأمان](#الأمان)

## ✨ المميزات

- **التعلم الآلي المتقدم**: استخدام LightGBM و XGBoost للتنبؤ بحركة الأسعار
- **إدارة المخاطر الذكية**: حساب أحجام الصفقات وإدارة المخاطر تلقائياً
- **المراقبة الحية**: تنبيهات Telegram ولوحة تحكم ويب
- **التداول 24/7**: يعمل بشكل مستمر على VPS
- **دعم متعدد الأزواج**: EURUSD, GBPUSD, XAUUSD وأكثر
- **تحليل متعدد الإطارات الزمنية**: من M5 إلى D1

## 🔧 المتطلبات

### البرمجيات المطلوبة:
- Python 3.9 أو أحدث
- MetaTrader 5
- حساب تداول (تجريبي أو حقيقي)
- VPS بنظام Ubuntu 22.04 (للنشر)

### المواصفات الموصى بها:
- RAM: 4 GB
- CPU: 2-4 cores
- Storage: 40 GB SSD
- Internet: اتصال مستقر

## 🚀 التثبيت

### 1. استنساخ المشروع
```bash
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading
```

### 2. إنشاء بيئة Python الافتراضية
```bash
python -m venv venv
source venv/bin/activate  # على Windows: venv\Scripts\activate
```

### 3. تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### 4. إعداد الملفات
```bash
cp .env.example .env
# قم بتحرير .env وأضف بيانات حسابك
```

## ⚙️ الإعداد

### 1. إعداد MetaTrader 5

1. قم بتثبيت MetaTrader 5
2. افتح حساب تجريبي أو استخدم حسابك الحقيقي
3. احصل على معلومات الاتصال:
   - رقم الحساب (Login)
   - كلمة المرور
   - اسم الخادم

### 2. تحديث ملف `.env`
```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. إعداد Telegram Bot (اختياري)

1. ابحث عن @BotFather في Telegram
2. أنشئ bot جديد: `/newbot`
3. احصل على token
4. ابدأ محادثة مع البوت واحصل على chat ID

## 📖 الاستخدام

### اختبار الاتصال
```bash
python main.py test
```

### جمع البيانات التاريخية
```bash
python main.py collect
```

### تدريب النماذج
```bash
python main.py train
# أو
python train_models.py
```

### بدء التداول الآلي
```bash
python main.py trade
```

### بدء خدمة المراقبة
```bash
python main.py monitor
```

### تشغيل لوحة التحكم
```bash
streamlit run dashboard.py
```

## 🌐 النشر على Hostinger VPS

### 1. إعداد VPS

1. احصل على VPS من Hostinger بنظام Ubuntu 22.04
2. اتصل بالخادم عبر SSH:
```bash
ssh username@your_vps_ip
```

### 2. تشغيل سكريبت الإعداد

```bash
# انسخ المشروع للخادم
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading

# شغل سكريبت الإعداد
chmod +x deployment/setup_vps.sh
./deployment/setup_vps.sh
```

### 3. بدء الخدمات

```bash
# بدء البوت
sudo systemctl start forex-trading-bot

# بدء المراقبة
sudo systemctl start forex-monitoring

# بدء لوحة التحكم (اختياري)
sudo systemctl start forex-dashboard
```

### 4. التحقق من الحالة

```bash
# حالة الخدمات
sudo systemctl status forex-trading-bot
sudo systemctl status forex-monitoring

# عرض السجلات
tail -f logs/trader.log
journalctl -u forex-trading-bot -f
```

## 📁 هيكل المشروع

```
forex-ml-trading/
├── config/              # ملفات الإعدادات
├── data/               # البيانات والنماذج
├── src/                # الكود المصدري
│   ├── data_collector.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   ├── predictor.py
│   ├── trader.py
│   ├── risk_manager.py
│   └── monitor.py
├── deployment/         # سكريبتات النشر
├── logs/              # ملفات السجلات
├── main.py           # نقطة البداية
├── dashboard.py      # لوحة التحكم
└── requirements.txt  # المكتبات المطلوبة
```

## 🔐 الأمان

### نصائح مهمة:

1. **لا تشارك بيانات حسابك أبداً**
2. **استخدم حساب تجريبي للاختبار أولاً**
3. **قم بتشفير ملف `.env`**
4. **استخدم جدار حماية على VPS**
5. **قم بعمل نسخ احتياطية دورية**

### إعداد جدار الحماية:
```bash
sudo ufw allow ssh
sudo ufw allow 8501  # للوحة التحكم
sudo ufw enable
```

## 📊 مؤشرات الأداء

النظام يستهدف:
- معدل دقة: > 65%
- نسبة شارب: > 1.2
- أقصى تراجع: < 15%
- معدل الربح الشهري: 5-10%

## 🛠️ استكشاف الأخطاء

### مشاكل الاتصال بـ MT5:
- تأكد من تثبيت MT5
- تحقق من بيانات الحساب
- تأكد من السماح للخبراء الآليين

### مشاكل التدريب:
- تحقق من وجود بيانات كافية
- زد حجم الذاكرة المتاحة
- قلل حجم البيانات المستخدمة

### مشاكل VPS:
- تأكد من المتطلبات الدنيا
- راجع السجلات: `journalctl -xe`
- تحقق من المساحة المتاحة: `df -h`

## 📞 الدعم

للمساعدة أو الإبلاغ عن مشاكل:
- افتح issue على GitHub
- راسلنا على [email@example.com]

## ⚖️ إخلاء المسؤولية

**تحذير**: التداول في الفوركس ينطوي على مخاطر عالية. هذا النظام للأغراض التعليمية والبحثية. استخدمه على مسؤوليتك الخاصة.

## 📜 الترخيص

MIT License - راجع ملف LICENSE للتفاصيل