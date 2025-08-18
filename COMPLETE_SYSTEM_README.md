# 🚀 النظام الكامل - Forex ML Trading System

## ⚡ نظرة سريعة
هذا هو **النظام الكامل** بدون أي تبسيط:
- ✅ 7.8 مليون سجل للتدريب
- ✅ 6 نماذج ML متقدمة
- ✅ 200+ ميزة تقنية
- ✅ 10 فرضيات تداول
- ✅ تعلم مستمر من كل صفقة
- ✅ حساب SL/TP ديناميكي
- ✅ API server على 69.62.121.53:5000

## 🔧 التثبيت على السيرفر Linux

### 1. رفع الملفات للسيرفر
```bash
# من جهازك المحلي
scp COMPLETE_SERVER_SETUP.sh root@69.62.121.53:/root/
scp *.py root@69.62.121.53:/root/
scp -r data root@69.62.121.53:/root/
```

### 2. تشغيل سكريبت التثبيت
```bash
# على السيرفر
ssh root@69.62.121.53
chmod +x COMPLETE_SERVER_SETUP.sh
./COMPLETE_SERVER_SETUP.sh
```

### 3. رفع ملفات النظام
```bash
# بعد انتهاء التثبيت
cd /opt/forex-ml-trading
# ارفع جميع ملفات Python هنا
```

## 📁 الملفات المطلوبة

### ملفات النظام الأساسية:
1. `unified_trading_learning_system.py` - النظام الموحد
2. `unified_prediction_server.py` - سيرفر التنبؤات
3. `train_with_real_data.py` - التدريب على البيانات
4. `live_trading_continuous_learning.py` - التعلم المستمر

### قاعدة البيانات:
- `data/forex_ml.db` - 7.8 مليون سجل

## 🚀 تشغيل النظام

### تشغيل يدوي:
```bash
cd /opt/forex-ml-trading
source venv_forex/bin/activate
python run_complete_system.py
```

### تشغيل كخدمة:
```bash
# السيرفر يعمل تلقائياً كخدمة
systemctl status forex-ml
systemctl restart forex-ml
systemctl stop forex-ml
```

### مراقبة السجلات:
```bash
# سجلات الخدمة
journalctl -u forex-ml -f

# سجلات التطبيق
tail -f /opt/forex-ml-trading/logs/server.log
```

## 📱 إعداد MT5

### 1. نسخ الإكسبيرت
انسخ `ForexMLBot_Advanced_V3_Unified.mq5` إلى مجلد Experts في MT5

### 2. إعدادات الإكسبيرت
```
Server Settings:
- ServerURL: http://69.62.121.53:5000
- ServerTimeout: 5000
- UseRemoteServer: True ✓

Trading Settings:
- LotSize: 0.01
- MinConfidence: 0.65
- CandlesToSend: 200
- MaxPositions: 3

Risk Management:
- UseServerSLTP: True ✓
- MoveToBreakeven: True ✓
- BreakevenPips: 30
```

### 3. السماح بـ WebRequest
Tools → Options → Expert Advisors:
- ✅ Allow automated trading
- ✅ Allow WebRequest for listed URL
- أضف: `http://69.62.121.53:5000`

## 📊 API Endpoints

### GET /status
```bash
curl http://69.62.121.53:5000/status
```
Response:
```json
{
    "status": "running",
    "version": "3.0-complete",
    "models_loaded": 8,
    "uptime_seconds": 3600
}
```

### POST /predict
يستقبل 200 شمعة ويرجع إشارة:
```json
Request:
{
    "symbol": "EURUSDm",
    "timeframe": "M15",
    "candles": [...]
}

Response:
{
    "action": "BUY",
    "confidence": 0.78,
    "sl_price": 1.0950,
    "tp1_price": 1.1050,
    "tp2_price": 1.1100,
    "risk_reward_ratio": 2.0
}
```

### POST /trade_result
يستقبل نتائج الصفقات للتعلم:
```json
{
    "symbol": "EURUSDm",
    "result": "WIN",
    "pips": 50,
    "entry_price": 1.1000,
    "exit_price": 1.1050
}
```

## 🧠 كيف يعمل النظام

### 1. التدريب الأولي
- يستخدم 7.8 مليون سجل
- يدرب 6 نماذج ML لكل زوج/إطار زمني
- يحسب 200+ ميزة تقنية
- يقيّم 10 فرضيات تداول

### 2. التنبؤ
- يستقبل 200 شمعة من MT5
- يحسب الميزات التكيفية
- يستخدم النماذج للتنبؤ
- يطابق الأنماط المربحة
- يحسب SL/TP ديناميكياً

### 3. التعلم المستمر
- يحلل كل صفقة مغلقة
- يكتشف أنماط جديدة
- يحدث أهمية الميزات
- يعيد تدريب النماذج دورياً

## 🔍 حل المشاكل

### السيرفر لا يستجيب
```bash
# تحقق من الخدمة
systemctl status forex-ml

# تحقق من المنفذ
netstat -tlnp | grep 5000

# تحقق من الجدار الناري
ufw status
```

### لا توجد إشارات
1. تحقق من وجود نماذج مدربة
2. تحقق من السجلات
3. تأكد من إرسال 200 شمعة

### أخطاء في MT5
1. تحقق من WebRequest
2. تحقق من عنوان السيرفر
3. زيادة Timeout

## 📈 مراقبة الأداء

### إحصائيات النظام
```bash
# استخدام الذاكرة
free -h

# استخدام المعالج
htop

# مساحة القرص
df -h
```

### إحصائيات التداول
- عدد الإشارات المرسلة
- معدل دقة التنبؤات
- إجمالي النقاط المحققة
- معدل الفوز

## 🔐 الأمان

1. استخدم HTTPS (اختياري)
2. قيّد الوصول بـ IP whitelist
3. أضف API key للحماية
4. نسخ احتياطي دوري

## 🚀 النظام جاهز!

بعد التثبيت وتشغيل السيرفر، النظام سيكون جاهزاً لـ:
- استقبال البيانات من MT5
- إرسال إشارات التداول
- التعلم من النتائج
- التحسن المستمر

**لا يوجد أي تبسيط - هذا هو النظام الكامل بكل الميزات!** 🎉