# 📈 دليل التداول الحقيقي - Forex ML Trading System

## 🎯 نظرة عامة على الحل

تم حل مشكلة عدم عمل MT5 على Linux VPS من خلال:
1. **Expert Advisor على Windows** - ينفذ الصفقات الحقيقية
2. **Python ML Server على Linux VPS** - يحلل ويعطي الإشارات
3. **Bridge API** - يربط بين الاثنين

```
[MT5 على Windows] <--HTTP--> [Python Server على Linux VPS]
      ↓                                    ↓
  صفقات حقيقية                     تحليل ذكي + تعلم مستمر
```

## 🚀 خطوات التشغيل الكامل

### الخطوة 1: إعداد Linux VPS (نظام ML)

```bash
# 1. الاتصال بـ VPS
ssh ubuntu@69.62.121.53

# 2. استنساخ المشروع
git clone https://github.com/your-repo/forex-ml-trading.git
cd forex-ml-trading

# 3. إعداد البيئة
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. إعداد قاعدة البيانات
python main.py test

# 5. جمع البيانات التاريخية
python main.py collect

# 6. التعلم من التاريخ
python learn_from_history.py

# 7. تدريب النماذج
python train_models.py

# 8. تشغيل خادم الجسر
python start_bridge_server.py
```

### الخطوة 2: إعداد MT5 على Windows

```
1. تثبيت MetaTrader 5
2. فتح حساب (تجريبي أولاً)
3. نسخ ForexMLBot.mq5 إلى:
   C:\...\MQL5\Experts\
4. تجميع EA في MetaEditor
5. السماح بـ WebRequest لعنوان VPS
6. تشغيل EA على الرسم البياني
```

### الخطوة 3: تكوين الاتصال

في إعدادات EA:
```
PythonServerURL: http://69.62.121.53:5000
RiskPerTrade: 0.01 (ابدأ بـ 1%)
MaxPositions: 1 (ابدأ بصفقة واحدة)
```

## 📊 المراقبة والمتابعة

### 1. من Linux VPS:
```bash
# سجلات الخادم
tail -f logs/mt5_bridge.log

# حالة النظام
curl http://localhost:5000/status

# واجهة المراقبة
streamlit run dashboard.py
```

### 2. من MT5:
- Terminal → Trade (الصفقات المفتوحة)
- Terminal → History (التاريخ)
- Terminal → Experts (سجل EA)

### 3. من Telegram:
```bash
# تشغيل البوت
python main.py telegram
```

## 🔄 دورة العمل الكاملة

1. **EA يفحص السوق** كل 60 ثانية
2. **يرسل السعر الحالي** لخادم Python
3. **Python يحلل** باستخدام ML + 50 مؤشر
4. **يرجع إشارة** (BUY/SELL/NO_TRADE)
5. **EA ينفذ الصفقة** إذا كانت الثقة عالية
6. **Python يتعلم** من نتيجة الصفقة

## ⚙️ الإعدادات المتقدمة

### تشغيل الخادم كخدمة دائمة:
```bash
# نسخ ملف الخدمة
sudo cp scripts/mt5-bridge.service /etc/systemd/system/

# تفعيل وتشغيل
sudo systemctl enable mt5-bridge
sudo systemctl start mt5-bridge

# مراقبة الحالة
sudo systemctl status mt5-bridge
```

### فتح جدار الحماية:
```bash
# السماح بالمنفذ 5000
sudo ufw allow 5000/tcp
sudo ufw reload
```

### إعداد HTTPS (اختياري):
```bash
# استخدام nginx كـ reverse proxy
sudo apt install nginx
sudo nano /etc/nginx/sites-available/mt5-bridge
```

## 🛡️ الأمان والحماية

1. **استخدم كلمات مرور قوية** لـ VPS و MT5
2. **قيّد الوصول للـ API** بـ IP محدد
3. **ابدأ بحساب تجريبي** لمدة شهر على الأقل
4. **ضع Stop Loss** لكل صفقة
5. **راقب يومياً** في البداية

## 📈 تحسين الأداء

### بعد أسبوع:
```bash
# تحليل الأداء
python analyze_performance.py

# تحسين النماذج
python auto_improve.py
```

### بعد شهر:
- راجع معدل النجاح
- اضبط حجم المخاطرة
- أضف أزواج جديدة
- حسّن الاستراتيجية

## 🚨 حل المشاكل الشائعة

### مشكلة: EA لا يتصل
```bash
# تحقق من تشغيل الخادم
curl http://69.62.121.53:5000/health

# تحقق من السجلات
sudo journalctl -u mt5-bridge -n 50
```

### مشكلة: لا توجد صفقات
- تحقق من وجود نماذج مدربة
- تأكد من فتح السوق
- راجع إعدادات الثقة

### مشكلة: خسائر متتالية
- قلل حجم المخاطرة
- أوقف التداول مؤقتاً
- أعد تدريب النماذج

## 📞 الدعم

- **السجلات**: `logs/` للتفاصيل
- **التوثيق**: `docs/` للمساعدة
- **النسخ الاحتياطي**: احفظ `models/` و `data/`

## ✅ قائمة التحقق قبل البدء

- [ ] VPS يعمل بـ Ubuntu 22.04
- [ ] Python 3.10+ مثبت
- [ ] قاعدة البيانات تعمل
- [ ] نماذج مدربة
- [ ] خادم الجسر يعمل
- [ ] EA مثبت على MT5
- [ ] حساب تجريبي جاهز
- [ ] Stop Loss محدد
- [ ] نسخة احتياطية موجودة

## 🎉 البدء!

```bash
# على VPS
cd ~/forex-ml-trading
source venv/bin/activate
python start_bridge_server.py

# على MT5
- شغّل EA على EURUSD H1
- راقب أول صفقة
- احتفل بالنجاح! 🚀
```