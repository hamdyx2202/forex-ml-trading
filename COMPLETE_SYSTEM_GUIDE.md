# 📚 دليل النظام الكامل للتداول بالذكاء الاصطناعي

## 🚀 النظام الكامل يشمل:

### 1. **6 نماذج ذكاء اصطناعي**
- LightGBM - للدقة العالية
- XGBoost - للأداء المتوازن
- Random Forest - للاستقرار
- Gradient Boosting - للتحسين التدريجي
- Extra Trees - للتنوع
- Neural Network - للأنماط المعقدة

### 2. **200+ ميزة تقنية**
- المتوسطات المتحركة (SMA, EMA)
- مؤشرات الزخم (RSI, MACD, Stochastic)
- مؤشرات التذبذب (Bollinger Bands, ATR)
- مؤشرات الحجم
- الأنماط الشمعية (30+ نمط)
- مستويات الدعم والمقاومة
- الميزات الزمنية (الجلسات، الأيام)
- مؤشرات السوق (ADX)

### 3. **10 فرضيات تداول**
1. Trend Following - متابعة الاتجاه
2. Mean Reversion - العودة للمتوسط
3. Momentum - الزخم
4. Volatility Breakout - اختراق التذبذب
5. Seasonality - الموسمية
6. Support/Resistance - الدعم والمقاومة
7. Market Structure - بنية السوق
8. Volume Analysis - تحليل الحجم
9. Pattern Recognition - التعرف على الأنماط
10. Correlation - الارتباط

### 4. **التعلم المستمر**
- تحليل الصفقات المغلقة
- اكتشاف الأنماط الرابحة
- تحديث النماذج تلقائياً
- التكيف مع ظروف السوق

### 5. **حساب SL/TP الديناميكي**
- يستخدم ATR للتقلب
- يتكيف مع قوة الترند (ADX)
- نسب مخاطرة/مكافأة متغيرة
- يراعي الجلسات التداولية

## 📋 خطوات التشغيل على السيرفر:

### 1. **رفع الملفات**
```bash
# من جهازك المحلي
scp complete_forex_ml_server.py root@69.62.121.53:/home/forex-ml-trading/
scp train_all_models.py root@69.62.121.53:/home/forex-ml-trading/
scp run_complete_system_server.sh root@69.62.121.53:/home/forex-ml-trading/
```

### 2. **على السيرفر Linux**
```bash
ssh root@69.62.121.53
cd /home/forex-ml-trading

# تفعيل البيئة الافتراضية
source venv_pro/bin/activate

# تدريب النماذج (مرة واحدة)
python3 train_all_models.py

# تشغيل السيرفر
python3 complete_forex_ml_server.py
```

### 3. **تشغيل في الخلفية**
```bash
chmod +x run_complete_system_server.sh
./run_complete_system_server.sh background
```

## 🔍 نقاط نهاية API:

### 1. **/status** - حالة السيرفر
```bash
curl http://69.62.121.53:5000/status
```

### 2. **/predict** - طلب إشارة
```json
POST /predict
{
    "symbol": "EURUSD",
    "timeframe": "M15",
    "candles": [
        {
            "time": "2024-01-01 12:00:00",
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
            "volume": 1000
        }
        // ... 200 شمعة
    ]
}
```

الاستجابة:
```json
{
    "symbol": "EURUSD",
    "timeframe": "M15",
    "action": "BUY",
    "confidence": 0.75,
    "current_price": 1.0855,
    "sl_price": 1.0805,
    "tp1_price": 1.0955,
    "tp2_price": 1.1005,
    "sl_pips": 50,
    "tp1_pips": 100,
    "tp2_pips": 150,
    "risk_reward_ratio": 2.0,
    "models_used": 6,
    "features_count": 200
}
```

### 3. **/trade_result** - تسجيل نتيجة صفقة
```json
POST /trade_result
{
    "symbol": "EURUSD",
    "timeframe": "M15",
    "entry_price": 1.0850,
    "exit_price": 1.0950,
    "profit_pips": 100,
    "exit_reason": "TP1"
}
```

### 4. **/train** - تدريب يدوي
```json
POST /train
{
    "symbol": "EURUSD",
    "timeframe": "H1"
}
```

### 5. **/models** - قائمة النماذج
```bash
curl http://69.62.121.53:5000/models
```

## 🛠️ حل المشاكل:

### مشكلة JSON الكبير
السيرفر الآن يتعامل مع:
- JSON حتى 50MB
- إصلاح JSON المكسور تلقائياً
- استخدام البيانات الجزئية إذا لزم
- معالجة آمنة للأخطاء

### النماذج غير محملة
```bash
# تدريب جميع النماذج
python3 train_all_models.py

# أو تدريب زوج محدد
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","timeframe":"M15"}'
```

### مراقبة الأداء
```bash
# السجلات الحية
tail -f complete_forex_ml_server.log

# إحصائيات السيرفر
curl http://69.62.121.53:5000/status
```

## 📊 في MT5:

### إعدادات الإكسبيرت
```
ServerURL: http://69.62.121.53:5000
UseRemoteServer: true
MinConfidence: 0.65
CandlesToSend: 200
```

### السماح بـ WebRequest
Tools → Options → Expert Advisors:
- ✅ Allow automated trading
- ✅ Allow WebRequest for listed URL
- أضف: `http://69.62.121.53:5000`

## ✅ النظام جاهز!

السيرفر الآن:
- يستقبل بيانات من جميع العملات
- يدعم جميع الأطر الزمنية
- يتدرب من 7.8M سجل
- يتعلم باستمرار من الصفقات
- يحسب SL/TP ديناميكياً
- يستخدم 6 نماذج ML
- يقيّم 10 فرضيات
- يحسب 200+ ميزة

**لا يوجد أي تبسيط - هذا هو النظام الكامل!**