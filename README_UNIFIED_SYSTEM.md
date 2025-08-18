# 🚀 Unified Forex ML Trading System

## 📊 نظرة عامة
نظام متكامل للتداول بالذكاء الاصطناعي يجمع بين:
- 📈 التدريب على 7.8 مليون سجل تاريخي
- 🤖 6 نماذج ML متقدمة
- 📡 سيرفر موحد يستقبل 200 شمعة
- 🎯 حساب ديناميكي لـ SL/TP
- 🧠 تعلم مستمر من كل صفقة
- 🔄 تحديث تلقائي للنماذج

## 🏗️ البنية الأساسية

```
النظام الموحد
├── unified_trading_learning_system.py  # النظام الأساسي
├── unified_prediction_server.py        # سيرفر التنبؤات
├── ForexMLBot_Advanced_V3_Unified.mq5  # إكسبيرت MT5
└── start_unified_system.py             # تشغيل النظام
```

## 🚀 البدء السريع

### 1. تثبيت المتطلبات
```bash
pip install pandas numpy scikit-learn lightgbm xgboost flask MetaTrader5
```

### 2. تشغيل النظام
```bash
python start_unified_system.py
```

### 3. إعداد MT5
1. انسخ `ForexMLBot_Advanced_V3_Unified.mq5` إلى مجلد Experts
2. أضف `http://localhost:5000` للـ URLs المسموحة
3. ألحق الإكسبيرت بالرسوم البيانية
4. فعّل التداول الآلي

## 📡 API Endpoints

### POST /predict
يستقبل 200 شمعة ويرجع إشارة تداول
```json
Request:
{
    "symbol": "EURUSDm",
    "timeframe": "M15",
    "candles": [
        {"time": "2024-01-01 12:00", "open": 1.1234, "high": 1.1240, ...}
    ]
}

Response:
{
    "action": "BUY",
    "confidence": 0.78,
    "sl_price": 1.1200,
    "tp1_price": 1.1300,
    "tp2_price": 1.1350,
    "risk_reward_ratio": 2.0
}
```

### POST /trade_result
يستقبل نتيجة الصفقة للتعلم
```json
{
    "symbol": "EURUSDm",
    "result": "WIN",
    "entry_price": 1.1234,
    "exit_price": 1.1284,
    "pips": 50
}
```

## 🧠 الميزات المتقدمة

### 1. حساب الميزات التكيفي
- يختار أهم الميزات تلقائياً
- يتكيف مع ظروف السوق المتغيرة
- 200+ مؤشر تقني

### 2. التعلم المستمر
- يحلل كل صفقة مغلقة
- يكتشف الأنماط المربحة
- يحدث النماذج كل 30 دقيقة

### 3. حساب SL/TP الديناميكي
- يستخدم ATR للتقلبات
- يعدل حسب قوة الإشارة (ADX)
- يراعي الجلسة التداولية

## 📊 قواعد البيانات

### forex_ml.db (7.8M سجل)
- بيانات تاريخية لجميع الأزواج
- أطر زمنية متعددة

### unified_forex_system.db
- أهمية الميزات
- أداء النماذج
- الأنماط المربحة

### trading_performance.db
- سجل الصفقات
- نتائج التعلم
- إحصائيات الأداء

## 🔧 التخصيص

### تعديل معاملات المخاطرة
في `unified_prediction_server.py`:
```python
self.risk_params = {
    'default_sl_pips': 50,      # SL الافتراضي
    'default_tp_pips': 100,     # TP الافتراضي
    'risk_reward_ratio': 2.0,   # نسبة المخاطرة/المكافأة
    'use_dynamic_sl': True      # استخدام SL ديناميكي
}
```

### تعديل فترة إعادة التدريب
في `unified_trading_learning_system.py`:
```python
self.retrain_interval = 24  # ساعات
self.min_new_trades = 100   # أقل عدد صفقات لإعادة التدريب
```

## 📈 مراقبة الأداء

### سجلات النظام
- `unified_system.log` - سجل النظام الأساسي
- `unified_server.log` - سجل السيرفر
- `training_real_data.log` - سجل التدريب

### لوحة MT5
تعرض الإكسبيرت معلومات حية:
- عدد الصفقات
- معدل الفوز
- إجمالي النقاط
- الصفقات المفتوحة

## 🛠️ حل المشاكل

### لا توجد إشارات
1. تحقق من تشغيل السيرفر
2. راجع سجلات السيرفر
3. تأكد من وجود نماذج مدربة

### خطأ WebRequest
1. أضف URL للقائمة المسموحة في MT5
2. تحقق من جدار الحماية
3. جرب زيادة Timeout

### أداء ضعيف
1. النظام يعيد التدريب تلقائياً
2. راجع الأنماط المكتشفة
3. تحقق من جودة البيانات

## 🚦 تشغيل الإنتاج

### 1. استخدم سيرفر مخصص
```python
# في start_unified_system.py
UseRemoteServer = True
ServerURL = "https://your-server.com"
```

### 2. قاعدة بيانات خارجية
```python
# في unified_trading_learning_system.py
self.historical_db = 'postgresql://user:pass@host/db'
```

### 3. مراقبة متقدمة
- استخدم Prometheus للمقاييس
- Grafana للوحات المعلومات
- تنبيهات للأخطاء الحرجة

## 📝 ملاحظات مهمة

1. **الأمان**: لا تشارك بيانات السيرفر
2. **الأداء**: النظام يحتاج 4GB RAM على الأقل
3. **البيانات**: تأكد من جودة البيانات التاريخية
4. **التحديثات**: النماذج تتحدث تلقائياً

## 🤝 المساهمة
لتحسين النظام:
1. أضف مؤشرات جديدة
2. حسّن خوارزميات التعلم
3. طوّر استراتيجيات جديدة

---
✨ **نظام متكامل للتداول الذكي** ✨