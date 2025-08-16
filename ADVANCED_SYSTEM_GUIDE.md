# 🚀 دليل النظام المتقدم الكامل - دقة 95%+

## 📌 نظرة عامة
هذا هو النظام المتقدم الكامل للتداول الآلي باستخدام 6 نماذج AI و 217+ مؤشر فني.

## 🏗️ بنية النظام

### 1️⃣ نظام التدريب المتقدم
```
train_advanced_complete.py
├── 6 نماذج AI (Random Forest, XGBoost, Neural Network, SVM, LightGBM, Ensemble)
├── 217+ مؤشر فني متقدم
├── تحسين تلقائي للمعاملات
└── نظام تقييم متعدد المستويات
```

### 2️⃣ نظام التدريب المستمر
```
continuous_training_system.py
├── مراقبة 24/7 للأداء
├── إعادة تدريب تلقائي عند انخفاض الأداء
├── تحديث النماذج عند توفر بيانات جديدة
└── تقارير أداء دورية
```

### 3️⃣ خادم التنبؤات المتقدم
```
mt5_prediction_server.py
├── خادم HTTP على المنفذ 5000
├── تنبؤات متعددة النماذج
├── حساب SL/TP ديناميكي
└── تحليل الثقة والمخاطر
```

### 4️⃣ نظام التداول الرئيسي
```
main.py + src/trader.py
├── اتصال مباشر مع MT5
├── إدارة مخاطر متقدمة
├── تنفيذ الصفقات تلقائياً
└── مراقبة الصفقات المفتوحة
```

## 🚀 طرق التشغيل

### الطريقة السهلة (موصى بها):
```bash
python RUN_ADVANCED_SYSTEM.py
```
ثم اختر من القائمة:
- 1 = التدريب الكامل (مرة واحدة)
- 2 = التدريب المستمر (24/7)
- 3 = التداول الآلي
- 4 = النظام الكامل

### الطريقة اليدوية (للمتقدمين):

#### أ. التدريب الكامل (مرة واحدة):
```bash
# تدريب جميع الأزواج بالنظام المتقدم
python train_advanced_complete.py

 📊 كيفية الاستخدام:

  # تدريب شامل بالمعالجة المتوازية (افتراضي)
  python train_advanced_complete_parallel.py

  # تحديد عدد العمليات المتوازية
  python train_advanced_complete_parallel.py --workers 4

  # اختبار سريع لعملة واحدة
  python train_advanced_complete_parallel.py --quick
# أو لتدريب زوج واحد
python train_advanced_complete.py --symbol EURUSD --timeframe H1
```

#### ب. التدريب المستمر:
```bash
# تشغيل نظام التدريب المستمر
python continuous_training_system.py
```

#### ج. التداول الآلي:
```bash
# 1. شغل خادم التنبؤات أولاً
python mt5_prediction_server.py

# 2. في نافذة جديدة، شغل نظام التداول
python main.py
```

#### د. لوحة التحكم:
```bash
streamlit run dashboard.py
```

## 📋 الأوامر المتقدمة

### تدريب أزواج محددة:
```bash
# تدريب EURUSD فقط
python train_advanced_complete.py --pairs EURUSD

# تدريب عدة أزواج
python train_advanced_complete.py --pairs EURUSD,GBPUSD,XAUUSD

# تدريب بإطار زمني محدد
python train_advanced_complete.py --timeframes H1,H4
```

### استيراد بيانات MT5:
```bash
# استيراد بيانات جديدة
python import_mt5_data.py --symbol EURUSD --timeframe H1 --days 365
```

### فحص الأداء:
```bash
# عرض أداء النماذج
python performance_tracker.py --show-report
```

## 🔧 ملفات الإعداد

### 1. المعايير الموحدة (`unified_standards.py`):
```python
UNIFIED_STANDARDS = {
    'MIN_ACCURACY': 0.80,  # الحد الأدنى للدقة
    'CONFIDENCE_THRESHOLD': 0.70,  # عتبة الثقة للتداول
    'MAX_RISK_PER_TRADE': 0.02,  # 2% مخاطرة
    'MIN_DATA_POINTS': 10000,  # الحد الأدنى للبيانات
}
```

### 2. إعدادات التداول:
```python
# في src/config.py
TRADING_CONFIG = {
    'lot_size': 0.01,
    'max_positions': 5,
    'use_trailing_stop': True,
    'news_filter': True,
}
```

## 📊 مؤشرات النظام

### المؤشرات الأساسية (75):
- Moving Averages (SMA, EMA, WMA)
- RSI, MACD, Stochastic
- Bollinger Bands, ATR
- Volume indicators

### المؤشرات المتقدمة (142+):
- Harmonic patterns
- Market microstructure
- Order flow analysis
- Sentiment indicators
- Multi-timeframe analysis
- Machine learning features

## 🛡️ إدارة المخاطر

### 1. حساب حجم الصفقة:
- يعتمد على رصيد الحساب
- الحد الأقصى للمخاطرة 2%
- يتكيف مع volatility

### 2. وقف الخسارة الديناميكي:
- يعتمد على ATR
- يتكيف مع ظروف السوق
- حماية من الفجوات السعرية

### 3. إدارة المحفظة:
- توزيع المخاطر
- حد أقصى للصفقات المفتوحة
- correlation analysis

## 🔍 تتبع الأداء

### التقارير المتاحة:
1. **تقرير الأداء اليومي**: في `results/daily_report.html`
2. **سجل الصفقات**: في `data/forex_ml.db` جدول `trades`
3. **أداء النماذج**: في `results/advanced/`
4. **لوحة التحكم**: http://localhost:8501

### قاعدة البيانات:
```sql
-- عرض آخر الصفقات
SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;

-- عرض أداء النماذج
SELECT * FROM models ORDER BY accuracy DESC;

-- إحصائيات عامة
SELECT symbol, COUNT(*) as trades, AVG(profit) as avg_profit 
FROM trades GROUP BY symbol;
```

## ⚠️ تحذيرات مهمة

1. **ابدأ بحساب تجريبي** حتى تتأكد من عمل النظام
2. **راقب النظام** في الأيام الأولى
3. **الأسواق المتقلبة** قد تؤثر على الأداء
4. **تحديث البيانات** ضروري للحفاظ على الدقة

## 🆘 حل المشاكل الشائعة

### 1. "ModuleNotFoundError":
```bash
pip install -r requirements.txt
```

### 2. "MT5 connection failed":
- تأكد من فتح MetaTrader 5
- تأكد من تفعيل Algo Trading
- راجع إعدادات الحساب

### 3. "No models found":
- شغل التدريب أولاً
- تأكد من وجود مجلد `models/`

### 4. "Low accuracy":
- زد كمية البيانات التدريبية
- جرب أطر زمنية مختلفة
- راجع جودة البيانات

## 📈 نصائح لتحسين الأداء

1. **البيانات**: كلما زادت البيانات، زادت الدقة
2. **التنويع**: درب على أزواج وأطر زمنية متعددة
3. **المراقبة**: استخدم لوحة التحكم لمتابعة الأداء
4. **التحديث**: أعد التدريب شهرياً على الأقل
5. **الاختبار**: اختبر على بيانات تاريخية أولاً

## 🎯 الخطوات الموصى بها

1. **البداية**:
   ```bash
   python RUN_ADVANCED_SYSTEM.py
   # اختر 1 للتدريب الأول
   ```

2. **الاختبار**:
   ```bash
   # بعد التدريب، اختبر على حساب تجريبي
   python RUN_ADVANCED_SYSTEM.py
   # اختر 3 للتداول
   ```

3. **الإنتاج**:
   ```bash
   # بعد التأكد من الأداء
   python RUN_ADVANCED_SYSTEM.py
   # اختر 4 للنظام الكامل
   ```

---

**تذكر**: هذا نظام متقدم يتطلب فهماً جيداً. ابدأ بحذر وراقب الأداء دائماً! 🚀