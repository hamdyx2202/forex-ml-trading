# دليل النظام الموحد - Unified System Guide

## 🎯 الهدف
توحيد جميع أنظمة التعلم والتدريب لتجنب مشاكل عدم التوافق

## 📊 المعايير الموحدة

### 1. عدد الميزات الثابت
```python
STANDARD_FEATURES = 70  # ثابت في جميع الأنظمة
```

### 2. أسماء النماذج الموحدة
```python
# التنسيق القياسي (بدون timestamps)
MODEL_NAME = "{symbol}_{timeframe}.pkl"

# أمثلة:
# ✅ EURUSDm_PERIOD_M5.pkl
# ✅ GBPUSDm_PERIOD_H1.pkl
# ❌ EURUSDm_PERIOD_M5_ensemble_20250814_152901.pkl
```

### 3. قائمة الميزات القياسية (70 ميزة)
- **Price Features (10)**: returns, log_returns, ratios...
- **Technical Indicators (40)**: RSI, SMA, EMA, MACD...
- **Pattern Features (10)**: trend, support/resistance...
- **Padding Features (10)**: padding_0 to padding_9

## 🚀 الأنظمة المحدثة

### 1. Advanced Learner الموحد
```bash
# للتشغيل على VPS
screen -S advanced_unified
cd /home/forex-ml-trading
source venv_pro/bin/activate
python src/advanced_learner_unified.py
# Ctrl+A ثم D للخروج
```

**الميزات:**
- ✅ ينتج 70 ميزة دائماً
- ✅ يحفظ بأسماء موحدة
- ✅ يتحقق من الأداء كل ساعة
- ✅ يحدث النماذج عند انخفاض الأداء

### 2. Continuous Learner الموحد
```bash
# للتشغيل على VPS
screen -S continuous_unified
cd /home/forex-ml-trading
source venv_pro/bin/activate
python src/continuous_learner_unified.py
# Ctrl+A ثم D للخروج
```

**الميزات:**
- ✅ يتعلم من نتائج التداول الفعلية
- ✅ ينتج 70 ميزة دائماً
- ✅ يحفظ بأسماء موحدة
- ✅ يحسن النماذج بناءً على الأداء الفعلي

### 3. Model Validator
```bash
# للتحقق من النماذج
python model_validator.py
```

**الوظائف:**
- ✅ يتحقق من صحة جميع النماذج
- ✅ يصلح النماذج غير المتوافقة
- ✅ ينشئ تقرير validation
- ✅ يوفر compatibility wrapper

## 📋 خطوات التطبيق على VPS

### 1. نسخ الملفات الجديدة
```bash
# من جهازك المحلي
scp unified_standards.py root@69.62.121.53:/home/forex-ml-trading/
scp src/advanced_learner_unified.py root@69.62.121.53:/home/forex-ml-trading/src/
scp src/continuous_learner_unified.py root@69.62.121.53:/home/forex-ml-trading/src/
scp model_validator.py root@69.62.121.53:/home/forex-ml-trading/
```

### 2. التحقق من النماذج الحالية
```bash
ssh root@69.62.121.53
cd /home/forex-ml-trading
source venv_pro/bin/activate

# تشغيل المُحقق
python model_validator.py
```

### 3. تشغيل أنظمة التعلم الموحدة
```bash
# Advanced Learner
screen -S advanced_unified
python src/advanced_learner_unified.py

# Continuous Learner (في screen منفصل)
screen -S continuous_unified
python src/continuous_learner_unified.py
```

### 4. مراقبة النظام
```bash
# عرض السجلات
tail -f logs/server.log

# عرض حالة screens
screen -ls

# العودة إلى screen
screen -r advanced_unified
screen -r continuous_unified
```

## ⚠️ تحذيرات مهمة

### 1. لا تشغل النظامين القديم والجديد معاً
```bash
# أوقف الأنظمة القديمة أولاً
pkill -f "advanced_learner_simple.py"
pkill -f "continuous_learner_simple.py"
```

### 2. النسخ الاحتياطي
```bash
# قبل تشغيل الأنظمة الجديدة
cp -r models models_backup_$(date +%Y%m%d)
```

### 3. التحقق من التوافق
```bash
# بعد أي تحديث
python model_validator.py
```

## 📊 المجلدات والملفات

### هيكل المجلدات الموحد
```
/home/forex-ml-trading/
├── models/
│   ├── unified/          # النماذج الموحدة
│   │   ├── EURUSDm_PERIOD_M5.pkl
│   │   ├── GBPUSDm_PERIOD_H1.pkl
│   │   └── validation_report.json
│   ├── backup/           # النسخ الاحتياطية
│   └── advanced/         # النماذج القديمة
├── src/
│   ├── advanced_learner_unified.py
│   ├── continuous_learner_unified.py
│   └── mt5_bridge_server_advanced.py
├── unified_standards.py
└── model_validator.py
```

## 🔍 التحقق من عمل النظام

### 1. التحقق من أنظمة التعلم
```bash
# عرض العمليات
ps aux | grep -E "(advanced|continuous)_learner"

# عرض السجلات
tail -f models/unified/performance_log.json
tail -f models/unified/continuous_learning_log.json
```

### 2. التحقق من النماذج
```bash
# عدد النماذج
ls -la models/unified/*.pkl | wc -l

# أحدث تحديث
ls -lt models/unified/*.pkl | head -5
```

### 3. التحقق من الأداء
```bash
# تقرير validation
cat models/unified/validation_report.json | jq '.'
```

## 🛠️ حل المشاكل الشائعة

### مشكلة: "No module named 'unified_standards'"
```bash
# تأكد من وجود الملف
ls -la unified_standards.py

# أضف المسار
export PYTHONPATH=/home/forex-ml-trading:$PYTHONPATH
```

### مشكلة: "Model validation failed"
```bash
# استخدم المُصلح
python model_validator.py

# أو استخدم compatibility wrapper
python compatibility_wrapper.py
```

### مشكلة: "Memory error"
```bash
# قلل عدد النماذج المعالجة
# عدل في advanced_learner_unified.py
# قلل symbols أو timeframes
```

## 📈 مؤشرات النجاح

✅ **النظام يعمل بشكل صحيح إذا:**
- جميع النماذج تجتاز validation
- لا توجد أخطاء feature mismatch
- أنظمة التعلم تعمل بدون أخطاء
- النماذج تُحدث تلقائياً
- الأداء يتحسن مع الوقت

## 🚨 متى تتدخل يدوياً؟

- إذا انخفضت دقة النماذج بشكل كبير
- إذا توقفت أنظمة التعلم عن العمل
- إذا ظهرت أخطاء feature mismatch
- إذا امتلأت المساحة التخزينية

## 📞 الدعم

في حالة وجود مشاكل:
1. راجع validation_report.json
2. تحقق من سجلات الأخطاء
3. شغل model_validator.py
4. استخدم compatibility wrapper كحل مؤقت

---

**ملاحظة**: هذا النظام الموحد يضمن استقرار طويل المدى ويمنع مشاكل عدم التوافق المستقبلية.