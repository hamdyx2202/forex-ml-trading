# دليل حل مشكلة قاعدة البيانات
# Database Issue Fix Guide

## المشكلة
```
Error: no such table: forex_data
FileNotFoundError: models/unified/training_summary.json
```

## الحلول المتاحة

### الحل 1: استخدام البحث التلقائي ✅ (مُوصى به)
```bash
# هذا الملف يبحث عن قاعدة البيانات تلقائياً
python retrain_with_auto_db.py
```

المميزات:
- يبحث في جميع المجلدات
- يتحقق من صحة الجداول
- يكتشف أسماء الأعمدة تلقائياً

### الحل 2: إصلاح الإعداد يدوياً
```bash
# 1. إنشاء المجلدات وإيجاد قاعدة البيانات
python fix_training_setup.py

# 2. تشغيل النسخة المُصلحة
python retrain_with_unified_features_fixed.py
```

### الحل 3: إنشاء قاعدة بيانات اختبار
إذا لم توجد قاعدة بيانات:
```bash
python fix_training_setup.py
# سينشئ forex_test_data.db تلقائياً
```

## البحث عن قاعدة البيانات يدوياً

### 1. البحث عن ملفات .db
```bash
find . -name "*.db" -type f 2>/dev/null
```

### 2. فحص محتوى قاعدة البيانات
```bash
sqlite3 your_database.db
.tables
.schema your_table_name
```

### 3. التحقق من البيانات
```sql
SELECT COUNT(*) FROM your_table;
SELECT DISTINCT symbol FROM your_table LIMIT 10;
SELECT DISTINCT timeframe FROM your_table LIMIT 10;
```

## أسماء قواعد البيانات المحتملة
- forex_data.db
- forex_ml_data.db
- mt5_data.db
- trading_data.db
- data/forex_data.db
- ../forex_data.db

## أسماء الجداول المحتملة
- forex_data
- mt5_data
- market_data
- ohlc_data
- candles

## الخطوات الموصى بها

### 1. جرب البحث التلقائي أولاً
```bash
python retrain_with_auto_db.py
```

### 2. إذا فشل، ابحث يدوياً
```bash
# ابحث عن أي ملف .db
find / -name "*.db" 2>/dev/null | grep -E "(forex|trading|mt5|data)"

# أو في المجلد الحالي فقط
find . -name "*.db" -ls
```

### 3. عندما تجد القاعدة، حدث المسار
في `retrain_with_unified_features.py`:
```python
def load_data(self, symbol: str, timeframe: str, db_path: str = 'YOUR_DB_PATH'):
```

## ملاحظات مهمة

### تنسيق البيانات المطلوب
الجدول يجب أن يحتوي على:
- `time` (Unix timestamp)
- `open`, `high`, `low`, `close` (أسعار)
- `volume` (اختياري)
- `symbol` (مثل: EURUSDm)
- `timeframe` (مثل: PERIOD_M5)

### إذا كانت أسماء الأعمدة مختلفة
عدّل الاستعلام في `load_data()`:
```python
query = f"""
    SELECT 
        timestamp as time,  # إذا كان الاسم مختلف
        o as open,          # أو open_price
        h as high,          # أو high_price
        l as low,           # أو low_price
        c as close,         # أو close_price
        v as volume         # أو vol
    FROM {table_name}
    WHERE symbol = ? AND timeframe = ?
"""
```

## للطوارئ: إنشاء قاعدة بيانات من CSV
إذا كانت البيانات في ملفات CSV:
```python
import pandas as pd
import sqlite3

# قراءة CSV
df = pd.read_csv('your_data.csv')

# إنشاء قاعدة بيانات
conn = sqlite3.connect('forex_data.db')
df.to_sql('forex_data', conn, if_exists='replace', index=False)
conn.close()
```

الحل الأسرع: استخدم `python retrain_with_auto_db.py` 🚀