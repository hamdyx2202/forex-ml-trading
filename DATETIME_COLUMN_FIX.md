# إصلاح خطأ "['datetime'] not in index"
# Fix for "['datetime'] not in index" Error

## المشكلة / The Problem
```
KeyError: "['datetime'] not in index"
```

السبب: AdaptiveFeatureEngineer كان يحاول الوصول لعمود `datetime` غير موجود

## بنية البيانات من EA / Data Structure from EA
```python
# EA يرسل:
{
    'time': 1732000000,  # Unix timestamp
    'open': 1.0500,
    'high': 1.0520,
    'low': 1.0480,
    'close': 1.0510,
    'volume': 1000
}

# الخادم يحول إلى:
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df.set_index('datetime', inplace=True)
# الآن datetime في الـ index، ليس عمود!
```

## الإصلاحات المطبقة / Applied Fixes

### 1. في `create_features()`:
```python
# قبل ❌
essential_cols = ['open', 'high', 'low', 'close', 'volume', 'time', 'datetime']
df_essential = df[essential_cols].copy()

# بعد ✅
essential_cols = ['open', 'high', 'low', 'close', 'volume', 'time']
available_cols = [col for col in essential_cols if col in df.columns]
df_essential = df[available_cols].copy()
```

### 2. في اختيار الميزات:
```python
# قبل ❌
keep_cols = essential_cols + selected_features

# بعد ✅
keep_cols = [col for col in essential_cols if col in df.columns] + selected_features
keep_cols = [col for col in keep_cols if col in df.columns]
```

### 3. في `prepare_for_prediction()`:
```python
# بعد ✅
# إزالة أي أعمدة غير موجودة
non_feature_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'datetime']
df_features = df_features.drop(columns=[col for col in non_feature_cols if col in df_features.columns])
```

## النتيجة / Result
- ✅ لا مزيد من أخطاء datetime
- ✅ يعمل مع البيانات كما يرسلها EA
- ✅ يحتفظ بـ 68 ميزة كما متوقع

## للتحقق / To Verify
```bash
# اختبار الإصلاح
python3 test_adaptive_fix.py

# تشغيل الخادم
python3 src/mt5_bridge_server_advanced.py
```

## ملاحظة مهمة / Important Note
الـ datetime موجود في df.index، ليس df.columns
استخدم df.index للوصول للتاريخ والوقت