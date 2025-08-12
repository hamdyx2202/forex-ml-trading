#!/usr/bin/env python3
"""
Fix XGBoost categorical error
إصلاح خطأ XGBoost مع البيانات الفئوية
"""

import os

# قراءة الملف الأصلي
with open('feature_engineer_fixed_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# استبدال الأسطر المشكلة
old_code = """        # Market regime
        if 'ATR_14' in df.columns:
            try:
                df['volatility_regime'] = pd.qcut(df['ATR_14'].dropna(), q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                df['volatility_regime'] = df['volatility_regime'].map({'low': 0, 'medium': 1, 'high': 2})
            except:
                df['volatility_regime'] = 0
                
        if 'volume' in df.columns:
            try:
                df['volume_regime'] = pd.qcut(df['volume'].dropna(), q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                df['volume_regime'] = df['volume_regime'].map({'low': 0, 'medium': 1, 'high': 2})
            except:
                df['volume_regime'] = 0"""

new_code = """        # Market regime
        if 'ATR_14' in df.columns:
            try:
                # استخدام qcut بدون labels لتجنب categorical
                df['volatility_regime'] = pd.qcut(df['ATR_14'].dropna(), q=3, labels=False, duplicates='drop')
                df['volatility_regime'] = df['volatility_regime'].fillna(1).astype(int)
            except:
                df['volatility_regime'] = 1
                
        if 'volume' in df.columns:
            try:
                # استخدام qcut بدون labels لتجنب categorical
                df['volume_regime'] = pd.qcut(df['volume'].dropna(), q=3, labels=False, duplicates='drop')
                df['volume_regime'] = df['volume_regime'].fillna(1).astype(int)
            except:
                df['volume_regime'] = 1"""

# استبدال الكود
content = content.replace(old_code, new_code)

# حفظ النسخة المحدثة
with open('feature_engineer_fixed_v3.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Created feature_engineer_fixed_v3.py with XGBoost fix")

# تحديث train_advanced_95_percent.py
with open('train_advanced_95_percent.py', 'r', encoding='utf-8') as f:
    train_content = f.read()

train_content = train_content.replace(
    'from feature_engineer_fixed_v2 import FeatureEngineer',
    'from feature_engineer_fixed_v3 import FeatureEngineer'
)

with open('train_advanced_95_percent.py', 'w', encoding='utf-8') as f:
    f.write(train_content)

print("✅ Updated train_advanced_95_percent.py to use v3")
print("\n🚀 Now run: python train_advanced_95_percent.py")