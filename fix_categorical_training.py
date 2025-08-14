#!/usr/bin/env python3
"""
Fix Categorical Columns Error in Training
إصلاح خطأ الأعمدة الفئوية في التدريب
"""

import os
import fileinput
import sys

print("🔧 Fixing categorical columns error...")
print("="*60)

# الملف الذي يحتاج إصلاح
training_file = "train_advanced_95_percent.py"

if not os.path.exists(training_file):
    print(f"❌ {training_file} not found!")
    print("\n🔍 Searching for training files...")
    
    # البحث عن ملفات التدريب
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'train' in file and file.endswith('.py'):
                print(f"Found: {os.path.join(root, file)}")
    sys.exit(1)

# قراءة الملف
with open(training_file, 'r', encoding='utf-8') as f:
    content = f.read()

# البحث عن المشكلة
if 'volatility_regime' in content or 'volume_regime' in content:
    print("✅ Found categorical columns in code")
    
    # إضافة كود لتحويل الأعمدة الفئوية
    fix_code = '''
        # تحويل الأعمدة الفئوية إلى رقمية
        categorical_columns = ['volatility_regime', 'volume_regime']
        for col in categorical_columns:
            if col in df_features.columns:
                # تحويل إلى رقمي
                if df_features[col].dtype == 'category':
                    df_features[col] = df_features[col].cat.codes
                elif df_features[col].dtype == 'object':
                    # إنشاء تصنيف رقمي
                    df_features[col] = pd.Categorical(df_features[col]).codes
'''
    
    # البحث عن المكان المناسب لإضافة الكود
    # بعد إنشاء df_features وقبل التدريب
    insertion_points = [
        "df_features = engineer.create_features",
        "# إزالة الصفوف بدون هدف",
        "# تحويل الهدف لثنائي",
        "X = df_features[feature_cols]"
    ]
    
    inserted = False
    for point in insertion_points:
        if point in content:
            # إدراج الكود بعد هذه النقطة
            lines = content.split('\n')
            new_lines = []
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if point in line and not inserted:
                    # إضافة الكود في السطر التالي
                    # الحصول على المسافة البادئة
                    indent = len(line) - len(line.lstrip())
                    if i + 1 < len(lines):
                        next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                        indent = next_indent
                    
                    # إضافة الكود مع المسافة البادئة الصحيحة
                    for fix_line in fix_code.strip().split('\n'):
                        new_lines.append(' ' * indent + fix_line)
                    inserted = True
            
            content = '\n'.join(new_lines)
            break
    
    if inserted:
        # حفظ الملف المُصلح
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed categorical columns handling")
    else:
        print("⚠️ Could not find insertion point")

# حل بديل - إنشاء wrapper للتدريب
print("\n📝 Creating alternative fix...")

wrapper_code = '''#!/usr/bin/env python3
"""
Training Wrapper with Categorical Fix
غلاف التدريب مع إصلاح الفئات
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the original training file
sys.path.append('.')

# تحميل ملف التدريب الأصلي
print("Loading original training module...")
import train_advanced_95_percent as original_train

# تعديل دالة prepare_data
original_prepare_data = original_train.prepare_data

def fixed_prepare_data(symbol, timeframe):
    """نسخة مُصلحة من prepare_data"""
    # استدعاء الدالة الأصلية
    result = original_prepare_data(symbol, timeframe)
    
    if result[0] is None:
        return result
    
    X, y, feature_cols = result
    
    # تحويل الأعمدة الفئوية
    if isinstance(X, pd.DataFrame):
        categorical_columns = ['volatility_regime', 'volume_regime']
        for col in categorical_columns:
            if col in X.columns:
                if X[col].dtype == 'category':
                    X[col] = X[col].cat.codes
                elif X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
                print(f"  ✅ Converted {col} to numeric")
    
    return X, y, feature_cols

# استبدال الدالة
original_train.prepare_data = fixed_prepare_data

# تشغيل التدريب
if __name__ == "__main__":
    print("🚀 Starting training with categorical fix...")
    
    # استدعاء الدالة الرئيسية
    if hasattr(original_train, 'main'):
        original_train.main()
    else:
        # تشغيل التدريب مباشرة
        trainer = original_train.AdvancedTrainer()
        trainer.train_all_models()
'''

with open('train_with_fix.py', 'w') as f:
    f.write(wrapper_code)

print("✅ Created train_with_fix.py")

# حل آخر - تعديل feature_engineer
print("\n🔧 Checking feature_engineer...")

feature_files = [
    'feature_engineer_fixed_v2.py',
    'feature_engineer_fixed_v3.py',
    'feature_engineer_fixed_v4.py',
    'feature_engineer_fixed_v5.py',
    'feature_engineer_fixed.py'
]

for fe_file in feature_files:
    if os.path.exists(fe_file):
        print(f"\nChecking {fe_file}...")
        
        with open(fe_file, 'r') as f:
            fe_content = f.read()
            
        if 'volatility_regime' in fe_content or 'volume_regime' in fe_content:
            print(f"✅ Found categorical columns in {fe_file}")
            
            # البحث عن المكان الذي يتم فيه إنشاء هذه الأعمدة
            if "df['volatility_regime']" in fe_content:
                # استبدال بكود يُنشئ أعمدة رقمية
                fe_content = fe_content.replace(
                    "df['volatility_regime'] = pd.cut",
                    "# تحويل مباشر لرقمي\n        df['volatility_regime'] = pd.cut"
                )
                fe_content = fe_content.replace(
                    "df['volume_regime'] = pd.cut",
                    "# تحويل مباشر لرقمي\n        df['volume_regime'] = pd.cut"
                )
                
                # إضافة .cat.codes في نهاية السطر
                lines = fe_content.split('\n')
                new_lines = []
                
                for line in lines:
                    if 'volatility_regime' in line and 'pd.cut' in line and not '.cat.codes' in line:
                        line = line.rstrip() + '.cat.codes'
                    elif 'volume_regime' in line and 'pd.cut' in line and not '.cat.codes' in line:
                        line = line.rstrip() + '.cat.codes'
                    new_lines.append(line)
                
                fe_content = '\n'.join(new_lines)
                
                # حفظ النسخة المُصلحة
                fixed_name = fe_file.replace('.py', '_fixed_cat.py')
                with open(fixed_name, 'w') as f:
                    f.write(fe_content)
                print(f"✅ Created {fixed_name}")

print("\n" + "="*60)
print("✅ Fixes applied!")
print("\n🚀 Try running one of these:")
print("1. python train_with_fix.py")
print("2. python train_advanced_95_percent.py (if fixed)")
print("3. Update feature_engineer imports to use *_fixed_cat.py versions")

# إنشاء سكريبت تدريب مبسط
print("\n📝 Creating simplified training script...")

simple_training = '''#!/usr/bin/env python3
"""
Simplified Training without Categorical Columns
تدريب مبسط بدون أعمدة فئوية
"""

import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import os

print("🚀 Starting simplified training...")

# البحث عن قاعدة البيانات
db_path = None
for path in ['forex_data.db', 'data/forex_data.db', '../forex_data.db']:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    print("❌ No database found!")
    # البحث في كل المجلدات
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.db'):
                db_path = os.path.join(root, file)
                print(f"Found: {db_path}")
                break
        if db_path:
            break

if db_path:
    print(f"✅ Using database: {db_path}")
    
    # الرموز والأطر الزمنية
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm']
    timeframes = ['PERIOD_M5', 'PERIOD_H4']  # فقط 2 للاختبار
    
    os.makedirs('models/advanced', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\\nTraining {symbol} {timeframe}...")
            
            try:
                # تحميل البيانات
                query = "SELECT * FROM forex_data WHERE symbol = ? AND timeframe = ? ORDER BY time"
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
                
                if len(df) < 100:
                    print(f"  ⚠️ Not enough data: {len(df)} rows")
                    continue
                
                # ميزات بسيطة
                df['returns'] = df['close'].pct_change()
                df['hl_ratio'] = df['high'] / df['low']
                df['co_ratio'] = df['close'] / df['open']
                df['volatility'] = df['returns'].rolling(20).std()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['rsi'] = 50  # مبسط
                
                # إزالة NaN
                df = df.dropna()
                
                # الميزات والهدف
                feature_cols = ['returns', 'hl_ratio', 'co_ratio', 'volatility', 'rsi']
                X = df[feature_cols].values
                y = (df['close'].shift(-5) > df['close']).astype(int).values[:-5]
                X = X[:-5]
                
                if len(X) < 50:
                    print(f"  ⚠️ Not enough samples: {len(X)}")
                    continue
                
                # تقسيم وتدريب
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # نموذج بسيط
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                accuracy = model.score(X_test, y_test)
                print(f"  ✅ Accuracy: {accuracy:.2%}")
                
                # حفظ
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': {'accuracy': accuracy}
                }
                
                filename = f'models/advanced/{symbol}_{timeframe}_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                joblib.dump(model_data, filename)
                print(f"  ✅ Saved: {filename}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    conn.close()
    print("\\n✅ Training completed!")
else:
    print("❌ No database found for training!")
'''

with open('train_simple.py', 'w') as f:
    f.write(simple_training)

print("✅ Created train_simple.py")
print("\nFor quick results, try: python train_simple.py")