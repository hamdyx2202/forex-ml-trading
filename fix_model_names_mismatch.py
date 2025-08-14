#!/usr/bin/env python3
"""
Fix Model Names Mismatch
إصلاح عدم تطابق أسماء النماذج
"""

import os
import shutil
from pathlib import Path
import glob

print("🔧 Fixing model names mismatch...")
print("=" * 60)

print("\n📊 المشكلة:")
print("الخادم يبحث عن: models/advanced/EURJPYm_PERIOD_H1.pkl")
print("الموجود فعلياً: models/advanced/EURJPYm_PERIOD_H1_ensemble_20250814_152901.pkl")

# الحل 1: إنشاء روابط رمزية
print("\n🔗 Solution 1: Creating symbolic links...")

models_dir = Path("models/advanced")
if models_dir.exists():
    # البحث عن جميع النماذج مع timestamps
    model_files = list(models_dir.glob("*_ensemble_*.pkl"))
    
    print(f"Found {len(model_files)} models with timestamps")
    
    for model_file in model_files:
        # استخراج اسم النموذج الأساسي
        filename = model_file.stem  # e.g., EURJPYm_PERIOD_H1_ensemble_20250814_152901
        
        if '_ensemble_' in filename:
            base_name = filename.split('_ensemble_')[0]  # EURJPYm_PERIOD_H1
            simple_name = base_name + '.pkl'
            simple_path = models_dir / simple_name
            
            # إنشاء رابط رمزي أو نسخة
            if not simple_path.exists():
                try:
                    # محاولة إنشاء رابط رمزي (يعمل على Linux)
                    os.symlink(model_file.name, simple_path)
                    print(f"✅ Created symlink: {simple_name} -> {model_file.name}")
                except:
                    # إذا فشل، انسخ الملف
                    shutil.copy2(model_file, simple_path)
                    print(f"✅ Copied: {model_file.name} -> {simple_name}")

# الحل 2: تحديث advanced_predictor_95.py لإظهار debugging
print("\n📝 Solution 2: Adding debug info to predictor...")

predictor_file = "src/advanced_predictor_95.py"
if os.path.exists(predictor_file):
    with open(predictor_file, 'r') as f:
        content = f.read()
    
    # إضافة debugging
    if 'def load_latest_models(self):' in content and 'print(f"Available models: {list(self.models.keys())}")' not in content:
        # إضافة سطر debugging
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            if 'print(f"✅ Loaded {loaded_count} advanced models")' in line:
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + 'print(f"Available models: {list(self.models.keys())}")')
        
        content = '\n'.join(new_lines)
        
        with open(predictor_file + '.backup', 'w') as f:
            f.write(content)
        
        with open(predictor_file, 'w') as f:
            f.write(content)
        
        print("✅ Added debug info to predictor")

# الحل 3: إنشاء wrapper محدث للخادم
print("\n🔧 Solution 3: Creating updated server wrapper...")

server_wrapper = '''#!/usr/bin/env python3
"""
Fixed Model Loading Server
خادم محدث لتحميل النماذج
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import glob
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# تحديث advanced_predictor لتحميل النماذج بأسماء مختلفة
from src.advanced_predictor_95 import AdvancedPredictor

# تعديل دالة load_latest_models
original_load = AdvancedPredictor.load_latest_models

def new_load_latest_models(self):
    """تحميل أحدث النماذج المتقدمة - محدث"""
    model_dir = Path("models/advanced")
    if not model_dir.exists():
        print("⚠️ No advanced models found")
        return
        
    # البحث عن جميع ملفات النماذج
    model_files = list(model_dir.glob("*.pkl"))
    if not model_files:
        print("⚠️ No models found")
        return
    
    print(f"\\n📁 Found {len(model_files)} model files")
    
    # معالجة كل ملف
    loaded_count = 0
    for model_file in model_files:
        try:
            filename = model_file.stem
            
            # تحديد المفتاح بناءً على اسم الملف
            if '_ensemble_' in filename:
                # نموذج مع timestamp
                key = filename.split('_ensemble_')[0]
            elif filename.count('_') >= 2:
                # نموذج بسيط (مثل EURJPYm_PERIOD_H1)
                key = filename
            else:
                # تخطي الملفات غير المعروفة
                continue
            
            print(f"🔍 Loading {filename} as key: {key}")
            
            # تحميل النموذج
            model_data = joblib.load(model_file)
            
            # إضافة النموذج بالمفتاح الأساسي
            self.models[key] = model_data['model']
            self.scalers[key] = model_data['scaler']
            self.metrics[key] = model_data.get('metrics', {})
            
            # إضافة نسخ بديلة للمفتاح للتوافق
            if '_PERIOD_' in key:
                # إضافة نسخة بدون PERIOD_
                alt_key = key.replace('_PERIOD_', '_')
                self.models[alt_key] = model_data['model']
                self.scalers[alt_key] = model_data['scaler']
                self.metrics[alt_key] = model_data.get('metrics', {})
                print(f"  ➕ Also added as: {alt_key}")
            
            loaded_count += 1
            print(f"  ✅ Loaded successfully")
                
        except Exception as e:
            print(f"  ❌ Error loading {model_file}: {e}")
    
    print(f"\\n✅ Loaded {loaded_count} models")
    print(f"📊 Available model keys: {list(self.models.keys())}")

# استبدال الدالة
AdvancedPredictor.load_latest_models = new_load_latest_models

# الآن استيراد وتشغيل الخادم
print("🚀 Starting server with fixed model loading...")
from src.mt5_bridge_server_advanced import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('server_fixed_models.py', 'w') as f:
    f.write(server_wrapper)

print("✅ Created server_fixed_models.py")

# الحل 4: فحص النماذج الموجودة
print("\n🔍 Checking existing models...")

if models_dir.exists():
    all_models = list(models_dir.glob("*.pkl"))
    print(f"\nAll .pkl files in models/advanced/:")
    for model in sorted(all_models):
        print(f"  • {model.name}")
    
    # عرض الأسماء المتوقعة
    print("\n📋 Expected model names by server:")
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 'AUDUSDm', 'USDCADm', 'NZDUSDm', 'EURJPYm']
    timeframes = ['PERIOD_M5', 'PERIOD_M15', 'PERIOD_H1', 'PERIOD_H4']
    
    for symbol in symbols:
        for timeframe in timeframes:
            expected_name = f"{symbol}_{timeframe}.pkl"
            expected_path = models_dir / expected_name
            if expected_path.exists():
                print(f"  ✅ {expected_name}")
            else:
                # البحث عن نموذج مع timestamp
                pattern = f"{symbol}_{timeframe}_*.pkl"
                matches = list(models_dir.glob(pattern))
                if matches:
                    print(f"  ⚠️ {expected_name} -> Found: {matches[0].name}")
                else:
                    print(f"  ❌ {expected_name} - Not found")

print("\n" + "="*60)
print("✅ Solutions created!")
print("\n🚀 Try one of these:")
print("1. python server_fixed_models.py  (Recommended)")
print("2. Restart server to use symlinks/copies")
print("3. Check debug output to see loaded model names")