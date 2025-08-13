#!/usr/bin/env python3
"""
Find and Copy Existing Models
البحث عن النماذج الموجودة ونسخها
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

print("🔍 Searching for existing models...")
print("="*60)

# 1. البحث عن ملفات النماذج
found_models = []
model_locations = {}

# البحث في جميع المجلدات
for root, dirs, files in os.walk('/', followlinks=True):
    # تجنب المجلدات غير المرغوبة
    if any(skip in root for skip in ['proc', 'sys', 'dev', 'tmp', '.git', '__pycache__']):
        continue
        
    try:
        for file in files:
            if file.endswith('.pkl') and any(pattern in file for pattern in ['ensemble', 'model', 'PERIOD']):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                
                # النماذج الحقيقية عادة > 1 MB
                if size > 0.5:
                    found_models.append((full_path, size))
                    
                    # تجميع حسب المجلد
                    parent_dir = os.path.dirname(full_path)
                    if parent_dir not in model_locations:
                        model_locations[parent_dir] = []
                    model_locations[parent_dir].append(file)
                    
    except PermissionError:
        continue
    except Exception as e:
        continue

# 2. عرض النتائج
if found_models:
    print(f"\n✅ Found {len(found_models)} model files:")
    
    # عرض حسب المجلد
    for location, files in model_locations.items():
        print(f"\n📁 {location}")
        print(f"   Files: {len(files)}")
        for f in files[:5]:  # أول 5 ملفات
            print(f"   • {f}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
    
    # اختيار أفضل مجلد (الذي يحتوي على أكثر نماذج)
    best_location = max(model_locations.items(), key=lambda x: len(x[1]))[0]
    print(f"\n🎯 Best location: {best_location}")
    print(f"   Contains {len(model_locations[best_location])} models")
    
    # 3. نسخ النماذج
    print("\n📋 Copying models to standard location...")
    
    # إنشاء المجلدات
    os.makedirs('models/advanced', exist_ok=True)
    os.makedirs('models/unified', exist_ok=True)
    
    copied_count = 0
    for model_file in model_locations[best_location]:
        src_path = os.path.join(best_location, model_file)
        
        # تحديد المجلد الهدف
        if 'unified' in src_path or 'v2' in model_file:
            dst_dir = 'models/unified'
        else:
            dst_dir = 'models/advanced'
            
        dst_path = os.path.join(dst_dir, model_file)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"   ✅ Copied: {model_file}")
        except Exception as e:
            print(f"   ❌ Failed to copy {model_file}: {e}")
    
    print(f"\n✅ Copied {copied_count} models successfully!")
    
    # 4. إنشاء symlink كبديل
    if copied_count == 0:
        print("\n🔗 Creating symbolic link instead...")
        try:
            # إزالة المجلد إذا كان موجود
            if os.path.exists('models/advanced') and not os.listdir('models/advanced'):
                os.rmdir('models/advanced')
                
            # إنشاء symlink
            os.symlink(best_location, 'models/advanced')
            print(f"✅ Created symlink: models/advanced → {best_location}")
        except Exception as e:
            print(f"❌ Failed to create symlink: {e}")
    
else:
    print("\n❌ No model files found!")
    print("\n📝 Searching for model locations in common paths...")
    
    # البحث في مسارات محددة
    common_paths = [
        '/home/*/models',
        '/root/models',
        '/var/*/models',
        '/opt/*/models',
        '../models',
        '../../models'
    ]
    
    for path_pattern in common_paths:
        from glob import glob
        matches = glob(path_pattern, recursive=True)
        if matches:
            print(f"Found: {matches}")

# 5. التحقق النهائي
print("\n📊 Final check...")

for model_dir in ['models/advanced', 'models/unified']:
    if os.path.exists(model_dir):
        pkl_files = list(Path(model_dir).glob('*.pkl'))
        if pkl_files:
            print(f"\n✅ {model_dir}: {len(pkl_files)} models")
            for f in pkl_files[:3]:
                size = os.path.getsize(f) / (1024 * 1024)
                print(f"   • {f.name} ({size:.1f} MB)")

# 6. تحديث مسار النماذج في predictor
print("\n🔧 Creating model path configuration...")

config = f'''# Model Path Configuration
# تكوين مسار النماذج

import os

# المسار الأساسي للنماذج
MODEL_BASE_PATH = "{best_location if found_models else 'models/advanced'}"

# مسارات بديلة
ALTERNATIVE_PATHS = [
    "models/advanced",
    "models/unified",
    "../models/advanced",
    "{best_location if found_models else ''}"
]

print(f"Using model path: {{MODEL_BASE_PATH}}")
'''

with open('model_config.py', 'w') as f:
    f.write(config)

print("\n✅ Created model_config.py")

if found_models:
    print("\n🎉 Models are ready!")
    print("\n🚀 Now restart the server:")
    print("   python src/mt5_bridge_server_advanced.py")
else:
    print("\n⚠️ No models found. You need to train them:")
    print("   python quick_train_models.py")