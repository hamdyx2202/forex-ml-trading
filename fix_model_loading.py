#!/usr/bin/env python3
"""
Fix model loading in advanced_predictor_95.py
إصلاح تحميل النماذج
"""

import shutil

# نسخ احتياطية
shutil.copy('src/advanced_predictor_95.py', 'src/advanced_predictor_95_backup.py')

# قراءة الملف
with open('src/advanced_predictor_95.py', 'r', encoding='utf-8') as f:
    content = f.read()

# تحديث دالة load_latest_models
old_code = """                # استخراج اسم الزوج والإطار الزمني
                filename = model_file.stem
                parts = filename.split('_')
                if len(parts) >= 4:
                    symbol = parts[0]
                    timeframe = parts[1]
                    key = f"{symbol}_{timeframe}" """

new_code = """                # استخراج اسم الزوج والإطار الزمني
                filename = model_file.stem
                parts = filename.split('_')
                if len(parts) >= 4:
                    symbol = parts[0]
                    # parts[1] = "PERIOD", parts[2] = "M5"/"H1"/etc
                    if parts[1] == "PERIOD" and len(parts) >= 3:
                        timeframe = f"{parts[1]}_{parts[2]}"  # PERIOD_M5
                    else:
                        timeframe = parts[1]
                    key = f"{symbol}_{timeframe}"
                    
                    print(f"Loading model: {key} from {filename}") """

content = content.replace(old_code, new_code)

# أيضاً تحديث في predict_with_confidence لطباعة النماذج المتاحة
debug_code = """
            if key not in self.models:
                logger.warning(f"No model found for {key}")
                logger.info(f"Available models: {list(self.models.keys())}")
"""

# البحث عن المكان المناسب
pos = content.find('if key not in self.models:')
if pos > 0:
    # إضافة طباعة النماذج المتاحة
    end_pos = content.find('logger.warning(f"No model found for {key}")', pos)
    if end_pos > 0:
        end_pos = content.find('\n', end_pos)
        content = content[:end_pos] + '\n                logger.info(f"Available models: {list(self.models.keys())}")' + content[end_pos:]

# حفظ الملف المحدث
with open('src/advanced_predictor_95.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed model loading in advanced_predictor_95.py")

# أيضاً إنشاء سكريبت لعرض النماذج المتاحة
debug_script = '''#!/usr/bin/env python3
"""
Debug script to check loaded models
"""

from pathlib import Path
import joblib

model_dir = Path("models/advanced")
if model_dir.exists():
    model_files = list(model_dir.glob("*_ensemble_*.pkl"))
    print(f"\\n📊 Found {len(model_files)} model files:\\n")
    
    for model_file in model_files:
        print(f"File: {model_file.name}")
        
        # Extract key from filename
        filename = model_file.stem
        parts = filename.split('_')
        if len(parts) >= 4:
            symbol = parts[0]
            if parts[1] == "PERIOD" and len(parts) >= 3:
                timeframe = f"{parts[1]}_{parts[2]}"
            else:
                timeframe = parts[1]
            key = f"{symbol}_{timeframe}"
            print(f"  → Key: {key}")
        print()
else:
    print("❌ No models/advanced directory found!")
    print("Current directory:", Path.cwd())
    print("\\nSearching for .pkl files...")
    pkl_files = list(Path(".").rglob("*.pkl"))
    if pkl_files:
        print(f"Found {len(pkl_files)} .pkl files:")
        for f in pkl_files[:10]:
            print(f"  • {f}")
'''

with open('check_models.py', 'w') as f:
    f.write(debug_script)

print("✅ Created check_models.py")
print("\n🚀 Now run:")
print("1. python check_models.py  # للتحقق من النماذج")
print("2. Restart server: python src/mt5_bridge_server_advanced.py")