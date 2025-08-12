#!/usr/bin/env python3
"""
Fix model name extraction to preserve full timeframe
إصلاح استخراج أسماء النماذج
"""

import re

# قراءة advanced_predictor_95.py
with open('src/advanced_predictor_95.py', 'r', encoding='utf-8') as f:
    content = f.read()

# البحث عن الكود الذي يستخرج اسم النموذج
old_patterns = [
    # النمط القديم المحتمل
    """                # استخراج اسم الزوج والإطار الزمني
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
                    
                    print(f"Loading model: {key} from {filename}")""",
    
    # نمط آخر محتمل
    """                filename = model_file.stem
                parts = filename.split('_')
                if len(parts) >= 4:
                    symbol = parts[0]
                    timeframe = parts[1]
                    key = f"{symbol}_{timeframe}" """
]

# الكود الجديد المصحح
new_code = """                # استخراج اسم النموذج بشكل صحيح
                filename = model_file.stem  # مثال: EURUSDm_PERIOD_M5_ensemble_20250812_142405
                
                # إزالة الجزء الأخير (ensemble_timestamp)
                # نبحث عن _ensemble_ ونحذف كل شيء بعده
                if '_ensemble_' in filename:
                    model_key = filename.split('_ensemble_')[0]  # EURUSDm_PERIOD_M5
                else:
                    # احتياطي: إذا لم نجد _ensemble_
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # نأخذ أول 3 أجزاء (Symbol_PERIOD_Timeframe)
                        model_key = '_'.join(parts[:3])
                    else:
                        model_key = filename
                
                print(f"Loading model: {model_key} from {filename}")
                key = model_key"""

# محاولة استبدال النمط الأول
replaced = False
for old_pattern in old_patterns:
    if old_pattern in content:
        content = content.replace(old_pattern, new_code)
        replaced = True
        print("✅ Found and replaced old pattern")
        break

# إذا لم نجد النمط المطابق، نبحث بطريقة أخرى
if not replaced:
    # البحث عن المكان الذي يتم فيه تحميل النماذج
    import_section = content.find("for model_file in model_files:")
    if import_section > 0:
        # نبحث عن نهاية حلقة for
        try_block = content.find("try:", import_section)
        if try_block > 0:
            # نجد نهاية كتلة try
            except_block = content.find("except Exception as e:", try_block)
            if except_block > 0:
                # نستبدل المحتوى بين try و except
                old_content = content[try_block:except_block]
                
                new_try_content = """try:
                # استخراج اسم النموذج بشكل صحيح
                filename = model_file.stem  # مثال: EURUSDm_PERIOD_M5_ensemble_20250812_142405
                
                # إزالة الجزء الأخير (ensemble_timestamp)
                if '_ensemble_' in filename:
                    model_key = filename.split('_ensemble_')[0]  # EURUSDm_PERIOD_M5
                else:
                    # احتياطي
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        model_key = '_'.join(parts[:3])
                    else:
                        model_key = filename
                
                print(f"Loading model: {model_key}")
                
                # تحميل النموذج
                model_data = joblib.load(model_file)
                self.models[model_key] = model_data['model']
                self.scalers[model_key] = model_data.get('scaler')
                self.metrics[model_key] = model_data.get('metrics', {})
                
                loaded_count += 1
                
            """
                
                content = content[:try_block] + new_try_content + content[except_block:]
                replaced = True
                print("✅ Replaced try block content")

if replaced:
    # حفظ الملف المحدث
    with open('src/advanced_predictor_95.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Updated advanced_predictor_95.py")
else:
    print("⚠️ Could not find pattern to replace. Manual fix needed.")

# إنشاء سكريبت اختبار سريع
test_script = '''#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_predictor_95 import AdvancedPredictor

print("Testing model loading...")
predictor = AdvancedPredictor()

print(f"\\nLoaded {len(predictor.models)} models:")
for i, (key, model) in enumerate(predictor.models.items()):
    print(f"{i+1}. {key}")
    if i >= 9:  # Show first 10 only
        print(f"... and {len(predictor.models) - 10} more")
        break

# Test specific models
test_keys = [
    "EURUSDm_PERIOD_M5",
    "EURUSDm_PERIOD_H1", 
    "GBPUSDm_PERIOD_M5",
    "XAUUSDm_PERIOD_H4"
]

print("\\nChecking specific models:")
for key in test_keys:
    exists = key in predictor.models
    print(f"  {key}: {'✅' if exists else '❌'}")
'''

with open('test_model_names.py', 'w') as f:
    f.write(test_script)

print("✅ Created test_model_names.py")
print("\n🚀 Now run:")
print("1. python test_model_names.py")
print("2. python src/mt5_bridge_server_advanced.py")