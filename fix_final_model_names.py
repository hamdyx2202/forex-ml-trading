#!/usr/bin/env python3
"""
Final fix - Server must use exact model names
الإصلاح النهائي - يجب أن يستخدم الخادم نفس أسماء النماذج
"""

import os

print("🔧 Fixing model name mismatch...")

# Fix in mt5_bridge_server_advanced.py
server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # البحث عن المكان الذي يتم فيه بناء model_key
    old_patterns = [
        # النمط المحتمل 1
        'model_key = f"{symbol}_{model_timeframe}"',
        # النمط المحتمل 2  
        'model_key = f"{symbol.rstrip(\'m\')}_{model_timeframe}"',
        # النمط المحتمل 3
        'alt_key = f"{symbol.rstrip(\'m\')}_{model_timeframe}"'
    ]
    
    # استبدال كل الأنماط
    replaced = False
    for pattern in old_patterns:
        if pattern in content:
            # لا نحذف 'm' ونستخدم underscore
            new_pattern = pattern.replace('.rstrip(\'m\')', '')
            content = content.replace(pattern, new_pattern)
            replaced = True
            print(f"✅ Fixed pattern: {pattern}")
    
    # البحث عن نمط آخر محتمل في predict_with_confidence
    if 'predict_with_confidence' in content:
        # تحديث الجزء الخاص بإزالة m
        old_code = """                    if model_key not in self.predictor.models:
                        logger.warning(f"No model found for {model_key}")
                        # محاولة بدون suffix
                        alt_key = f"{symbol.rstrip('m')}_{model_timeframe}"
                        if alt_key in self.predictor.models:
                            model_key = alt_key
                        else:
                            return {
                                'action': 'NO_TRADE',
                                'confidence': 0,
                                'reason': f'No model for {symbol} {timeframe}'
                            }"""
        
        new_code = """                    if model_key not in self.predictor.models:
                        logger.warning(f"No model found for {model_key}")
                        logger.info(f"Available models: {list(self.predictor.models.keys())[:5]}...")
                        return {
                            'action': 'NO_TRADE',
                            'confidence': 0,
                            'reason': f'No model for {symbol} {timeframe}'
                        }"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            replaced = True
            print("✅ Removed alt_key logic")
    
    if replaced:
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Updated mt5_bridge_server_advanced.py")
    else:
        print("⚠️ Could not find patterns to replace in server")

# Fix in advanced_predictor_95.py
predictor_file = "src/advanced_predictor_95.py"
if os.path.exists(predictor_file):
    with open(predictor_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # البحث عن predict_with_confidence
    if 'predict_with_confidence' in content:
        # تحديث بناء model_key
        old_patterns = [
            'key = f"{symbol}_{timeframe}"',
            'model_key = f"{symbol}_{timeframe}"',
            'key = f"{symbol.rstrip(\'m\')}_{timeframe}"'
        ]
        
        replaced = False
        for pattern in old_patterns:
            if pattern in content:
                new_pattern = pattern.replace('.rstrip(\'m\')', '')
                content = content.replace(pattern, new_pattern)
                replaced = True
                print(f"✅ Fixed in predictor: {pattern}")
        
        # أيضاً تحديث أي مكان يحذف m
        if 'symbol.rstrip(\'m\')' in content:
            content = content.replace('symbol.rstrip(\'m\')', 'symbol')
            replaced = True
            print("✅ Removed all rstrip('m') calls")
        
        if replaced:
            with open(predictor_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Updated advanced_predictor_95.py")

# إنشاء سكريبت اختبار نهائي
test_script = '''#!/usr/bin/env python3
"""
Final test - Check model name matching
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_predictor_95 import AdvancedPredictor

print("🔍 Testing model name matching...")

# تحميل النماذج
predictor = AdvancedPredictor()
print(f"\\n✅ Loaded {len(predictor.models)} models")

# عرض بعض النماذج
print("\\nSample model names:")
for i, key in enumerate(list(predictor.models.keys())[:5]):
    print(f"  {i+1}. {key}")

# اختبار التطابق
test_cases = [
    ("GBPUSDm", "PERIOD_M15"),
    ("EURUSDm", "PERIOD_M5"),
    ("XAUUSDm", "PERIOD_H1"),
    ("USDJPYm", "PERIOD_H4")
]

print("\\n🧪 Testing key generation:")
for symbol, timeframe in test_cases:
    # هذا ما يجب أن يستخدمه الخادم
    expected_key = f"{symbol}_{timeframe}"
    exists = expected_key in predictor.models
    print(f"  {symbol} + {timeframe} → {expected_key}: {'✅' if exists else '❌'}")

# اختبار بدون m
print("\\n⚠️ Testing without 'm' (should fail):")
for symbol, timeframe in test_cases:
    wrong_key = f"{symbol.rstrip('m')}_{timeframe}"
    exists = wrong_key in predictor.models
    print(f"  {symbol} → {wrong_key}: {'✅ WARNING!' if exists else '❌ Expected'}")
'''

with open('test_final_names.py', 'w') as f:
    f.write(test_script)

print("✅ Created test_final_names.py")

print("\n🚀 Final steps:")
print("1. python test_final_names.py")
print("2. Restart server: python src/mt5_bridge_server_advanced.py")
print("\nThe models should now match exactly! 🎯")