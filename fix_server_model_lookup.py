#!/usr/bin/env python3
"""
Fix server to use correct model names (with 'm' and underscore)
إصلاح الخادم لاستخدام أسماء النماذج الصحيحة
"""

import os

print("🔧 Fixing server model lookup...")

server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # الإصلاح 1: إزالة محاولة حذف 'm'
    old_block = """                    if model_key not in self.predictor.models:
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
    
    new_block = """                    if model_key not in self.predictor.models:
                        logger.warning(f"No model found for {model_key}")
                        logger.info(f"Available models: {list(self.predictor.models.keys())[:5]}...")
                        return {
                            'action': 'NO_TRADE',
                            'confidence': 0,
                            'reason': f'No model for {model_key}'
                        }"""
    
    if old_block in content:
        content = content.replace(old_block, new_block)
        print("✅ Fixed model lookup logic")
    
    # الإصلاح 2: عدم حذف 'm' عند استدعاء predict_with_confidence
    old_line = "symbol=symbol.rstrip('m'),"
    if old_line in content:
        content = content.replace(old_line, "symbol=symbol,")
        print("✅ Fixed predict call - keeping 'm'")
    
    # الإصلاح 3: التأكد من بناء model_key بشكل صحيح
    # البحث عن المكان الذي يتم فيه بناء model_key
    if 'model_key = f"{symbol}_{model_timeframe}"' not in content:
        # إذا لم نجده، نبحث عن نمط آخر
        old_patterns = [
            'model_key = f"{symbol.rstrip(\'m\')}_{model_timeframe}"',
            'model_key = f"{symbol[:-1]}_{model_timeframe}"',
            'model_key = f"{base_symbol}_{model_timeframe}"'
        ]
        
        for pattern in old_patterns:
            if pattern in content:
                content = content.replace(pattern, 'model_key = f"{symbol}_{model_timeframe}"')
                print(f"✅ Fixed model_key generation: {pattern}")
    
    # حفظ الملف المحدث
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Server file updated successfully!")
else:
    print("❌ Server file not found!")

# إنشاء سكريبت للتحقق من النتيجة
verify_script = '''#!/usr/bin/env python3
"""
Verify model name matching is fixed
"""
import os
import sys

# Test model name generation
test_cases = [
    ("GBPUSDm", "PERIOD_M15"),
    ("EURUSDm", "PERIOD_H1"),
    ("XAUUSDm", "PERIOD_H4")
]

print("🧪 Testing model key generation:")
print("\\nExpected format: SYMBOL_PERIOD_TIMEFRAME")
print("Example: GBPUSDm_PERIOD_M15\\n")

for symbol, timeframe in test_cases:
    # الطريقة الصحيحة
    correct_key = f"{symbol}_{timeframe}"
    
    # الطريقة الخاطئة (مع space)
    wrong_key1 = f"{symbol} {timeframe}"
    
    # الطريقة الخاطئة (بدون m)
    wrong_key2 = f"{symbol.rstrip('m')}_{timeframe}"
    
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")
    print(f"  ✅ Correct: {correct_key}")
    print(f"  ❌ Wrong (space): {wrong_key1}")
    print(f"  ❌ Wrong (no m): {wrong_key2}")
    print()

print("\\n📝 What to look for in server logs:")
print("✅ Good: 'No model found for GBPUSDm_PERIOD_M15'")
print("❌ Bad: 'No model found for GBPUSD PERIOD_M15'")
print("❌ Bad: 'No model found for GBPUSD_PERIOD_M15'")
'''

with open('verify_model_fix.py', 'w') as f:
    f.write(verify_script)

print("\n✅ Created verify_model_fix.py")
print("\n🚀 Now run:")
print("1. python verify_model_fix.py")
print("2. Restart server: python src/mt5_bridge_server_advanced.py")
print("\nThe server should now look for the correct model names!")