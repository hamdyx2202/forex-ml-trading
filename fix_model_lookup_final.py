#!/usr/bin/env python3
"""
Final comprehensive fix for model name matching
إصلاح شامل نهائي لمطابقة أسماء النماذج
"""

import os
import re

print("🔧 Applying comprehensive model name fix...")

# Fix 1: Update mt5_bridge_server_advanced.py
server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find where model_timeframe is created
    # It should be PERIOD_M5 not PERIOD M5 (no space)
    
    # Look for timeframe mapping
    old_mapping = """timeframe_map = {
                    'M5': 'PERIOD_M5',
                    'M15': 'PERIOD_M15', 
                    'H1': 'PERIOD_H1',
                    'H4': 'PERIOD_H4'
                }"""
    
    # Make sure there's no space being added anywhere
    # The issue shows "GBPUSD PERIOD_M15" with a space
    
    # Find where model_key is being constructed
    # Replace any pattern that might add a space
    patterns_to_fix = [
        # Pattern 1: Space instead of underscore
        (r'model_key = f"{symbol} {model_timeframe}"', 'model_key = f"{symbol}_{model_timeframe}"'),
        # Pattern 2: Removing m suffix
        (r'model_key = f"{symbol\.rstrip\(\'m\'\)}_{model_timeframe}"', 'model_key = f"{symbol}_{model_timeframe}"'),
        # Pattern 3: Alt key with rstrip
        (r'alt_key = f"{symbol\.rstrip\(\'m\'\)}_{model_timeframe}"', 'alt_key = f"{symbol}_{model_timeframe}"'),
    ]
    
    for old_pattern, new_pattern in patterns_to_fix:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Remove all rstrip('m') calls
    content = content.replace("symbol.rstrip('m')", "symbol")
    content = content.replace('.rstrip("m")', '')
    content = content.replace(".rstrip('m')", '')
    
    # Fix the specific block that tries alternative keys
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
        print("✅ Fixed model lookup block")
    
    # Make sure model_key construction uses underscore
    # Find the section where model_key is built
    if "for symbol_request in data['requests']:" in content:
        # Look for model_key construction
        lines = content.split('\n')
        new_lines = []
        in_request_loop = False
        
        for line in lines:
            if "for symbol_request in data['requests']:" in line:
                in_request_loop = True
            
            # Fix any line that constructs model_key with space
            if in_request_loop and "model_key =" in line:
                # Ensure underscore between symbol and timeframe
                line = re.sub(r'f"{symbol}\s+{model_timeframe}"', r'f"{symbol}_{model_timeframe}"', line)
                line = re.sub(r'f"{symbol} {model_timeframe}"', r'f"{symbol}_{model_timeframe}"', line)
                
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
    
    # Save updated file
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated mt5_bridge_server_advanced.py")
else:
    print("❌ Server file not found!")

# Fix 2: Create a debug version to see exact model key generation
debug_script = '''#!/usr/bin/env python3
"""
Debug model key generation
تصحيح توليد مفاتيح النماذج
"""

# Test cases
test_symbols = ["GBPUSDm", "EURUSDm", "XAUUSDm"]
test_timeframes = ["M5", "M15", "H1", "H4"]

print("🔍 Testing model key generation:\\n")

# Correct timeframe mapping
timeframe_map = {
    'M5': 'PERIOD_M5',
    'M15': 'PERIOD_M15',
    'H1': 'PERIOD_H1',
    'H4': 'PERIOD_H4'
}

print("✅ CORRECT way (with underscore):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        model_key = f"{symbol}_{model_timeframe}"
        print(f"  {symbol} + {tf} → {model_key}")

print("\\n❌ WRONG way (with space):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        # This creates the wrong key with space
        wrong_key = f"{symbol} {model_timeframe}"
        print(f"  {symbol} + {tf} → {wrong_key}")

print("\\n❌ WRONG way (without m):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        # This removes the m suffix
        wrong_key = f"{symbol.rstrip('m')}_{model_timeframe}"
        print(f"  {symbol} + {tf} → {wrong_key}")

print("\\n📝 The server MUST use the first format (with underscore)!")
'''

with open('debug_model_keys.py', 'w') as f:
    f.write(debug_script)

print("✅ Created debug_model_keys.py")

# Fix 3: Create a patch to add debug logging
patch_script = '''#!/usr/bin/env python3
"""
Add debug logging to server
إضافة سجلات تصحيح للخادم
"""

import os

server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add debug logging after model_key construction
    # Find where model_key is constructed
    lines = content.split('\\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add debug log after model_key assignment
        if 'model_key = f"{symbol}_{model_timeframe}"' in line:
            indent = len(line) - len(line.lstrip())
            debug_line = ' ' * indent + 'logger.info(f"🔍 Model key: {model_key} (symbol={symbol}, timeframe={model_timeframe})")'
            new_lines.append(debug_line)
        
        # Also log when searching for model
        if 'if model_key not in self.predictor.models:' in line:
            indent = len(line) - len(line.lstrip())
            debug_line = ' ' * indent + 'logger.info(f"🔍 Searching for model: {model_key}")'
            new_lines.insert(-1, debug_line)
    
    content = '\\n'.join(new_lines)
    
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added debug logging to server")
'''

with open('add_debug_logging.py', 'w') as f:
    f.write(patch_script)

print("✅ Created add_debug_logging.py")

print("\n🚀 Steps to fix the issue:")
print("1. python debug_model_keys.py  # See correct format")
print("2. python add_debug_logging.py  # Add debug logs") 
print("3. Restart server and check logs")
print("\nThe server should now use GBPUSDm_PERIOD_M15 (not GBPUSD PERIOD_M15)!")