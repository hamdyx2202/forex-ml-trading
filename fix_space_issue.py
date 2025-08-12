#!/usr/bin/env python3
"""
Fix the space issue in model key generation
إصلاح مشكلة المسافة في توليد مفتاح النموذج
"""

import os
import re

print("🔧 Fixing space issue in model key...")

server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The issue is that the server shows "GBPUSD PERIOD_M15" with a space
    # This might be in the error message formatting
    
    # Find all places where we format error messages with symbol and timeframe
    # Replace patterns that might introduce a space
    
    # Pattern 1: Error messages with {symbol} {timeframe}
    content = re.sub(
        r"f'No model for \{symbol\} \{timeframe\}'",
        "f'No model for {model_key}'",
        content
    )
    
    # Pattern 2: Any other error message patterns
    content = re.sub(
        r"f'No model for \{symbol\} \{model_timeframe\}'",
        "f'No model for {model_key}'",
        content
    )
    
    # Pattern 3: Logger messages
    content = re.sub(
        r'logger\.warning\(f"No model found for \{symbol\} \{timeframe\}"\)',
        'logger.warning(f"No model found for {model_key}")',
        content
    )
    
    # Pattern 4: Make sure model_key uses underscore
    # Find the specific block
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        # If this line constructs model_key, ensure underscore
        if 'model_key = f"' in line and '{symbol}' in line and '{model_timeframe}' in line:
            # Make sure there's an underscore between symbol and model_timeframe
            if '_{model_timeframe}' not in line:
                line = line.replace('{model_timeframe}', '_{model_timeframe}')
                print(f"✅ Fixed line {i+1}: Added underscore")
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # Save the file
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed space issue in server")
else:
    print("❌ Server file not found!")

# Create a verification script
verify_script = '''#!/usr/bin/env python3
"""
Verify the fix worked
التحقق من نجاح الإصلاح
"""

print("🔍 Checking for space issues in server...")

with open("src/mt5_bridge_server_advanced.py", "r") as f:
    content = f.read()

# Check for problematic patterns
issues = []

# Pattern 1: {symbol} {timeframe} with space
import re
matches = re.findall(r'\\{symbol\\}\\s+\\{timeframe\\}', content)
if matches:
    issues.append("Found {symbol} {timeframe} with space")

# Pattern 2: {symbol} {model_timeframe} with space  
matches = re.findall(r'\\{symbol\\}\\s+\\{model_timeframe\\}', content)
if matches:
    issues.append("Found {symbol} {model_timeframe} with space")

# Pattern 3: Check model_key construction
matches = re.findall(r'model_key\\s*=\\s*f"[^"]*"', content)
for match in matches:
    if '} {' in match:  # Space between interpolations
        issues.append(f"Model key with space: {match}")

if issues:
    print("❌ Found issues:")
    for issue in issues:
        print(f"  • {issue}")
else:
    print("✅ No space issues found!")

# Show model_key constructions
print("\\n📝 Model key constructions found:")
for match in re.findall(r'model_key\\s*=\\s*f"[^"]*"', content):
    print(f"  • {match}")
'''

with open('verify_space_fix.py', 'w') as f:
    f.write(verify_script)

print("✅ Created verify_space_fix.py")
print("\n🚀 Run: python3 verify_space_fix.py")