#!/usr/bin/env python3
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
matches = re.findall(r'\{symbol\}\s+\{timeframe\}', content)
if matches:
    issues.append("Found {symbol} {timeframe} with space")

# Pattern 2: {symbol} {model_timeframe} with space  
matches = re.findall(r'\{symbol\}\s+\{model_timeframe\}', content)
if matches:
    issues.append("Found {symbol} {model_timeframe} with space")

# Pattern 3: Check model_key construction
matches = re.findall(r'model_key\s*=\s*f"[^"]*"', content)
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
print("\n📝 Model key constructions found:")
for match in re.findall(r'model_key\s*=\s*f"[^"]*"', content):
    print(f"  • {match}")
