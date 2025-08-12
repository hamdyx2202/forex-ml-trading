#!/usr/bin/env python3
"""
Quick fix for model loading issue
"""

import os
import sys
from pathlib import Path

print("🔍 Checking current directory...")
print(f"Current dir: {os.getcwd()}")

print("\n📁 Looking for models directory...")
# Check various possible locations
paths_to_check = [
    "models/advanced",
    "/home/forex-ml-trading/models/advanced",
    "../models/advanced",
    "./models/advanced"
]

model_dir = None
for path in paths_to_check:
    if os.path.exists(path):
        model_dir = path
        print(f"✅ Found models at: {path}")
        break

if not model_dir:
    print("❌ Models directory not found!")
    print("\n🔍 Searching for .pkl files...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pkl"):
                print(f"Found: {os.path.join(root, file)}")
else:
    print(f"\n📊 Models in {model_dir}:")
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            print(f"  • {file}")

# Fix advanced_predictor to use absolute path
print("\n🔧 Fixing model path in advanced_predictor_95.py...")

# Read the file
predictor_path = "src/advanced_predictor_95.py"
if os.path.exists(predictor_path):
    with open(predictor_path, 'r') as f:
        content = f.read()
    
    # Update to use absolute path
    old_line = 'model_dir = Path("models/advanced")'
    new_line = 'model_dir = Path("/home/forex-ml-trading/models/advanced")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(predictor_path, 'w') as f:
            f.write(content)
        print("✅ Updated model path to absolute path")
    else:
        print("⚠️ Path already updated or different format")
else:
    print("❌ advanced_predictor_95.py not found!")

print("\n✅ Done! Restart the server now.")