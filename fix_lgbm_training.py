#!/usr/bin/env python3
"""
🔧 إصلاح خطأ LightGBM
"""

import fileinput
import sys

# إصلاح المشكلة في train_with_real_data.py
filename = 'train_with_real_data.py'

print(f"🔧 Fixing LightGBM verbose parameter in {filename}...")

# قراءة الملف وإصلاحه
with open(filename, 'r') as file:
    content = file.read()

# استبدال verbose بـ verbosity
content = content.replace("'verbose': -1", "'verbosity': -1")

# حفظ التغييرات
with open(filename, 'w') as file:
    file.write(content)

print("✅ Fixed! Now LightGBM will work correctly.")
print("\n📌 You can now continue training with: python3 train_with_real_data.py")