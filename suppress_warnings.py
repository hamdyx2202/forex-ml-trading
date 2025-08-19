#!/usr/bin/env python3
"""
🔇 إخفاء تحذيرات الأداء أثناء التدريب
"""

import warnings
import os

# إضافة هذا في بداية السكريبتات
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

# أو تشغيل مع متغير بيئة
os.environ['PYTHONWARNINGS'] = 'ignore::pandas.errors.PerformanceWarning'

print("✅ Performance warnings suppressed")
print("\nلتشغيل التدريب بدون تحذيرات:")
print("python3 -W ignore::pandas.errors.PerformanceWarning train_all_pairs_enhanced.py")
print("\nأو:")
print("export PYTHONWARNINGS='ignore::pandas.errors.PerformanceWarning'")
print("python3 train_all_pairs_enhanced.py")