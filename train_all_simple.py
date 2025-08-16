#!/usr/bin/env python3
"""
Train All Models - تدريب جميع العملات المتاحة
"""

import sys
import time
from datetime import datetime
from train_models_simple import SimpleModelTrainer

def main():
    """تدريب جميع العملات المتاحة"""
    print("🌟 بدء التدريب الشامل لجميع العملات")
    print("="*80)
    print(f"🕐 وقت البدء: {datetime.now()}")
    print("="*80)
    
    start_time = time.time()
    
    # إنشاء المدرب
    trainer = SimpleModelTrainer()
    
    # تدريب جميع العملات
    trainer.train_all()
    
    # حساب الوقت المستغرق
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "="*80)
    print(f"✅ اكتمل التدريب الشامل!")
    print(f"⏱️ الوقت المستغرق: {hours} ساعة و {minutes} دقيقة و {seconds} ثانية")
    print(f"🕐 وقت الانتهاء: {datetime.now()}")
    print("="*80)
    
    print("\n💡 الخطوات التالية:")
    print("  1. راجع النماذج في مجلد models/")
    print("  2. اختبر النماذج باستخدام test_models.py")
    print("  3. شغّل السيرفر لاستخدام النماذج")

if __name__ == "__main__":
    main()