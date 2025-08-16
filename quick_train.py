#!/usr/bin/env python3
"""
Quick Training Script - تدريب سريع للاختبار
"""

import sys
import os

# استخدام النموذج المبسط المتاح
from train_models_simple import SimpleModelTrainer

def main():
    """تشغيل تدريب سريع"""
    print("🚀 بدء التدريب السريع")
    print("="*60)
    
    # إنشاء المدرب
    trainer = SimpleModelTrainer()
    
    # عملات للاختبار السريع
    test_combinations = [
        ("EURUSD", "M5"),
        ("GBPUSD", "M15"),
        ("XAUUSD", "H1"),
        ("BTCUSD", "H4"),
        ("USDJPY", "M30"),
        ("EURJPY", "H1")
    ]
    
    successful = 0
    failed = 0
    
    # تدريب كل عملة
    for symbol, timeframe in test_combinations:
        try:
            print(f"\n📊 تدريب {symbol} {timeframe}...")
            scores = trainer.train_symbol(symbol, timeframe)
            
            if scores:
                successful += 1
                print(f"✅ نجح - دقة: {scores['test_accuracy']:.4f}")
            else:
                failed += 1
                print(f"⚠️ فشل - بيانات غير كافية")
                
        except Exception as e:
            failed += 1
            print(f"❌ خطأ: {str(e)}")
    
    # الملخص
    print("\n" + "="*60)
    print("📊 ملخص التدريب السريع")
    print("="*60)
    print(f"✅ نجح: {successful}")
    print(f"❌ فشل: {failed}")
    print(f"📈 معدل النجاح: {successful/(successful+failed)*100:.1f}%")
    
    print("\n✅ اكتمل التدريب السريع!")
    
    # نصائح
    if failed > 0:
        print("\n💡 نصائح:")
        print("  • تأكد من وجود بيانات كافية (1000+ سجل)")
        print("  • تحقق من اتصال قاعدة البيانات")
        print("  • راجع السجلات للتفاصيل")

if __name__ == "__main__":
    main()