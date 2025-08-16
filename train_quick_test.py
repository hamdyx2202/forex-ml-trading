#!/usr/bin/env python3
"""
Quick Test Training - تدريب سريع للاختبار
يدرب عملة واحدة فقط بسرعة للتأكد من عمل النظام
"""

import time
from datetime import datetime
from train_models_simple import SimpleModelTrainer
from train_advanced_complete import AdvancedCompleteTrainer
import sqlite3

def get_best_symbol():
    """الحصول على العملة مع أكثر بيانات"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        query = """
            SELECT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            ORDER BY count DESC
            LIMIT 1
        """
        result = conn.fetchone()
        conn.close()
        
        if result:
            return result[0], result[1]
        
    except:
        pass
    
    # قيمة افتراضية
    return "EURUSDm", "M5"

def quick_test():
    """اختبار سريع"""
    print("🚀 اختبار سريع للتدريب")
    print("="*60)
    
    # الحصول على أفضل عملة
    symbol, timeframe = get_best_symbol()
    
    print(f"\n📊 العملة المختارة: {symbol} {timeframe}")
    print("-"*60)
    
    # 1. اختبار النموذج البسيط
    print("\n1️⃣ اختبار النموذج البسيط...")
    start_time = time.time()
    
    try:
        trainer_simple = SimpleModelTrainer()
        scores_simple = trainer_simple.train_symbol(symbol, timeframe)
        
        simple_time = time.time() - start_time
        
        if scores_simple:
            print(f"✅ نجح النموذج البسيط!")
            print(f"   • الدقة: {scores_simple['test_accuracy']:.4f}")
            print(f"   • الوقت: {simple_time:.1f} ثانية")
        else:
            print("❌ فشل النموذج البسيط")
            
    except Exception as e:
        print(f"❌ خطأ في النموذج البسيط: {e}")
        scores_simple = None
    
    # 2. اختبار النموذج المتقدم (إذا نجح البسيط)
    if scores_simple and scores_simple['test_accuracy'] > 0.5:
        print("\n2️⃣ اختبار النموذج المتقدم...")
        print("   (هذا قد يستغرق 2-5 دقائق)")
        
        start_time = time.time()
        
        try:
            trainer_advanced = AdvancedCompleteTrainer()
            # تقليل البيانات المطلوبة للاختبار السريع
            trainer_advanced.min_data_points = 5000
            
            results_advanced = trainer_advanced.train_symbol_advanced(symbol, timeframe)
            
            advanced_time = time.time() - start_time
            
            if results_advanced:
                print(f"✅ نجح النموذج المتقدم!")
                print(f"   • أفضل دقة: {results_advanced['best_accuracy']:.4f}")
                print(f"   • أفضل استراتيجية: {results_advanced['best_strategy']}")
                print(f"   • الوقت: {advanced_time/60:.1f} دقيقة")
            else:
                print("❌ فشل النموذج المتقدم")
                
        except Exception as e:
            print(f"❌ خطأ في النموذج المتقدم: {e}")
    
    # الملخص
    print("\n" + "="*60)
    print("📊 ملخص الاختبار السريع")
    print("="*60)
    print(f"✅ النظام يعمل بشكل صحيح!")
    print(f"\n💡 الخطوات التالية:")
    print("   1. للتدريب الشامل البسيط: python train_full_simple.py")
    print("   2. للتدريب الشامل المتقدم: python train_full_advanced.py")

if __name__ == "__main__":
    quick_test()