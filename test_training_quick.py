#!/usr/bin/env python3
"""
اختبار سريع للتأكد من عمل التدريب
"""

import sqlite3
from pathlib import Path

def test_training():
    print("🔍 اختبار التدريب السريع...")
    
    # فحص قاعدة البيانات
    db_path = Path("data/forex_ml.db")
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # الحصول على أول زوج متاح
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as count
        FROM price_data
        GROUP BY symbol, timeframe
        HAVING count >= 1000
        ORDER BY count DESC
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        symbol, timeframe, count = result
        print(f"✅ سيتم اختبار: {symbol} {timeframe} ({count:,} سجل)")
        
        # اختبار التدريب البسيط
        from train_models_simple import SimpleModelTrainer
        
        trainer = SimpleModelTrainer()
        scores = trainer.train_symbol(symbol, timeframe)
        
        if scores:
            print(f"✅ نجح التدريب!")
            print(f"   • الدقة: {scores['test_accuracy']:.4f}")
            print(f"   • F1: {scores['f1']:.4f}")
        else:
            print("❌ فشل التدريب")
    else:
        print("❌ لا توجد بيانات كافية!")

if __name__ == "__main__":
    test_training()
