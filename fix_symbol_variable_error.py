#!/usr/bin/env python3
"""
إصلاح خطأ المتغير symbol في ملفات التدريب
"""

def fix_train_advanced_complete():
    """إصلاح خطأ symbol في train_advanced_complete.py"""
    
    with open("train_advanced_complete.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # إصلاح في calculate_dynamic_sl_tp
    content = content.replace(
        "pip_value = self.calculate_pip_value(symbol if symbol else 'EURUSD')",
        "pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')"
    )
    
    # إصلاح في create_advanced_targets_with_sl_tp
    content = content.replace(
        "pip_value = self.calculate_pip_value(symbol if symbol else 'EURUSD')",
        "pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')"
    )
    
    with open("train_advanced_complete.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ تم إصلاح train_advanced_complete.py")

def test_available_data():
    """اختبار سريع للبيانات المتاحة"""
    
    test_script = '''#!/usr/bin/env python3
"""
اختبار البيانات المتاحة وتشغيل تدريب سريع
"""

import sqlite3
from pathlib import Path

def test_data_and_train():
    db_path = Path("data/forex_ml.db")
    
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة!")
        return
    
    print("✅ قاعدة البيانات موجودة")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # الحصول على أول 5 أزواج مع بيانات كافية
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as count
        FROM price_data
        GROUP BY symbol, timeframe
        HAVING count >= 1000
        ORDER BY count DESC
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("❌ لا توجد بيانات كافية!")
        return
    
    print("\\n📊 البيانات المتاحة للتدريب:")
    print("-" * 50)
    for symbol, timeframe, count in results:
        print(f"{symbol:<10} {timeframe:<5} {count:,} سجل")
    
    # اختبار التدريب البسيط على أول زوج
    symbol, timeframe, count = results[0]
    print(f"\\n🚀 اختبار التدريب على {symbol} {timeframe}...")
    
    try:
        from train_models_simple import SimpleModelTrainer
        trainer = SimpleModelTrainer()
        scores = trainer.train_symbol(symbol, timeframe)
        
        if scores:
            print("✅ نجح التدريب!")
            print(f"   • الدقة: {scores['test_accuracy']:.4f}")
            print(f"   • F1: {scores['f1']:.4f}")
            
            # اختبار التدريب المتقدم
            print("\\n🎯 اختبار التدريب المتقدم...")
            from train_advanced_complete import AdvancedCompleteTrainer
            adv_trainer = AdvancedCompleteTrainer()
            adv_trainer.min_data_points = 1000  # تقليل الحد الأدنى للاختبار
            
            results = adv_trainer.train_symbol_advanced(symbol, timeframe)
            if results:
                print("✅ نجح التدريب المتقدم!")
                print(f"   • أفضل دقة: {results['best_accuracy']:.4f}")
                print(f"   • أفضل استراتيجية: {results['best_strategy']}")
        else:
            print("❌ فشل التدريب")
            
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_and_train()
'''
    
    with open("test_data_and_train.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ تم إنشاء test_data_and_train.py")

def main():
    print("🔧 إصلاح خطأ المتغير symbol...")
    fix_train_advanced_complete()
    test_available_data()
    
    print("\n✅ تم الإصلاح!")
    print("\n💡 الخطوات التالية:")
    print("1. تشغيل: python test_data_and_train.py")
    print("2. إذا نجح: python train_full_advanced.py")

if __name__ == "__main__":
    main()