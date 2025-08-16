#!/usr/bin/env python3
"""
Update Symbol Names - تحديث أسماء العملات في قاعدة البيانات
"""

import sqlite3
from pathlib import Path

def update_symbols():
    """تحديث أسماء العملات لإزالة suffix غير مرغوب"""
    
    # البحث عن قاعدة البيانات
    db_paths = ["data/forex_ml.db", "data/forex_data.db"]
    db_path = None
    
    for path in db_paths:
        if Path(path).exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ لم يتم العثور على قاعدة بيانات!")
        return
    
    print(f"📊 تحديث أسماء العملات في: {db_path}")
    print("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # الحصول على العملات الحالية
        cursor.execute("SELECT DISTINCT symbol FROM price_data")
        symbols = [row[0] for row in cursor.fetchall()]
        
        print(f"🪙 العملات الحالية: {symbols}")
        
        # قائمة التحديثات
        updates = []
        for symbol in symbols:
            new_symbol = symbol
            
            # إزالة suffix
            if symbol.endswith('m'):
                new_symbol = symbol[:-1]
                updates.append((symbol, new_symbol))
            elif symbol.endswith('.a'):
                new_symbol = symbol[:-2]
                updates.append((symbol, new_symbol))
        
        if not updates:
            print("✅ لا توجد تحديثات مطلوبة")
            return
        
        print(f"\n📝 التحديثات المخططة:")
        for old, new in updates:
            print(f"   {old} → {new}")
        
        # السؤال عن التأكيد
        confirm = input("\nهل تريد تطبيق هذه التحديثات؟ (y/n): ").strip().lower()
        
        if confirm == 'y':
            # تطبيق التحديثات
            for old_symbol, new_symbol in updates:
                cursor.execute("""
                    UPDATE price_data 
                    SET symbol = ? 
                    WHERE symbol = ?
                """, (new_symbol, old_symbol))
                
                rows_affected = cursor.rowcount
                print(f"✅ تم تحديث {rows_affected:,} سجل: {old_symbol} → {new_symbol}")
            
            conn.commit()
            print("\n✅ تمت جميع التحديثات بنجاح!")
            
            # إعادة فحص
            cursor.execute("SELECT DISTINCT symbol FROM price_data")
            new_symbols = [row[0] for row in cursor.fetchall()]
            print(f"\n🪙 العملات بعد التحديث: {new_symbols}")
            
        else:
            print("❌ تم إلغاء التحديث")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ خطأ: {e}")

def create_direct_training_script():
    """إنشاء سكريبت تدريب يعمل مع الأسماء الحالية"""
    
    script_content = '''#!/usr/bin/env python3
"""
Direct Training - تدريب مباشر بالأسماء الحالية
"""

from train_models_simple import SimpleModelTrainer

def main():
    trainer = SimpleModelTrainer()
    
    # تحديث أسماء العملات للمطابقة مع قاعدة البيانات
    test_combinations = [
        ("EURUSDm", "M5"),    # أو أي suffix موجود
        ("GBPUSDm", "M15"),
        ("XAUUSDm", "H1"),
        # أضف المزيد حسب ما هو متاح
    ]
    
    print("🚀 تدريب مباشر بالأسماء الحالية")
    
    for symbol, timeframe in test_combinations:
        try:
            print(f"\\n📊 تدريب {symbol} {timeframe}...")
            scores = trainer.train_symbol(symbol, timeframe)
            
            if scores:
                print(f"✅ نجح - دقة: {scores['test_accuracy']:.4f}")
            else:
                print(f"⚠️ بيانات غير كافية")
                
        except Exception as e:
            print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("train_direct.py", "w") as f:
        f.write(script_content)
    
    print("\n✅ تم إنشاء: train_direct.py")
    print("   يمكنك تشغيله مباشرة بدون تحديث الأسماء")

def main():
    print("🔧 أداة تحديث أسماء العملات")
    print("="*60)
    print("\nالخيارات:")
    print("1. تحديث أسماء العملات في قاعدة البيانات")
    print("2. إنشاء سكريبت تدريب بالأسماء الحالية")
    print("3. الخياران معاً")
    
    choice = input("\nاختيارك (1-3): ").strip()
    
    if choice in ['1', '3']:
        update_symbols()
    
    if choice in ['2', '3']:
        create_direct_training_script()

if __name__ == "__main__":
    main()