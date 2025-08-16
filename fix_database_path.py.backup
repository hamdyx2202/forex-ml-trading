#!/usr/bin/env python3
"""
Fix Database Path - تصحيح مسار قاعدة البيانات في جميع الملفات
"""

import os
import glob

def fix_database_paths():
    """تحديث مسارات قاعدة البيانات في جميع ملفات Python"""
    print("🔧 تحديث مسارات قاعدة البيانات...")
    
    # البحث عن جميع ملفات Python
    python_files = glob.glob("*.py") + glob.glob("src/*.py")
    
    updates_count = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # التحديثات المطلوبة
            original_content = content
            
            # تحديث المسارات
            content = content.replace('"data/forex_data.db"', '"data/forex_ml.db"')
            content = content.replace("'data/forex_data.db'", "'data/forex_ml.db'")
            content = content.replace('Path("data/forex_data.db")', 'Path("data/forex_ml.db")')
            content = content.replace("data/forex_data.db", "data/forex_ml.db")
            
            # حفظ الملف إذا تم التعديل
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ تم تحديث: {file_path}")
                updates_count += 1
                
        except Exception as e:
            print(f"⚠️ تخطي {file_path}: {e}")
    
    print(f"\n📊 تم تحديث {updates_count} ملف")
    
def show_database_info():
    """عرض معلومات قاعدة البيانات الحالية"""
    import sqlite3
    from pathlib import Path
    
    db_path = Path("data/forex_ml.db")
    
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # إحصائيات عامة
        cursor.execute("SELECT COUNT(*) FROM price_data")
        total_records = cursor.fetchone()[0]
        
        # إحصائيات حسب العملة والفريم
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        
        stats = cursor.fetchall()
        
        print(f"\n📊 معلومات قاعدة البيانات: {db_path}")
        print(f"   إجمالي السجلات: {total_records:,}")
        print(f"   عدد الأزواج: {len(stats)}")
        
        print("\n📈 عينة من البيانات المتاحة:")
        for symbol, timeframe, count in stats[:10]:
            print(f"   • {symbol} {timeframe}: {count:,} سجل")
        
        if len(stats) > 10:
            print(f"   • ... و {len(stats) - 10} زوج آخر")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ خطأ: {e}")

def main():
    """الدالة الرئيسية"""
    print("🚀 تصحيح مسارات قاعدة البيانات")
    print("="*60)
    
    # تحديث المسارات
    fix_database_paths()
    
    # عرض معلومات قاعدة البيانات
    show_database_info()
    
    print("\n✅ يمكنك الآن تشغيل التدريب مباشرة:")
    print("   python train_advanced_complete.py")
    print("\n   أو:")
    print("   python train_models_simple.py")

if __name__ == "__main__":
    main()