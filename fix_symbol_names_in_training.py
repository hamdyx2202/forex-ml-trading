#!/usr/bin/env python3
"""
إصلاح أسماء الرموز في ملفات التدريب
إزالة suffix 'm' من جميع الأسماء
"""

import os
import glob
import re

def fix_symbol_names():
    """تحديث أسماء الرموز لإزالة 'm' من النهاية"""
    
    # الملفات المطلوب تحديثها
    files_to_update = [
        'train_full_advanced.py',
        'continuous_learner_advanced_v2.py',
        'integrate_continuous_with_mt5.py',
        'train_models_simple.py',
        'train_full_simple.py',
        'train_quick_test.py'
    ]
    
    # قائمة الرموز التي تحتاج تحديث
    symbols_to_fix = [
        ('EURUSDm', 'EURUSD'),
        ('GBPUSDm', 'GBPUSD'),
        ('USDJPYm', 'USDJPY'),
        ('AUDUSDm', 'AUDUSD'),
        ('USDCADm', 'USDCAD'),
        ('NZDUSDm', 'NZDUSD'),
        ('USDCHFm', 'USDCHF'),
        ('XAUUSDm', 'XAUUSD'),
        ('XAGUSDm', 'XAGUSD'),
        ('BTCUSDm', 'BTCUSD'),
        ('ETHUSDm', 'ETHUSD'),
        ('BNBUSDm', 'BNBUSD'),
        ('US30m', 'US30'),
        ('NAS100m', 'NAS100'),
        ('SP500m', 'SP500'),
        ('OILm', 'OIL'),
        ('WTIm', 'WTI'),
    ]
    
    updated_files = []
    
    for file_path in files_to_update:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # استبدال كل رمز
            for old_symbol, new_symbol in symbols_to_fix:
                # استبدال في strings
                content = content.replace(f"'{old_symbol}'", f"'{new_symbol}'")
                content = content.replace(f'"{old_symbol}"', f'"{new_symbol}"')
                # استبدال في lists
                content = content.replace(f" {old_symbol},", f" {new_symbol},")
                content = content.replace(f"[{old_symbol}", f"[{new_symbol}")
                content = content.replace(f" {old_symbol}]", f" {new_symbol}]")
            
            # حفظ الملف إذا تم التعديل
            if content != original_content:
                # حفظ نسخة احتياطية
                backup_path = f"{file_path}.backup_symbols"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # حفظ المحتوى المحدث
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                updated_files.append(file_path)
                print(f"✅ تم تحديث: {file_path}")
        
        except Exception as e:
            print(f"❌ خطأ في {file_path}: {e}")
    
    print(f"\n📊 تم تحديث {len(updated_files)} ملف")
    
    # إنشاء ملف اختبار سريع
    create_quick_test()

def create_quick_test():
    """إنشاء ملف اختبار سريع للتحقق من البيانات"""
    
    test_content = '''#!/usr/bin/env python3
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
'''
    
    with open("test_training_quick.py", "w", encoding='utf-8') as f:
        f.write(test_content)
    
    print("\n✅ تم إنشاء: test_training_quick.py")

def create_update_script():
    """إنشاء سكريبت لتحديث train_advanced_complete.py"""
    
    update_script = '''#!/usr/bin/env python3
"""
تحديث ملف التدريب المتقدم لإزالة suffix من الأسماء
"""

def update_advanced_training():
    # قراءة الملف
    with open("train_advanced_complete.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # البحث عن السطر الذي يحتوي على symbol.replace('m', '')
    # وتحديثه ليزيل suffix بشكل صحيح
    content = content.replace(
        "symbol = row['symbol'].replace('m', '')  # إزالة suffix",
        "symbol = row['symbol']  # استخدام الاسم كما هو"
    )
    
    # أيضاً في train_full_advanced.py
    content = content.replace(
        "pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')",
        "pip_value = self.calculate_pip_value(symbol if symbol else 'EURUSD')"
    )
    
    # حفظ الملف
    with open("train_advanced_complete.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ تم تحديث train_advanced_complete.py")
    
    # نفس الشيء لـ train_full_advanced.py
    try:
        with open("train_full_advanced.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace(
            "symbol = row['symbol'].replace('m', '')  # إزالة suffix",
            "symbol = row['symbol']  # استخدام الاسم كما هو"
        )
        
        with open("train_full_advanced.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ تم تحديث train_full_advanced.py")
    except:
        pass

if __name__ == "__main__":
    update_advanced_training()
'''
    
    with open("update_advanced_training_names.py", "w", encoding='utf-8') as f:
        f.write(update_script)
    
    print("✅ تم إنشاء: update_advanced_training_names.py")

def main():
    print("🔧 إصلاح أسماء الرموز في ملفات التدريب...")
    print("=" * 60)
    
    fix_symbol_names()
    create_update_script()
    
    print("\n✅ تم إصلاح أسماء الرموز!")
    print("\n💡 الخطوات التالية:")
    print("1. تشغيل: python update_advanced_training_names.py")
    print("2. تشغيل: python test_training_quick.py")
    print("3. إعادة التدريب: python train_full_advanced.py")

if __name__ == "__main__":
    main()