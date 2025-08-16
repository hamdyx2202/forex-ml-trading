#!/usr/bin/env python3
"""
Check Available Data - فحص البيانات المتاحة في قاعدة البيانات
"""

import sqlite3
import pandas as pd
from pathlib import Path

def check_data():
    """فحص البيانات المتاحة"""
    # تحديد قاعدة البيانات الصحيحة
    db_paths = [
        "data/forex_ml.db",
        "data/forex_data.db",
        "forex_data.db"
    ]
    
    db_path = None
    for path in db_paths:
        if Path(path).exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ لم يتم العثور على قاعدة بيانات!")
        return
    
    print(f"📊 فحص قاعدة البيانات: {db_path}")
    print("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # الحصول على جميع العملات والفريمات
        query = """
            SELECT symbol, timeframe, COUNT(*) as count,
                   MIN(time) as first_record,
                   MAX(time) as last_record
            FROM price_data
            GROUP BY symbol, timeframe
            ORDER BY count DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("❌ لا توجد بيانات في قاعدة البيانات!")
            return
        
        print(f"✅ عدد الأزواج المتاحة: {len(df)}")
        print(f"📈 إجمالي السجلات: {df['count'].sum():,}")
        
        print("\n📋 البيانات المتاحة:")
        print("-"*60)
        print(f"{'Symbol':<15} {'Timeframe':<10} {'Records':<15} {'Sufficient':<10}")
        print("-"*60)
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            count = row['count']
            sufficient = "✅" if count >= 1000 else "❌"
            
            print(f"{symbol:<15} {timeframe:<10} {count:<15,} {sufficient:<10}")
        
        # إحصائيات
        print("\n📊 إحصائيات:")
        print(f"   عملات بيانات كافية (1000+): {len(df[df['count'] >= 1000])}")
        print(f"   عملات بيانات قليلة (<1000): {len(df[df['count'] < 1000])}")
        
        # أسماء العملات الفريدة
        unique_symbols = df['symbol'].unique()
        print(f"\n🪙 العملات المتاحة: {', '.join(unique_symbols)}")
        
        # اقتراحات
        print("\n💡 ملاحظات:")
        if any('m' in s for s in unique_symbols):
            print("   • العملات تحتوي على suffix 'm' (مثل EURUSDm)")
            print("   • يجب تحديث أسماء العملات في ملفات التدريب")
        
        conn.close()
        
        # إنشاء ملف تدريب محدث
        create_updated_training_script(df)
        
    except Exception as e:
        print(f"❌ خطأ: {e}")

def create_updated_training_script(df):
    """إنشاء سكريبت تدريب محدث بالعملات الصحيحة"""
    
    # الحصول على أفضل العملات (أكثر بيانات)
    top_pairs = df[df['count'] >= 1000].head(10)
    
    if top_pairs.empty:
        print("\n⚠️ لا توجد عملات بيانات كافية!")
        return
    
    script_content = '''#!/usr/bin/env python3
"""
Quick Training with Available Data - تدريب سريع بالبيانات المتاحة
"""

from train_models_simple import SimpleModelTrainer
import sys

def main():
    print("🚀 تدريب سريع بالعملات المتاحة")
    print("="*60)
    
    trainer = SimpleModelTrainer()
    
    # العملات المتاحة مع بيانات كافية
    available_pairs = [
'''
    
    # إضافة العملات المتاحة
    for _, row in top_pairs.iterrows():
        script_content += f'        ("{row["symbol"]}", "{row["timeframe"]}"),  # {row["count"]:,} records\n'
    
    script_content += '''    ]
    
    successful = 0
    failed = 0
    
    print(f"📊 سيتم تدريب {len(available_pairs)} زوج")
    
    for symbol, timeframe in available_pairs:
        try:
            print(f"\\n📈 تدريب {symbol} {timeframe}...")
            scores = trainer.train_symbol(symbol, timeframe)
            
            if scores:
                successful += 1
                print(f"✅ نجح - دقة: {scores['test_accuracy']:.4f}")
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            print(f"❌ خطأ: {str(e)}")
    
    print("\\n" + "="*60)
    print("📊 الملخص النهائي")
    print("="*60)
    print(f"✅ نجح: {successful}")
    print(f"❌ فشل: {failed}")
    
    if successful > 0:
        print(f"📈 معدل النجاح: {successful/(successful+failed)*100:.1f}%")

if __name__ == "__main__":
    main()
'''
    
    # حفظ السكريبت
    with open("train_available_data.py", "w", encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n✅ تم إنشاء سكريبت تدريب محدث: train_available_data.py")
    print("🚀 لتشغيله: python train_available_data.py")

if __name__ == "__main__":
    check_data()