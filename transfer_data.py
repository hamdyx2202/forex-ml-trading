#!/usr/bin/env python3
"""
Transfer Data - نقل البيانات بين قواعد البيانات
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def transfer_price_data(source_db, target_db):
    """نقل بيانات الأسعار من قاعدة لأخرى"""
    print(f"📤 نقل البيانات من {source_db} إلى {target_db}...")
    
    try:
        # الاتصال بقواعد البيانات
        source_conn = sqlite3.connect(source_db)
        target_conn = sqlite3.connect(target_db)
        
        # إنشاء الجدول في قاعدة البيانات الهدف
        target_conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                spread INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, time)
            )
        """)
        
        # إنشاء الفهارس
        target_conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
        target_conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
        
        # قراءة البيانات على دفعات
        batch_size = 10000
        offset = 0
        total_transferred = 0
        
        while True:
            print(f"⏳ معالجة الدفعة من {offset} إلى {offset + batch_size}...")
            
            # قراءة دفعة من البيانات
            query = f"""
                SELECT symbol, timeframe, time, open, high, low, close, volume, spread
                FROM price_data
                LIMIT {batch_size} OFFSET {offset}
            """
            
            df = pd.read_sql_query(query, source_conn)
            
            if df.empty:
                break
            
            # نقل البيانات
            df.to_sql('price_data', target_conn, if_exists='append', index=False)
            
            total_transferred += len(df)
            offset += batch_size
            
            print(f"✅ تم نقل {total_transferred:,} سجل")
        
        # حفظ التغييرات
        target_conn.commit()
        
        # التحقق من النقل
        source_count = pd.read_sql_query("SELECT COUNT(*) as count FROM price_data", source_conn)['count'][0]
        target_count = pd.read_sql_query("SELECT COUNT(*) as count FROM price_data", target_conn)['count'][0]
        
        print(f"\n📊 ملخص النقل:")
        print(f"   المصدر: {source_count:,} سجل")
        print(f"   الهدف: {target_count:,} سجل")
        
        if source_count == target_count:
            print("✅ تم نقل جميع البيانات بنجاح!")
        else:
            print(f"⚠️ هناك فرق: {source_count - target_count:,} سجل")
        
        # إغلاق الاتصالات
        source_conn.close()
        target_conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في نقل البيانات: {e}")
        return False

def quick_transfer():
    """نقل سريع من forex_ml.db إلى forex_data.db"""
    source = Path("data/forex_ml.db")
    target = Path("data/forex_data.db")
    
    if not source.exists():
        print(f"❌ قاعدة البيانات المصدر غير موجودة: {source}")
        return False
    
    print("🚀 نقل البيانات من forex_ml.db إلى forex_data.db")
    print("="*60)
    
    # نقل البيانات
    success = transfer_price_data(source, target)
    
    if success:
        print("\n✅ يمكنك الآن تشغيل التدريب:")
        print("   python train_advanced_complete.py")
    
    return success

def main():
    """الدالة الرئيسية"""
    if len(sys.argv) > 1 and '--quick' in sys.argv:
        quick_transfer()
    elif len(sys.argv) >= 5 and sys.argv[1] == '--from' and sys.argv[3] == '--to':
        source = sys.argv[2]
        target = sys.argv[4]
        transfer_price_data(source, target)
    else:
        # النقل التلقائي
        quick_transfer()

if __name__ == "__main__":
    main()