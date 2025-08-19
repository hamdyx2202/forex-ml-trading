#!/usr/bin/env python3
"""
🔍 فحص قاعدة البيانات
📊 يتحقق من وجود البيانات المطلوبة للتدريب
"""

import os
import sqlite3
import pandas as pd

def check_database():
    """فحص شامل لقاعدة البيانات"""
    db_path = './data/forex_ml.db'
    
    print("="*60)
    print("🔍 Database Check Tool")
    print("="*60)
    
    # 1. تحقق من وجود الملف
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        print("\n💡 Solution:")
        print("   1. Create ./data directory: mkdir -p ./data")
        print("   2. Copy your database to: ./data/forex_ml.db")
        print("   3. Or run data collection script")
        return
    
    # 2. حجم قاعدة البيانات
    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"✅ Database found: {db_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 3. فحص الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📊 Tables found: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")
        
        # 4. فحص جدول price_data
        if 'price_data' not in [t[0] for t in tables]:
            print("\n❌ Table 'price_data' not found!")
            print("   This table is required for training")
            conn.close()
            return
        
        # 5. إحصائيات البيانات
        print("\n📈 Price Data Statistics:")
        
        # إجمالي السجلات
        cursor.execute("SELECT COUNT(*) FROM price_data")
        total_records = cursor.fetchone()[0]
        print(f"   Total records: {total_records:,}")
        
        # الأطر الزمنية المتاحة
        cursor.execute("SELECT DISTINCT timeframe FROM price_data")
        timeframes = [tf[0] for tf in cursor.fetchall()]
        print(f"   Timeframes: {', '.join(timeframes)}")
        
        # 6. تفاصيل كل إطار زمني
        print("\n📊 Data by timeframe:")
        for tf in ['M15', 'M30', 'H1', 'H4', 'D1']:
            query = f"""
            SELECT COUNT(DISTINCT symbol) as pairs, COUNT(*) as records
            FROM price_data WHERE timeframe = '{tf}'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            if result and result[1] > 0:
                print(f"   {tf}: {result[0]} pairs, {result[1]:,} records")
        
        # 7. أفضل الأزواج للتدريب
        print("\n🏆 Best pairs for training (M15 with >2000 candles):")
        query = """
        SELECT symbol, COUNT(*) as count 
        FROM price_data 
        WHERE timeframe = 'M15'
        GROUP BY symbol 
        HAVING count > 2000
        ORDER BY count DESC
        LIMIT 20
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("   ❌ No pairs with sufficient M15 data!")
            
            # عرض ما هو متاح
            print("\n📊 Available M15 data:")
            query_all = """
            SELECT symbol, COUNT(*) as count 
            FROM price_data 
            WHERE timeframe = 'M15'
            GROUP BY symbol 
            ORDER BY count DESC
            LIMIT 10
            """
            df_all = pd.read_sql_query(query_all, conn)
            for _, row in df_all.iterrows():
                status = "✅" if row['count'] > 2000 else "⚠️"
                print(f"   {status} {row['symbol']}: {row['count']:,} candles")
        else:
            for _, row in df.iterrows():
                print(f"   ✅ {row['symbol']}: {row['count']:,} candles")
        
        # 8. نطاق التواريخ
        print("\n📅 Date range:")
        cursor.execute("SELECT MIN(time), MAX(time) FROM price_data")
        min_date, max_date = cursor.fetchone()
        if min_date and max_date:
            print(f"   From: {min_date}")
            print(f"   To: {max_date}")
        
        conn.close()
        
        # 9. توصيات
        print("\n💡 Recommendations:")
        if total_records == 0:
            print("   ❌ Database is empty - run data collection first")
        elif df.empty:
            print("   ⚠️ Insufficient M15 data - collect more historical data")
        else:
            print(f"   ✅ Database is ready for training!")
            print(f"   ✅ {len(df)} pairs available with sufficient data")
        
    except Exception as e:
        print(f"\n❌ Error reading database: {e}")
        print("   Please check database integrity")

if __name__ == "__main__":
    check_database()