#!/usr/bin/env python3
"""
Script to debug database issues
"""

import sqlite3
import pandas as pd

def check_database():
    """فحص قاعدة البيانات بالتفصيل"""
    print("🔍 Checking database structure...\n")
    
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        
        # 1. الجداول الموجودة
        print("📊 Tables in database:")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            print(f"  - {table[0]}")
        
        # 2. هيكل جدول price_data
        print("\n📋 Structure of price_data table:")
        cursor.execute("PRAGMA table_info(price_data)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # 3. عينة من البيانات
        print("\n📈 Sample data:")
        sample_query = "SELECT * FROM price_data LIMIT 5"
        df_sample = pd.read_sql_query(sample_query, conn)
        print(df_sample)
        
        # 4. الرموز المتاحة مع عدد السجلات
        print("\n🎯 Available symbols:")
        symbols_query = """
            SELECT symbol, timeframe, COUNT(*) as count,
                   MIN(time) as first_date, MAX(time) as last_date
            FROM price_data
            GROUP BY symbol, timeframe
            ORDER BY count DESC
        """
        df_symbols = pd.read_sql_query(symbols_query, conn)
        
        for idx, row in df_symbols.iterrows():
            if idx < 10:  # أول 10 فقط
                print(f"  {row['symbol']} {row['timeframe']} - {row['count']:,} records")
        
        # 5. اختبار استعلام محدد
        print("\n🔍 Testing specific queries:")
        test_symbols = ['EURUSD', 'EURUSDm', 'EUR/USD', 'EURUSD.', 'EURUSD.m']
        for symbol in test_symbols:
            cursor.execute("SELECT COUNT(*) FROM price_data WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  ✅ {symbol}: {count:,} records")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database()