#!/usr/bin/env python3
"""
Check available symbols in database
"""
import sqlite3
import pandas as pd

# الاتصال بقاعدة البيانات
conn = sqlite3.connect("data/forex_ml.db")

# الحصول على الرموز والأطر الزمنية المتاحة
query = """
    SELECT symbol, timeframe, COUNT(*) as count
    FROM price_data
    GROUP BY symbol, timeframe
    HAVING count >= 1000
    ORDER BY symbol, timeframe
"""

df = pd.read_sql_query(query, conn)
conn.close()

# عرض النتائج
print("Available symbols with data:\n")
print(df.to_string())

# الرموز الفريدة
unique_symbols = df['symbol'].unique()
print(f"\nUnique symbols ({len(unique_symbols)}):")
for symbol in sorted(unique_symbols):
    print(f"  '{symbol}',")

# الأطر الزمنية المتاحة
unique_timeframes = df['timeframe'].unique()
print(f"\nAvailable timeframes ({len(unique_timeframes)}):")
for tf in sorted(unique_timeframes):
    print(f"  {tf}")