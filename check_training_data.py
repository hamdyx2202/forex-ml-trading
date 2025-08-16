#!/usr/bin/env python3
"""
التحقق من البيانات المتاحة في قاعدة البيانات الصحيحة
"""

import sqlite3
import pandas as pd
from pathlib import Path

def check_database():
    db_path = Path("data/forex_ml.db")
    
    if not db_path.exists():
        print(f"❌ قاعدة البيانات غير موجودة: {db_path}")
        return
    
    print(f"✅ قاعدة البيانات موجودة: {db_path}")
    print(f"📊 الحجم: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # إحصائيات عامة
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_data")
        total_records = cursor.fetchone()[0]
        print(f"\n📈 إجمالي السجلات: {total_records:,}")
        
        # إحصائيات حسب الرمز والإطار الزمني
        query = """
            SELECT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING count >= 10000
            ORDER BY count DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        print(f"\n🎯 الأزواج مع بيانات كافية (10,000+ سجل):")
        print("-" * 60)
        print(f"{'Symbol':<15} {'Timeframe':<10} {'Records':<15}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            print(f"{row['symbol']:<15} {row['timeframe']:<10} {row['count']:<15,}")
        
        # ملخص
        unique_symbols = df['symbol'].nunique()
        unique_timeframes = df['timeframe'].nunique()
        
        print(f"\n📊 الملخص:")
        print(f"   • عدد الأزواج الفريدة: {unique_symbols}")
        print(f"   • عدد الأطر الزمنية: {unique_timeframes}")
        print(f"   • إجمالي المجموعات: {len(df)}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ خطأ في قراءة البيانات: {e}")

if __name__ == "__main__":
    check_database()
