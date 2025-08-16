#!/usr/bin/env python3
"""
التحقق من البيانات المتاحة في قاعدة البيانات - نسخة بسيطة
"""

import sqlite3
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
        cursor = conn.cursor()
        
        # إحصائيات عامة
        cursor.execute("SELECT COUNT(*) FROM price_data")
        total_records = cursor.fetchone()[0]
        print(f"\n📈 إجمالي السجلات: {total_records:,}")
        
        # الأزواج مع أكثر من 10000 سجل
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING count >= 10000
            ORDER BY count DESC
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"\n🎯 أفضل 20 زوج (10,000+ سجل):")
            print("-" * 60)
            print(f"{'Symbol':<15} {'Timeframe':<10} {'Records':<15}")
            print("-" * 60)
            
            for symbol, timeframe, count in results:
                print(f"{symbol:<15} {timeframe:<10} {count:<15,}")
        
        # الأزواج الفريدة
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
        unique_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT DISTINCT symbol FROM price_data LIMIT 10")
        sample_symbols = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📊 الملخص:")
        print(f"   • عدد الأزواج الفريدة: {unique_symbols}")
        print(f"   • عينة من الأزواج: {', '.join(sample_symbols[:5])}")
        
        # فحص بعض الأزواج المحددة
        test_pairs = ['EURUSDm', 'GBPUSDm', 'XAUUSDm', 'BTCUSDm', 'US30m']
        
        print(f"\n🔍 فحص أزواج محددة:")
        for pair in test_pairs:
            cursor.execute(
                "SELECT COUNT(*) FROM price_data WHERE symbol = ?",
                (pair,)
            )
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"   ✅ {pair}: {count:,} سجل")
            else:
                print(f"   ❌ {pair}: لا توجد بيانات")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ خطأ في قراءة البيانات: {e}")

if __name__ == "__main__":
    check_database()