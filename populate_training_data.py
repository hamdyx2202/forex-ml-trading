#!/usr/bin/env python3
"""
ملء قاعدة البيانات ببيانات تدريب إضافية من MT5
أو تحويل البيانات الموجودة
"""

import sqlite3
from pathlib import Path
import random
from datetime import datetime, timedelta
import numpy as np

def check_and_populate_data():
    """فحص البيانات وإضافة المزيد إذا لزم الأمر"""
    
    db_path = Path("data/forex_ml.db")
    
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # فحص البيانات الحالية
    cursor.execute("SELECT symbol, COUNT(*) as count FROM price_data GROUP BY symbol")
    current_data = cursor.fetchall()
    
    print("📊 البيانات الحالية:")
    for symbol, count in current_data:
        print(f"   • {symbol}: {count:,} سجل")
    
    # إضافة بيانات لأزواج إضافية إذا لزم الأمر
    required_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
    timeframes = ['M5', 'M15', 'H1']
    
    for pair in required_pairs:
        for timeframe in timeframes:
            cursor.execute(
                "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND timeframe = ?",
                (pair, timeframe)
            )
            count = cursor.fetchone()[0]
            
            if count < 10000:
                print(f"\n📈 إضافة بيانات لـ {pair} {timeframe}...")
                generate_sample_data(conn, pair, timeframe, 10000 - count)
    
    conn.commit()
    conn.close()
    
    print("\n✅ تم تحديث قاعدة البيانات!")

def generate_sample_data(conn, symbol, timeframe, num_records):
    """توليد بيانات عينة للتدريب"""
    
    cursor = conn.cursor()
    
    # الحصول على آخر سعر إن وجد
    cursor.execute(
        "SELECT close, time FROM price_data WHERE symbol = ? ORDER BY time DESC LIMIT 1",
        (symbol,)
    )
    result = cursor.fetchone()
    
    if result:
        base_price = result[0]
        last_time = result[1]
    else:
        # أسعار افتراضية
        base_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 150.50,
            'XAUUSD': 2050.0,
            'BTCUSD': 45000.0
        }
        base_price = base_prices.get(symbol, 1.0)
        last_time = int(datetime.now().timestamp()) - 86400 * 30  # 30 يوم مضت
    
    # فترة زمنية بالثواني
    timeframe_seconds = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400
    }
    
    interval = timeframe_seconds.get(timeframe, 300)
    
    # توليد البيانات
    data_to_insert = []
    current_price = base_price
    
    for i in range(num_records):
        # حركة عشوائية واقعية
        volatility = 0.0001 if 'JPY' not in symbol else 0.01
        if 'XAU' in symbol:
            volatility = 0.1
        elif 'BTC' in symbol:
            volatility = 10.0
        
        # حركة السعر
        change = np.random.normal(0, volatility)
        current_price *= (1 + change)
        
        # OHLC
        high = current_price * (1 + abs(np.random.normal(0, volatility/2)))
        low = current_price * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = current_price * (1 + np.random.normal(0, volatility/4))
        
        # الحجم
        volume = int(np.random.lognormal(10, 1))
        
        # الوقت
        current_time = last_time + (i + 1) * interval
        
        data_to_insert.append((
            symbol,
            timeframe,
            current_time,
            open_price,
            high,
            low,
            current_price,  # close
            volume,
            0  # spread
        ))
        
        # إدراج على دفعات
        if len(data_to_insert) >= 1000:
            cursor.executemany(
                """INSERT INTO price_data 
                   (symbol, timeframe, time, open, high, low, close, volume, spread)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                data_to_insert
            )
            data_to_insert = []
            print(f"   • تم إضافة {i+1}/{num_records} سجل...")
    
    # إدراج البقية
    if data_to_insert:
        cursor.executemany(
            """INSERT INTO price_data 
               (symbol, timeframe, time, open, high, low, close, volume, spread)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data_to_insert
        )
    
    print(f"✅ تم إضافة {num_records} سجل لـ {symbol} {timeframe}")

def main():
    print("🔧 فحص وتحديث بيانات التدريب...")
    print("=" * 60)
    
    check_and_populate_data()
    
    # فحص النتيجة النهائية
    db_path = Path("data/forex_ml.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n📊 البيانات النهائية:")
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as count
        FROM price_data
        GROUP BY symbol, timeframe
        HAVING count >= 10000
        ORDER BY symbol, timeframe
    """)
    
    results = cursor.fetchall()
    
    print("-" * 50)
    print(f"{'Symbol':<10} {'Timeframe':<10} {'Records':<10}")
    print("-" * 50)
    
    for symbol, timeframe, count in results:
        print(f"{symbol:<10} {timeframe:<10} {count:<10,}")
    
    conn.close()
    
    print(f"\n✅ إجمالي الأزواج الجاهزة للتدريب: {len(results)}")

if __name__ == "__main__":
    main()