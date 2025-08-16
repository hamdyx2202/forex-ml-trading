#!/usr/bin/env python3
"""
ملء قاعدة البيانات ببيانات تدريب - نسخة بسيطة
"""

import sqlite3
from pathlib import Path
import random
from datetime import datetime, timedelta

def populate_data():
    """إضافة بيانات تدريب"""
    
    db_path = Path("data/forex_ml.db")
    
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # الأزواج المطلوبة
    required_pairs = [
        ('EURUSD', 1.0850),
        ('GBPUSD', 1.2650),
        ('USDJPY', 150.50),
        ('XAUUSD', 2050.0),
        ('BTCUSD', 45000.0)
    ]
    
    timeframes = ['M5', 'M15', 'H1']
    
    # فترات زمنية بالثواني
    timeframe_seconds = {
        'M5': 300,
        'M15': 900,
        'H1': 3600
    }
    
    for symbol, base_price in required_pairs:
        for timeframe in timeframes:
            # فحص البيانات الموجودة
            cursor.execute(
                "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe)
            )
            existing_count = cursor.fetchone()[0]
            
            if existing_count >= 10000:
                print(f"✅ {symbol} {timeframe} لديه بيانات كافية ({existing_count:,})")
                continue
            
            records_needed = 10000 - existing_count
            print(f"📈 إضافة {records_needed:,} سجل لـ {symbol} {timeframe}...")
            
            # توليد البيانات
            interval = timeframe_seconds[timeframe]
            current_time = int(datetime.now().timestamp()) - (records_needed * interval)
            current_price = base_price
            
            data_batch = []
            
            for i in range(records_needed):
                # حركة السعر العشوائية
                volatility = 0.001
                if 'JPY' in symbol:
                    volatility = 0.1
                elif 'XAU' in symbol:
                    volatility = 1.0
                elif 'BTC' in symbol:
                    volatility = 100.0
                
                # تغيير السعر
                change = random.uniform(-volatility, volatility)
                current_price += change
                
                # OHLC
                high = current_price + random.uniform(0, volatility)
                low = current_price - random.uniform(0, volatility)
                open_price = current_price + random.uniform(-volatility/2, volatility/2)
                
                # الحجم
                volume = random.randint(100, 10000)
                
                data_batch.append((
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
                
                current_time += interval
                
                # إدراج على دفعات
                if len(data_batch) >= 1000:
                    cursor.executemany(
                        """INSERT INTO price_data 
                           (symbol, timeframe, time, open, high, low, close, volume, spread)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        data_batch
                    )
                    data_batch = []
                    conn.commit()
            
            # إدراج البقية
            if data_batch:
                cursor.executemany(
                    """INSERT INTO price_data 
                       (symbol, timeframe, time, open, high, low, close, volume, spread)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    data_batch
                )
                conn.commit()
            
            print(f"✅ تم إضافة البيانات لـ {symbol} {timeframe}")
    
    # عرض الملخص النهائي
    print("\n📊 البيانات النهائية:")
    print("-" * 60)
    
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as count
        FROM price_data
        GROUP BY symbol, timeframe
        HAVING count >= 10000
        ORDER BY symbol, timeframe
    """)
    
    results = cursor.fetchall()
    
    print(f"{'Symbol':<15} {'Timeframe':<10} {'Records':<15}")
    print("-" * 60)
    
    for symbol, timeframe, count in results:
        print(f"{symbol:<15} {timeframe:<10} {count:<15,}")
    
    conn.close()
    
    print(f"\n✅ إجمالي الأزواج الجاهزة للتدريب: {len(results)}")
    print("\n🚀 يمكنك الآن تشغيل: python train_full_advanced.py")

if __name__ == "__main__":
    print("🔧 إضافة بيانات التدريب...")
    print("=" * 60)
    populate_data()