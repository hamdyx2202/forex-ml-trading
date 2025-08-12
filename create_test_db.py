#!/usr/bin/env python3
"""
Create test database with sample data
إنشاء قاعدة بيانات اختبارية مع بيانات تجريبية
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random

def create_test_database():
    """إنشاء قاعدة بيانات اختبارية"""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect("data/forex_ml.db")
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
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
    
    print("✅ Created database structure")
    
    # Generate test data
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframes = ["M5", "M15", "H1"]
    
    base_time = datetime.now() - timedelta(days=30)
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"📊 Generating data for {symbol} {timeframe}...")
            
            # Starting price
            if symbol == "EURUSD":
                price = 1.0900
            elif symbol == "GBPUSD":
                price = 1.2700
            else:  # USDJPY
                price = 145.00
                
            # Time intervals
            if timeframe == "M5":
                interval = timedelta(minutes=5)
                bars = 2000
            elif timeframe == "M15":
                interval = timedelta(minutes=15)
                bars = 1000
            else:  # H1
                interval = timedelta(hours=1)
                bars = 500
                
            current_time = base_time
            
            for i in range(bars):
                # Generate OHLC
                change = random.uniform(-0.0010, 0.0010)
                open_price = price
                close_price = price + change
                high_price = max(open_price, close_price) + random.uniform(0, 0.0005)
                low_price = min(open_price, close_price) - random.uniform(0, 0.0005)
                volume = random.randint(100, 5000)
                spread = random.randint(1, 5)
                
                # Insert data
                try:
                    cursor.execute("""
                        INSERT INTO price_data 
                        (symbol, timeframe, time, open, high, low, close, volume, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timeframe, int(current_time.timestamp()),
                        open_price, high_price, low_price, close_price,
                        volume, spread
                    ))
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates
                    
                # Update for next bar
                price = close_price
                current_time += interval
                
    conn.commit()
    
    # Show summary
    cursor.execute("SELECT COUNT(*) FROM price_data")
    total = cursor.fetchone()[0]
    
    print(f"\n✅ Database created successfully!")
    print(f"📊 Total records: {total:,}")
    
    # Show data by symbol
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as cnt 
        FROM price_data 
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    
    print("\n📈 Data summary:")
    for row in cursor.fetchall():
        print(f"  • {row[0]} {row[1]}: {row[2]:,} bars")
        
    conn.close()
    
    print("\n✅ Test database ready for training!")

if __name__ == "__main__":
    create_test_database()