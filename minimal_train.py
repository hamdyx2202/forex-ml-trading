#!/usr/bin/env python3
"""
Minimal training script - Works without external dependencies
تدريب أساسي يعمل بدون مكتبات خارجية
"""

import sqlite3
import json
import os
from datetime import datetime

def print_separator():
    print("="*60)

def check_database():
    """التحقق من وجود قاعدة البيانات"""
    db_path = "data/forex_ml.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        print("Please run ForexMLBatchDataSender.mq5 first to load data")
        return False
        
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"📊 Found tables: {[t[0] for t in tables]}")
    
    # Check data count
    try:
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        print(f"📈 Total price records: {count:,}")
        
        # Check symbols
        cursor.execute("SELECT DISTINCT symbol, COUNT(*) as cnt FROM price_data GROUP BY symbol")
        symbols = cursor.fetchall()
        print("\n📊 Data by symbol:")
        for symbol, cnt in symbols:
            print(f"  • {symbol}: {cnt:,} records")
            
        # Check timeframes
        cursor.execute("SELECT DISTINCT timeframe, COUNT(*) as cnt FROM price_data GROUP BY timeframe")
        timeframes = cursor.fetchall()
        print("\n⏱️ Data by timeframe:")
        for tf, cnt in timeframes:
            print(f"  • {tf}: {cnt:,} records")
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        conn.close()
        return False
        
    conn.close()
    return count > 0

def simple_analysis():
    """تحليل بسيط للبيانات"""
    db_path = "data/forex_ml.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n🔍 Performing simple analysis...")
    
    # Get sample data
    cursor.execute("""
        SELECT time, open, high, low, close, volume 
        FROM price_data 
        WHERE symbol = (SELECT symbol FROM price_data LIMIT 1)
        AND timeframe = (SELECT timeframe FROM price_data LIMIT 1)
        ORDER BY time DESC
        LIMIT 100
    """)
    
    data = cursor.fetchall()
    
    if data:
        # Calculate simple statistics
        closes = [row[4] for row in data]
        avg_close = sum(closes) / len(closes)
        max_close = max(closes)
        min_close = min(closes)
        
        print(f"\n📊 Last 100 bars analysis:")
        print(f"  • Average close: {avg_close:.5f}")
        print(f"  • Max close: {max_close:.5f}")
        print(f"  • Min close: {min_close:.5f}")
        print(f"  • Range: {max_close - min_close:.5f}")
        
        # Simple trend detection
        recent_avg = sum(closes[:20]) / 20
        older_avg = sum(closes[80:]) / 20
        
        if recent_avg > older_avg:
            trend = "📈 UPTREND"
        else:
            trend = "📉 DOWNTREND"
            
        print(f"  • Simple trend: {trend}")
        
    conn.close()

def create_simple_model():
    """إنشاء نموذج بسيط جداً"""
    print("\n🤖 Creating simple trading rules...")
    
    # Simple moving average crossover strategy
    model = {
        "type": "simple_ma_crossover",
        "fast_period": 10,
        "slow_period": 30,
        "created_at": datetime.now().isoformat(),
        "rules": {
            "buy": "fast_ma > slow_ma",
            "sell": "fast_ma < slow_ma",
            "risk_per_trade": 0.01
        }
    }
    
    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/simple_model.json", "w") as f:
        json.dump(model, f, indent=2)
        
    print("✅ Simple model created and saved to models/simple_model.json")
    
    return model

def main():
    """البرنامج الرئيسي"""
    print_separator()
    print("🚀 Minimal Forex ML Training")
    print_separator()
    
    # Check database
    if not check_database():
        print("\n⚠️ No data found to train on!")
        print("\nTo load data:")
        print("1. Open MetaTrader 5")
        print("2. Load ForexMLBatchDataSender.mq5 Expert Advisor")
        print("3. Run it on any chart")
        print("4. Wait for data to be sent")
        print("5. Run this script again")
        return
        
    # Analyze data
    simple_analysis()
    
    # Create model
    model = create_simple_model()
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("\nNext steps:")
    print("1. The system is now ready to make predictions")
    print("2. Start the server: python3 src/mt5_bridge_server_linux.py")
    print("3. Use ForexMLBot.mq5 in MT5 to receive signals")
    print("="*60)

if __name__ == "__main__":
    main()