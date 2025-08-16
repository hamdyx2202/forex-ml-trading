#!/usr/bin/env python3
"""
اختبار بسيط للاتصال بين MT5 والخادم
Simple test without heavy dependencies
"""

import requests
import json
from datetime import datetime
import random

def generate_test_candles(num_candles=200):
    """توليد بيانات شموع اختبارية"""
    candles = []
    base_price = 1.0850
    current_time = int(datetime.now().timestamp()) - (num_candles * 300)  # 5 دقائق لكل شمعة
    
    for i in range(num_candles):
        # حركة عشوائية
        change = random.uniform(-0.0010, 0.0010)
        base_price += change
        
        high = base_price + random.uniform(0, 0.0005)
        low = base_price - random.uniform(0, 0.0005)
        open_price = base_price + random.uniform(-0.0002, 0.0002)
        
        candles.append({
            "time": current_time + (i * 300),
            "open": round(open_price, 5),
            "high": round(high, 5),
            "low": round(low, 5),
            "close": round(base_price, 5),
            "volume": random.randint(100, 1000)
        })
    
    return candles

def test_server():
    """اختبار الخادم"""
    
    print("🔍 Testing MT5 Prediction Server...")
    print("=" * 50)
    
    # 1. فحص صحة الخادم
    print("\n1️⃣ Health Check:")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"   • Status: {data.get('status')}")
            print(f"   • Version: {data.get('version')}")
            print(f"   • Message: {data.get('message', '')}")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server at http://localhost:5000")
        print("   Please run: python3 start_mt5_server_simple.py")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. اختبار التنبؤ
    print("\n2️⃣ Prediction Test:")
    
    test_data = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "candles": generate_test_candles(200),
        "account_balance": 10000.0,
        "account_equity": 10000.0,
        "open_positions": 0,
        "strategies": ["scalping", "short_term"]
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/predict_advanced",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Received prediction successfully")
            print(f"\n📊 Results:")
            print(f"   • Symbol: {result.get('symbol')}")
            print(f"   • Timeframe: {result.get('timeframe')}")
            print(f"   • Current Price: {result.get('current_price')}")
            
            predictions = result.get('predictions', {})
            if predictions:
                print(f"\n🎯 Predictions ({len(predictions)} strategies):")
                for strategy, pred in predictions.items():
                    print(f"\n   {strategy.upper()}:")
                    signal_text = "BUY" if pred['signal'] == 2 else "SELL" if pred['signal'] == 0 else "HOLD"
                    print(f"   • Signal: {signal_text}")
                    print(f"   • Confidence: {pred['confidence']:.2%}")
                    print(f"   • Stop Loss: {pred['stop_loss']:.5f}")
                    print(f"   • TP1: {pred['take_profit_1']:.5f}")
                    print(f"   • TP2: {pred['take_profit_2']:.5f}")
                    print(f"   • TP3: {pred['take_profit_3']:.5f}")
        else:
            print(f"❌ Prediction error: {response.status_code}")
            print(f"   Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. اختبار إرسال نتيجة صفقة
    print("\n3️⃣ Trade Result Test:")
    
    trade_result = {
        "ticket": 12345678,
        "symbol": "EURUSD",
        "timeframe": "M5",
        "strategy": "scalping",
        "entry_price": 1.0850,
        "exit_price": 1.0870,
        "stop_loss": 1.0830,
        "take_profit": 1.0870,
        "profit": 25.50,
        "profit_pips": 20,
        "duration_minutes": 45,
        "result": "win"
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/update_trade_result",
            json=trade_result,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Trade result sent successfully")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. اختبار الأداء
    print("\n4️⃣ Performance Test:")
    
    try:
        response = requests.get("http://localhost:5000/api/get_performance", timeout=5)
        
        if response.status_code == 200:
            performance = response.json()
            overall = performance.get('overall', {})
            
            print("✅ Performance stats:")
            print(f"   • Total trades: {overall.get('total_trades', 0)}")
            print(f"   • Winning trades: {overall.get('winning_trades', 0)}")
            print(f"   • Win rate: {overall.get('win_rate', 0):.1f}%")
            print(f"   • Total profit: ${overall.get('total_profit', 0):.2f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\n💡 Next steps:")
    print("1. Make sure the server is running")
    print("2. Install ForexMLBot_Advanced_V2.mq5 in MT5")
    print("3. Add http://localhost:5000 to WebRequest URLs")
    print("4. Monitor the EA Journal in MT5")

if __name__ == "__main__":
    test_server()