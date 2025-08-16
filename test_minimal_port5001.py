#!/usr/bin/env python3
"""
اختبار بسيط للخادم على المنفذ 5001
Test for server on port 5001
"""

import urllib.request
import urllib.error
import json
from datetime import datetime
import random

def test_server():
    """Test the minimal server on port 5001"""
    base_url = "http://localhost:5001"
    
    print("🔍 Testing MT5 Server on Port 5001...")
    print("=" * 50)
    
    # 1. Health Check
    print("\n1️⃣ Health Check:")
    try:
        response = urllib.request.urlopen(f"{base_url}/api/health", timeout=5)
        data = json.loads(response.read().decode())
        print("✅ Server is healthy")
        print(f"   • Status: {data.get('status')}")
        print(f"   • Version: {data.get('version')}")
        print(f"   • Message: {data.get('message')}")
    except urllib.error.URLError as e:
        print("❌ Cannot connect to server on port 5001")
        print("   Please run: python3 mt5_server_port5001.py")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Prediction Test
    print("\n2️⃣ Prediction Test:")
    
    # Generate test candles
    candles = []
    base_price = 1.0850
    for i in range(200):
        candles.append({
            "time": int(datetime.now().timestamp()) - (200-i)*300,
            "open": round(base_price + random.uniform(-0.001, 0.001), 5),
            "high": round(base_price + random.uniform(0, 0.002), 5),
            "low": round(base_price - random.uniform(0, 0.002), 5),
            "close": round(base_price + random.uniform(-0.001, 0.001), 5),
            "volume": random.randint(100, 1000)
        })
        base_price += random.uniform(-0.0005, 0.0005)
    
    test_data = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "candles": candles,
        "strategies": ["scalping", "short_term"]
    }
    
    try:
        req = urllib.request.Request(
            f"{base_url}/api/predict_advanced",
            data=json.dumps(test_data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode())
        
        print("✅ Received prediction")
        print(f"   • Symbol: {result.get('symbol')}")
        print(f"   • Current Price: {result.get('current_price')}")
        
        predictions = result.get('predictions', {})
        for strategy, pred in predictions.items():
            signal_text = "BUY" if pred['signal'] == 2 else "SELL"
            print(f"\n   {strategy.upper()}:")
            print(f"   • Signal: {signal_text}")
            print(f"   • Confidence: {pred['confidence']:.1%}")
            print(f"   • SL: {pred['stop_loss']}")
            print(f"   • TP1: {pred['take_profit_1']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\n💡 Don't forget to update EA to use port 5001:")
    print("   InpServerURL = \"http://localhost:5001/api/predict_advanced\"")

if __name__ == "__main__":
    test_server()