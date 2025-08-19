#!/usr/bin/env python3
"""
🧪 اختبار السيرفر
"""

import requests
import json

# اختبر محلي أولاً
server_urls = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://69.62.121.53:5000"
]

print("🧪 Testing Advanced ML Server...")
print("="*50)

for url in server_urls:
    print(f"\n📡 Testing: {url}")
    try:
        # اختبر /status
        response = requests.get(f"{url}/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is running!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Error: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - server not running")
    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout - server slow or blocked")
    except Exception as e:
        print(f"❌ Error: {e}")

# اختبر طلب تنبؤ
print("\n\n📊 Testing prediction endpoint...")
test_data = {
    "symbol": "EURUSD",
    "timeframe": "M15",
    "candles": [
        {"time": f"2024-01-01 12:{i:02d}:00", 
         "open": 1.0850 + i*0.0001, 
         "high": 1.0851 + i*0.0001,
         "low": 1.0849 + i*0.0001,
         "close": 1.0850 + i*0.0001,
         "volume": 1000}
        for i in range(200)
    ]
}

try:
    response = requests.post(
        "http://localhost:5000/predict",
        json=test_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction received!")
        print(f"   Action: {result.get('action')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   SL: {result.get('sl_pips', 0)} pips")
        print(f"   TP1: {result.get('tp1_pips', 0)} pips")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Prediction test failed: {e}")

print("\n" + "="*50)
print("✅ Test complete!")