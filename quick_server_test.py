#!/usr/bin/env python3
"""
اختبار سريع للخادم
"""

import requests
import json
import sys

def test_server(base_url="http://localhost:5000"):
    print(f"🧪 Testing server at: {base_url}")
    print("=" * 50)
    
    # 1. Health check
    print("\n1️⃣ Health Check:")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Test endpoint
    print("\n2️⃣ Test Endpoint:")
    try:
        test_data = {"test": "data", "number": 123}
        r = requests.post(
            f"{base_url}/test",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Get Signal - مثل EA
    print("\n3️⃣ Get Signal (like EA):")
    try:
        # بيانات مثل التي يرسلها EA
        signal_data = {"symbol": "EURUSDm", "price": 1.1000}
        
        print(f"Sending: {json.dumps(signal_data)}")
        
        r = requests.post(
            f"{base_url}/get_signal",
            json=signal_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status: {r.status_code}")
        print(f"Headers: {dict(r.headers)}")
        print(f"Response: {r.text}")
        
        if r.status_code == 200:
            data = r.json()
            print("\n✅ Signal Details:")
            print(f"  Action: {data.get('action')}")
            print(f"  Confidence: {data.get('confidence')}")
            print(f"  SL: {data.get('sl')}")
            print(f"  TP: {data.get('tp')}")
            print(f"  Lot: {data.get('lot')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Get Signal - بيانات مختلفة
    print("\n4️⃣ Get Signal (GBPUSD):")
    try:
        signal_data = {"symbol": "GBPUSDm", "price": 1.2500}
        
        r = requests.post(
            f"{base_url}/get_signal",
            json=signal_data,
            timeout=10
        )
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ {data.get('symbol')} → {data.get('action')} ({data.get('confidence')*100:.0f}%)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")

if __name__ == "__main__":
    # يمكن تمرير URL كمعامل
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "http://localhost:5000"
    
    test_server(url)