#!/usr/bin/env python3
"""
Test script for historical data endpoint
اختبار endpoint البيانات التاريخية
"""

import requests
import json
from datetime import datetime

def test_historical_endpoint(url="http://localhost:5000"):
    """Test the historical data endpoint with MT5-like data"""
    
    # Sample data similar to what MT5 sends
    test_data = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "data": [
            {
                "time": "2025.08.15 12:00",
                "open": 1.09850,
                "high": 1.09880,
                "low": 1.09840,
                "close": 1.09870,
                "volume": 1250,
                "spread": 10
            },
            {
                "time": "2025.08.15 12:05",
                "open": 1.09870,
                "high": 1.09890,
                "low": 1.09860,
                "close": 1.09885,
                "volume": 1100,
                "spread": 12
            }
        ]
    }
    
    print(f"🔍 Testing {url}/api/historical_data")
    print(f"📊 Sending data for {test_data['symbol']} {test_data['timeframe']}")
    print(f"📈 Bars count: {len(test_data['data'])}")
    
    try:
        # Test the debug endpoint first
        debug_response = requests.post(
            f"{url}/api/debug/historical",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        print("\n🐛 Debug response:")
        print(json.dumps(debug_response.json(), indent=2))
        
        # Test the actual endpoint
        response = requests.post(
            f"{url}/api/historical_data",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\n📡 Response status: {response.status_code}")
        print(f"📦 Response data:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\n✅ Success! Historical data endpoint is working correctly")
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection failed! Make sure the server is running on {url}")
        print("   Run: python main_linux.py server")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_historical_endpoint(url)