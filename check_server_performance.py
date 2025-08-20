#!/usr/bin/env python3
"""
فحص أداء السيرفر وسرعة الاستجابة
"""

import time
import requests
import statistics

def test_server_response(url="http://69.62.121.53:5000/predict", iterations=10):
    """اختبار سرعة استجابة السيرفر"""
    
    # بيانات اختبارية
    test_data = {
        "symbol": "EURUSD",
        "timeframe": "M15",
        "candles": [
            {
                "time": "2024-01-01 12:00:00",
                "open": 1.1000,
                "high": 1.1010,
                "low": 1.0990,
                "close": 1.1005,
                "tick_volume": 100,
                "spread": 2,
                "real_volume": 0
            }
        ] * 200  # 200 شمعة
    }
    
    response_times = []
    errors = 0
    
    print(f"🔍 Testing server response time ({iterations} requests)...")
    print("=" * 50)
    
    for i in range(iterations):
        try:
            start_time = time.time()
            response = requests.post(url, json=test_data, timeout=30)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # بالميلي ثانية
            response_times.append(response_time)
            
            status = "✅" if response.status_code == 200 else "❌"
            print(f"Request {i+1}: {status} {response_time:.1f}ms")
            
            if response.status_code != 200:
                errors += 1
                
        except requests.exceptions.Timeout:
            print(f"Request {i+1}: ⏱️ TIMEOUT (>30s)")
            errors += 1
        except Exception as e:
            print(f"Request {i+1}: ❌ ERROR: {e}")
            errors += 1
        
        time.sleep(1)  # تأخير ثانية بين الطلبات
    
    print("\n" + "=" * 50)
    print("📊 Results:")
    
    if response_times:
        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"✅ Successful: {len(response_times)}/{iterations}")
        print(f"❌ Failed: {errors}/{iterations}")
        print(f"⏱️ Average: {avg_time:.1f}ms")
        print(f"⚡ Fastest: {min_time:.1f}ms")
        print(f"🐌 Slowest: {max_time:.1f}ms")
        
        if avg_time > 5000:  # أكثر من 5 ثواني
            print("\n⚠️ WARNING: Server response is very slow!")
            print("This might cause timeouts in MT5")
        elif avg_time > 2000:  # أكثر من ثانيتين
            print("\n⚠️ Server response is slow")
        else:
            print("\n✅ Server response time is good")
    else:
        print("❌ All requests failed!")

if __name__ == "__main__":
    test_server_response()