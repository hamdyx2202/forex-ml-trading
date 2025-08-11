#!/usr/bin/env python3
"""
اختبار خادم الجسر MT5
"""

import requests
import json
from datetime import datetime
import sys
import time

def test_server(server_url="http://localhost:5000"):
    """اختبار جميع نقاط النهاية للخادم"""
    
    print(f"🧪 اختبار خادم الجسر: {server_url}")
    print("=" * 50)
    
    # 1. اختبار الصحة
    print("\n1️⃣ اختبار الصحة...")
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            print("✅ الخادم يعمل بشكل جيد")
            print(f"   Response: {response.json()}")
        else:
            print("❌ فشل اختبار الصحة")
    except Exception as e:
        print(f"❌ خطأ في الاتصال: {e}")
        return
    
    # 2. اختبار الحصول على إشارة
    print("\n2️⃣ اختبار الحصول على إشارة...")
    
    # إرسال عدة أسعار لبناء تاريخ
    print("   إرسال بيانات أسعار متعددة...")
    prices = [1.0850, 1.0852, 1.0848, 1.0851, 1.0853, 
              1.0855, 1.0854, 1.0856, 1.0858, 1.0860,
              1.0862, 1.0865, 1.0863, 1.0864, 1.0866,
              1.0868, 1.0870, 1.0869, 1.0871, 1.0872]
    
    for i, price in enumerate(prices):
        signal_data = {
            "symbol": "EURUSD",
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        
        if i == len(prices) - 1:  # آخر سعر فقط
            break
        else:
            # إرسال الأسعار لبناء التاريخ
            requests.post(
                f"{server_url}/get_signal",
                json=signal_data,
                headers={'Content-Type': 'application/json'}
            )
            time.sleep(0.1)  # تأخير صغير
    
    # الآن اختبار الإشارة الأخيرة
    signal_data = {
        "symbol": "EURUSD",
        "price": prices[-1],
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            f"{server_url}/get_signal",
            json=signal_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            signal = response.json()
            print("✅ تم الحصول على الإشارة")
            print(f"   Action: {signal.get('action')}")
            print(f"   Confidence: {signal.get('confidence', 0):.1%}")
            print(f"   SL: {signal.get('sl')}")
            print(f"   TP: {signal.get('tp')}")
            print(f"   Lot: {signal.get('lot')}")
        else:
            print(f"❌ فشل الحصول على الإشارة: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 3. اختبار تأكيد الصفقة
    print("\n3️⃣ اختبار تأكيد الصفقة...")
    trade_data = {
        "symbol": "EURUSD",
        "action": "BUY",
        "lot": 0.01,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            f"{server_url}/confirm_trade",
            json=trade_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ تم تأكيد الصفقة")
            print(f"   Status: {result.get('status')}")
            print(f"   Trade ID: {result.get('trade_id')}")
        else:
            print(f"❌ فشل تأكيد الصفقة: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 4. اختبار تقرير النتيجة
    print("\n4️⃣ اختبار تقرير نتيجة الصفقة...")
    result_data = {
        "symbol": "EURUSD",
        "volume": 0.01,
        "profit": 10.50,
        "price": 1.0860,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            f"{server_url}/report_trade",
            json=result_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ تم استلام تقرير النتيجة")
            print(f"   Status: {result.get('status')}")
            print(f"   Analysis: {result.get('analysis')}")
        else:
            print(f"❌ فشل تقرير النتيجة: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 5. اختبار حالة النظام
    print("\n5️⃣ اختبار حالة النظام...")
    try:
        response = requests.get(f"{server_url}/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ تم الحصول على حالة النظام")
            print(f"   Server Time: {status.get('server_time')}")
            print(f"   Active Trades: {status.get('active_trades')}")
            print(f"   System Health: {status.get('system_health')}")
        else:
            print(f"❌ فشل الحصول على الحالة: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    print("\n" + "=" * 50)
    print("✅ اكتمل الاختبار")

if __name__ == "__main__":
    # يمكن تمرير عنوان الخادم كمعامل
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_server(server_url)