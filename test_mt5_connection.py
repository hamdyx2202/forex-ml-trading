#!/usr/bin/env python3
"""
اختبار الاتصال بين MT5 والخادم
"""

import requests
import json
import numpy as np
from datetime import datetime, timedelta

def generate_test_candles(num_candles=200):
    """توليد بيانات شموع اختبارية"""
    candles = []
    base_price = 1.0850
    current_time = int(datetime.now().timestamp()) - (num_candles * 300)  # 5 دقائق لكل شمعة
    
    for i in range(num_candles):
        # حركة عشوائية
        change = np.random.uniform(-0.0010, 0.0010)
        base_price += change
        
        high = base_price + np.random.uniform(0, 0.0005)
        low = base_price - np.random.uniform(0, 0.0005)
        open_price = base_price + np.random.uniform(-0.0002, 0.0002)
        
        candles.append({
            "time": current_time + (i * 300),
            "open": round(open_price, 5),
            "high": round(high, 5),
            "low": round(low, 5),
            "close": round(base_price, 5),
            "volume": np.random.randint(100, 1000)
        })
    
    return candles

def test_prediction_server():
    """اختبار خادم التنبؤات"""
    
    print("🔍 اختبار خادم التنبؤات...")
    
    # 1. فحص صحة الخادم
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            print("✅ الخادم يعمل بشكل صحيح")
            print(f"   • الحالة: {response.json()['status']}")
            print(f"   • النماذج المحملة: {response.json()['models_loaded']}")
        else:
            print("❌ الخادم لا يستجيب بشكل صحيح")
            return
    except Exception as e:
        print(f"❌ لا يمكن الاتصال بالخادم: {e}")
        print("تأكد من تشغيل: python mt5_prediction_server.py")
        return
    
    # 2. اختبار التنبؤ
    print("\n📊 اختبار التنبؤ...")
    
    # بيانات اختبارية
    test_data = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "candles": generate_test_candles(200),
        "account_balance": 10000.0,
        "account_equity": 10000.0,
        "open_positions": 0,
        "strategies": ["scalping", "short_term", "medium_term"]
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/predict_advanced",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ تم استلام التنبؤ بنجاح")
            
            print(f"\n📈 النتائج:")
            print(f"   • الرمز: {result['symbol']}")
            print(f"   • الإطار الزمني: {result['timeframe']}")
            print(f"   • السعر الحالي: {result['current_price']}")
            
            predictions = result.get('predictions', {})
            if predictions:
                print(f"\n🎯 التنبؤات ({len(predictions)} استراتيجية):")
                
                for strategy, pred in predictions.items():
                    print(f"\n   {strategy.upper()}:")
                    signal_text = "شراء" if pred['signal'] == 2 else "بيع" if pred['signal'] == 0 else "محايد"
                    print(f"   • الإشارة: {signal_text}")
                    print(f"   • الثقة: {pred['confidence']:.2%}")
                    print(f"   • Stop Loss: {pred['stop_loss']:.5f}")
                    print(f"   • TP1: {pred['take_profit_1']:.5f}")
                    print(f"   • TP2: {pred['take_profit_2']:.5f}")
                    print(f"   • TP3: {pred['take_profit_3']:.5f}")
            else:
                print("⚠️ لا توجد تنبؤات (ربما جميع الإشارات محايدة)")
        else:
            print(f"❌ خطأ في التنبؤ: {response.status_code}")
            print(f"   التفاصيل: {response.text}")
            
    except Exception as e:
        print(f"❌ خطأ في إرسال الطلب: {e}")
    
    # 3. اختبار إرسال نتيجة صفقة
    print("\n📝 اختبار إرسال نتيجة صفقة...")
    
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
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ تم إرسال نتيجة الصفقة بنجاح")
        else:
            print(f"❌ خطأ في إرسال النتيجة: {response.status_code}")
            
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 4. اختبار الحصول على الأداء
    print("\n📊 اختبار الحصول على الأداء...")
    
    try:
        response = requests.get("http://localhost:5000/api/get_performance")
        
        if response.status_code == 200:
            performance = response.json()
            overall = performance.get('overall', {})
            
            print("✅ إحصائيات الأداء:")
            print(f"   • إجمالي الصفقات: {overall.get('total_trades', 0)}")
            print(f"   • الصفقات الرابحة: {overall.get('winning_trades', 0)}")
            print(f"   • معدل الفوز: {overall.get('win_rate', 0):.1f}%")
            print(f"   • إجمالي الربح: ${overall.get('total_profit', 0):.2f}")
            
            by_strategy = performance.get('by_strategy', [])
            if by_strategy:
                print("\n   حسب الاستراتيجية:")
                for strat in by_strategy:
                    win_rate = (strat['wins'] / strat['trades'] * 100) if strat['trades'] > 0 else 0
                    print(f"   • {strat['strategy']}: {strat['trades']} صفقة، {win_rate:.1f}% فوز")
                    
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    print("\n✅ اكتمل الاختبار!")
    print("\n💡 إذا كانت جميع الاختبارات ناجحة، يمكنك:")
    print("1. تشغيل EA في MT5")
    print("2. التأكد من إضافة http://localhost:5000 في WebRequest")
    print("3. مراقبة Journal في MT5")

if __name__ == "__main__":
    test_prediction_server()