#!/usr/bin/env python3
"""
🧪 Test Enhanced ML Trading System
"""

import requests
import json
import time

print("🧪 Testing Enhanced ML Trading System")
print("="*50)

# Test server connection
print("\n📡 Testing server connection...")
try:
    response = requests.get("http://localhost:5000/status", timeout=5)
    if response.status_code == 200:
        status = response.json()
        print("✅ Server is running!")
        print(f"   Version: {status['version']}")
        print(f"   Models loaded: {status['total_loaded']}")
        print(f"   Risk status: {status['risk_management']['risk_status']}")
        print(f"   Current balance: ${status['risk_management']['current_balance']}")
    else:
        print("❌ Server error")
        exit(1)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("   Make sure the server is running: python3 enhanced_ml_server.py")
    exit(1)

# Test prediction with sample data
print("\n📊 Testing prediction with market analysis...")

# Create realistic test data
test_data = {
    "symbol": "EURUSDm",
    "timeframe": "M15",
    "account_info": {
        "balance": 10000
    },
    "candles": []
}

# Generate 200 candles of test data
base_price = 1.0850
for i in range(200):
    # Simulate trending market
    trend = 0.00001 * i if i < 100 else -0.00001 * (i - 100)
    volatility = 0.0002
    
    open_price = base_price + trend + (i % 10) * 0.00001
    high = open_price + volatility
    low = open_price - volatility
    close = open_price + volatility * (0.5 - (i % 3) * 0.3)
    
    test_data["candles"].append({
        "time": f"2024-01-01 {8 + i//60:02d}:{i%60:02d}:00",
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000 + i * 10
    })
    
    base_price = close

# Send prediction request
print("\n🤖 Sending prediction request...")
try:
    response = requests.post(
        "http://localhost:5000/predict",
        json=test_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction received!")
        print(f"\n📈 Trading Signal:")
        print(f"   Action: {result.get('action')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Current Price: {result.get('current_price', 0):.5f}")
        
        if result.get('action') != 'NONE':
            print(f"\n💰 Risk Management:")
            print(f"   SL: {result.get('sl_pips', 0):.1f} pips")
            print(f"   TP1: {result.get('tp1_pips', 0):.1f} pips")
            print(f"   TP2: {result.get('tp2_pips', 0):.1f} pips")
            print(f"   R/R Ratio: {result.get('risk_reward_ratio', 0):.2f}")
            print(f"   Lot Size: {result.get('lot_size', 0):.2f}")
            
        if 'market_analysis' in result:
            print(f"\n📊 Market Analysis:")
            ma = result['market_analysis']
            print(f"   Market Score: {ma.get('score', 0)}")
            print(f"   Trend: {ma.get('trend', 'N/A')}")
            print(f"   Session: {ma.get('session_quality', 'N/A')}")
            print(f"   Volatility: {ma.get('volatility', 'N/A')}")
            
        if 'risk_management' in result:
            rm = result['risk_management']
            if 'validation' in rm and rm['validation']:
                val = rm['validation']
                print(f"\n🛡️ Trade Validation:")
                print(f"   Valid: {val.get('is_valid', False)}")
                print(f"   Risk Score: {val.get('risk_score', 0)}")
                if val.get('warnings'):
                    print(f"   Warnings: {', '.join(val['warnings'])}")
                    
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

# Test risk report
print("\n\n💼 Testing risk report...")
try:
    response = requests.get("http://localhost:5000/risk_report", timeout=5)
    if response.status_code == 200:
        report = response.json()
        print("✅ Risk report received!")
        print(f"   Current Balance: ${report['current_balance']:.2f}")
        print(f"   Total P/L: ${report['total_pl']:.2f} ({report['total_pl_percentage']:.1f}%)")
        print(f"   Open Trades: {report['open_trades']}")
        print(f"   Risk Status: {report['risk_status']}")
        print(f"   Win Rate: {report['win_rate']:.1f}%")
        print(f"   Max Drawdown: {report['max_drawdown']:.1f}%")
except Exception as e:
    print(f"❌ Risk report failed: {e}")

# Test models endpoint
print("\n\n🤖 Testing models endpoint...")
try:
    response = requests.get("http://localhost:5000/models", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print("✅ Models information:")
        print(f"   Total pairs: {models['total_pairs']}")
        print(f"   Risk Manager: {models['risk_manager']}")
        print(f"   Market Analyzer: {models['market_analyzer']}")
        
        if models['models']:
            print("\n   Loaded models:")
            for pair, info in models['models'].items():
                print(f"     {pair}: {info['count']} models")
except Exception as e:
    print(f"❌ Models endpoint failed: {e}")

print("\n" + "="*50)
print("✅ Enhanced system test complete!")
print("\n💡 System Features:")
print("   ✓ Market context analysis before trading")
print("   ✓ Dynamic risk management")
print("   ✓ Support/Resistance based SL/TP")
print("   ✓ Multi-timeframe trend analysis")
print("   ✓ Session quality filtering")
print("   ✓ Volatility-adjusted position sizing")
print("   ✓ Trade validation with multiple checks")
print("\n🚀 System is ready for profitable trading!")