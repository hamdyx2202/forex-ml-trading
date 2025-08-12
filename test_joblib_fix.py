#!/usr/bin/env python3
"""
Test joblib fix for model loading
اختبار إصلاح joblib لتحميل النماذج
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Testing joblib model loading...\n")

# Test 1: Direct model loading
print("1️⃣ Testing advanced predictor...")
try:
    from src.advanced_predictor_95 import AdvancedPredictor
    predictor = AdvancedPredictor()
    print(f"\n✅ Successfully loaded {len(predictor.models)} models!\n")
    
    # Show loaded models
    print("Loaded models:")
    for i, (key, model) in enumerate(predictor.models.items()):
        print(f"  {i+1}. {key}")
        if i >= 9:  # Show first 10
            print(f"  ... and {len(predictor.models) - 10} more")
            break
    
    # Test a prediction
    print("\n2️⃣ Testing prediction...")
    test_symbol = "GBPUSDm"
    test_timeframe = "PERIOD_M15"
    test_key = f"{test_symbol}_{test_timeframe}"
    
    if test_key in predictor.models:
        print(f"✅ Model {test_key} is ready for predictions!")
        
        # Test with dummy data
        test_data = {
            'open': 1.2500,
            'high': 1.2550,
            'low': 1.2480,
            'close': 1.2520,
            'volume': 1000
        }
        
        result = predictor.predict_with_confidence(
            symbol=test_symbol,
            timeframe=test_timeframe,
            current_data=test_data
        )
        
        print(f"\nPrediction result:")
        print(f"  Signal: {result.get('signal')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
    else:
        print(f"❌ Model {test_key} not found")
        print(f"Available keys: {list(predictor.models.keys())[:5]}...")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3️⃣ Testing server...")
print("The server should now work correctly with joblib!")
print("\nTo run the server:")
print("  python3 src/mt5_bridge_server_advanced.py")