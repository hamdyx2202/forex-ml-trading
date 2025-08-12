#!/usr/bin/env python3
"""
Test model matching after all fixes
اختبار مطابقة النماذج بعد جميع الإصلاحات
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Testing model matching...\n")

# Test 1: Load models and show their names
print("1️⃣ Loading models...")
try:
    from src.advanced_predictor_95 import AdvancedPredictor
    predictor = AdvancedPredictor()
    print(f"✅ Loaded {len(predictor.models)} models\n")
    
    print("Model names (first 10):")
    for i, key in enumerate(list(predictor.models.keys())[:10]):
        print(f"  {i+1}. {key}")
    print()
except Exception as e:
    print(f"❌ Error loading models: {e}\n")
    predictor = None

# Test 2: Test model key generation
print("2️⃣ Testing model key generation...")
test_cases = [
    ("GBPUSDm", "M15", "PERIOD_M15"),
    ("EURUSDm", "M5", "PERIOD_M5"),
    ("XAUUSDm", "H1", "PERIOD_H1"),
]

for symbol, timeframe, model_timeframe in test_cases:
    # This is how the server should generate keys
    model_key = f"{symbol}_{model_timeframe}"
    print(f"\nSymbol: {symbol}, Timeframe: {timeframe}")
    print(f"  Generated key: {model_key}")
    
    if predictor:
        exists = model_key in predictor.models
        print(f"  Model exists: {'✅ YES' if exists else '❌ NO'}")
        
        # Check wrong formats
        wrong_key1 = f"{symbol} {model_timeframe}"  # With space
        wrong_key2 = f"{symbol.rstrip('m')}_{model_timeframe}"  # Without m
        
        if wrong_key1 in predictor.models:
            print(f"  ⚠️ WARNING: Model with space exists: {wrong_key1}")
        if wrong_key2 in predictor.models:
            print(f"  ⚠️ WARNING: Model without 'm' exists: {wrong_key2}")

# Test 3: Simulate server behavior
print("\n3️⃣ Simulating server model lookup...")
if predictor:
    # Simulate what the server does
    symbol = "GBPUSDm"
    timeframe = "M15"
    
    # Map timeframe
    timeframe_map = {
        'M5': 'PERIOD_M5',
        'M15': 'PERIOD_M15',
        'H1': 'PERIOD_H1',
        'H4': 'PERIOD_H4'
    }
    
    model_timeframe = timeframe_map.get(timeframe, timeframe)
    model_key = f"{symbol}_{model_timeframe}"
    
    print(f"\nServer receives: symbol={symbol}, timeframe={timeframe}")
    print(f"Maps to: model_timeframe={model_timeframe}")
    print(f"Generates key: {model_key}")
    print(f"Model exists: {'✅ YES' if model_key in predictor.models else '❌ NO'}")

print("\n✅ Test complete!")
print("\nIf models are not found, check that:")
print("1. Model names match exactly (with 'm' suffix)")
print("2. No spaces in model keys")
print("3. Correct format: SYMBOL_PERIOD_TIMEFRAME")