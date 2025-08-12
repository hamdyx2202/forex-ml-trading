#!/usr/bin/env python3
"""
Debug model key generation
تصحيح توليد مفاتيح النماذج
"""

# Test cases
test_symbols = ["GBPUSDm", "EURUSDm", "XAUUSDm"]
test_timeframes = ["M5", "M15", "H1", "H4"]

print("🔍 Testing model key generation:\n")

# Correct timeframe mapping
timeframe_map = {
    'M5': 'PERIOD_M5',
    'M15': 'PERIOD_M15',
    'H1': 'PERIOD_H1',
    'H4': 'PERIOD_H4'
}

print("✅ CORRECT way (with underscore):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        model_key = f"{symbol}_{model_timeframe}"
        print(f"  {symbol} + {tf} → {model_key}")

print("\n❌ WRONG way (with space):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        # This creates the wrong key with space
        wrong_key = f"{symbol} {model_timeframe}"
        print(f"  {symbol} + {tf} → {wrong_key}")

print("\n❌ WRONG way (without m):")
for symbol in test_symbols:
    for tf in test_timeframes:
        model_timeframe = timeframe_map[tf]
        # This removes the m suffix
        wrong_key = f"{symbol.rstrip('m')}_{model_timeframe}"
        print(f"  {symbol} + {tf} → {wrong_key}")

print("\n📝 The server MUST use the first format (with underscore)!")
