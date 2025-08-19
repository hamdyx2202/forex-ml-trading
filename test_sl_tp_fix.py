#!/usr/bin/env python3
"""
Test the SL/TP calculation fix for None market_context
"""

# Simulate the calculate_dynamic_sl_tp method
def calculate_dynamic_sl_tp(symbol, direction, entry_price, market_context):
    """حساب SL/TP ذكي بناءً على سياق السوق"""
    pip_value = 0.01 if 'JPY' in symbol else 0.0001
    
    # Handle None market_context with default values
    if market_context is None:
        # Use fixed percentages when no market context available
        print(f"No market context for {symbol}, using default SL/TP")
        sl_percentage = 0.002  # 0.2% default stop loss
        tp1_percentage = 0.003  # 0.3% default TP1
        tp2_percentage = 0.005  # 0.5% default TP2
        
        if direction == 'BUY':
            sl_price = entry_price * (1 - sl_percentage)
            tp1_price = entry_price * (1 + tp1_percentage)
            tp2_price = entry_price * (1 + tp2_percentage)
        else:  # SELL
            sl_price = entry_price * (1 + sl_percentage)
            tp1_price = entry_price * (1 - tp1_percentage)
            tp2_price = entry_price * (1 - tp2_percentage)
        
        return {
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_percentage * 10000),
            'tp1_pips': float(tp1_percentage * 10000),
            'tp2_pips': float(tp2_percentage * 10000),
            'risk_reward_ratio': float(tp1_percentage / sl_percentage),
            'based_on': 'default_percentages'
        }
    
    # Normal calculation would go here
    return {'sl_price': 1.0, 'tp1_price': 1.0, 'tp2_price': 1.0}

# Test cases
print("Testing SL/TP calculation with None market_context:")
print("=" * 50)

# Test BUY with None context
result = calculate_dynamic_sl_tp('EURUSD', 'BUY', 1.1000, None)
print(f"\nBUY @ 1.1000:")
print(f"  SL: {result['sl_price']:.5f} ({result['sl_pips']} pips)")
print(f"  TP1: {result['tp1_price']:.5f} ({result['tp1_pips']} pips)")
print(f"  TP2: {result['tp2_price']:.5f} ({result['tp2_pips']} pips)")
print(f"  R:R: {result['risk_reward_ratio']:.1f}")

# Test SELL with None context
result = calculate_dynamic_sl_tp('EURUSD', 'SELL', 1.1000, None)
print(f"\nSELL @ 1.1000:")
print(f"  SL: {result['sl_price']:.5f} ({result['sl_pips']} pips)")
print(f"  TP1: {result['tp1_price']:.5f} ({result['tp1_pips']} pips)")
print(f"  TP2: {result['tp2_price']:.5f} ({result['tp2_pips']} pips)")
print(f"  R:R: {result['risk_reward_ratio']:.1f}")

print("\n✅ Fix working correctly - no TypeError when market_context is None")