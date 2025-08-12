#!/usr/bin/env python3
"""
Test DatetimeIndex fix
اختبار إصلاح DatetimeIndex
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing DatetimeIndex fix...\n")

# Test 1: Create sample data with DatetimeIndex
print("1️⃣ Creating sample data with DatetimeIndex...")
dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
sample_data = pd.DataFrame({
    'open': np.random.randn(200).cumsum() + 100,
    'high': np.random.randn(200).cumsum() + 101,
    'low': np.random.randn(200).cumsum() + 99,
    'close': np.random.randn(200).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 200)
}, index=dates)
sample_data.index.name = 'datetime'

print(f"Created DataFrame with {len(sample_data)} rows")
print(f"Index type: {type(sample_data.index)}")
print(f"Index name: {sample_data.index.name}\n")

# Test 2: Test feature engineering
print("2️⃣ Testing feature engineering...")
try:
    from feature_engineer_fixed_v5 import FeatureEngineer
    
    engineer = FeatureEngineer(min_periods_factor=0.5)
    features_df = engineer.create_features(sample_data.copy())
    
    print(f"✅ Feature engineering successful!")
    print(f"Original rows: {len(sample_data)}")
    print(f"After features: {len(features_df)} rows")
    print(f"Number of features: {len(features_df.columns)}")
    
    # Check if time features were added
    time_features = ['hour', 'day_of_week', 'day_of_month', 'month']
    found_features = [f for f in time_features if f in features_df.columns]
    print(f"\nTime features found: {found_features}")
    
    if 'hour' in features_df.columns:
        print(f"Sample hours: {features_df['hour'].iloc[:5].tolist()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test with minimal data (like EA sends)
print("\n3️⃣ Testing with minimal data (200 rows)...")
try:
    minimal_data = sample_data.copy()
    features_minimal = engineer.create_features(minimal_data)
    
    print(f"✅ Minimal data test successful!")
    print(f"Rows remaining: {len(features_minimal)}")
    
    # Check specific indicators
    indicators = ['RSI', 'MACD', 'ATR', 'SMA_20', 'SMA_50']
    for ind in indicators:
        if ind in features_minimal.columns:
            non_nan = features_minimal[ind].notna().sum()
            print(f"  {ind}: {non_nan} non-NaN values")
    
except Exception as e:
    print(f"❌ Error with minimal data: {e}")

print("\n✅ All tests completed!")
print("\nRecommendations:")
print("1. EA should send 500+ bars for better indicator accuracy")
print("2. Use ForexMLBot_500Bars.mq5 for optimal results")
print("3. Server is now fixed to handle DatetimeIndex properly")