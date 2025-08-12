#!/usr/bin/env python3
"""
Test adaptive feature engineer fix
اختبار إصلاح AdaptiveFeatureEngineer
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("🔍 Testing AdaptiveFeatureEngineer fix...\n")

# Simulate data from EA (no datetime column, datetime is in index)
print("1️⃣ Creating sample data like EA sends...")
data = {
    'time': [1732000000 + i*300 for i in range(200)],  # timestamps
    'open': np.random.randn(200).cumsum() + 100,
    'high': np.random.randn(200).cumsum() + 101,
    'low': np.random.randn(200).cumsum() + 99,
    'close': np.random.randn(200).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 200)
}

df = pd.DataFrame(data)
# Convert time to datetime index (like server does)
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df.set_index('datetime', inplace=True)

print(f"Created DataFrame:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Index type: {type(df.index)}")
print(f"  Index name: {df.index.name}")

# Test feature engineering
print("\n2️⃣ Testing feature engineering...")
try:
    from feature_engineer_adaptive import AdaptiveFeatureEngineer
    
    engineer = AdaptiveFeatureEngineer(target_features=68)
    features_df = engineer.create_features(df.copy())
    
    print(f"✅ Feature engineering successful!")
    print(f"  Original rows: {len(df)}")
    print(f"  After features: {len(features_df)} rows")
    
    # Count actual features
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    print(f"  Number of features: {len(feature_cols)}")
    
    # Check if we have the target number
    if len(feature_cols) == 68:
        print(f"  ✅ Exactly 68 features as expected!")
    else:
        print(f"  ⚠️ Got {len(feature_cols)} features, expected 68")
    
    # Test prepare_for_prediction
    print("\n3️⃣ Testing prepare_for_prediction...")
    pred_df = engineer.prepare_for_prediction(df.copy())
    print(f"✅ Prediction preparation successful!")
    print(f"  Features for prediction: {pred_df.shape[1]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test completed!")
print("\nIf successful, the server should now work without 'datetime' errors!")