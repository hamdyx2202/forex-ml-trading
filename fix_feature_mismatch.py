#!/usr/bin/env python3
"""
Fix feature mismatch by extracting feature names from models
إصلاح عدم تطابق الميزات باستخراج أسماء الميزات من النماذج
"""

import joblib
import json
import os
from pathlib import Path

print("🔍 Extracting feature names from trained models...\n")

# Find model directory
model_dirs = [
    "models/advanced",
    "../models/advanced",
    "../../models/advanced",
    "/home/forex-ml-trading/models/advanced"
]

model_dir = None
for dir_path in model_dirs:
    if os.path.exists(dir_path):
        pkl_files = list(Path(dir_path).glob("*.pkl"))
        if pkl_files:
            model_dir = dir_path
            break

if not model_dir:
    print("❌ Could not find model directory!")
    exit(1)

print(f"✅ Found models in: {model_dir}")

# Load a model to extract feature names
pkl_files = list(Path(model_dir).glob("*.pkl"))
if pkl_files:
    print(f"\nLoading model: {pkl_files[0].name}")
    
    try:
        model_data = joblib.load(pkl_files[0])
        model = model_data['model']
        
        # Try different ways to get feature names
        feature_names = None
        
        # Method 1: From model directly
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
            print(f"✅ Found feature names in model: {len(feature_names)} features")
        
        # Method 2: From estimators (if VotingClassifier)
        elif hasattr(model, 'estimators_'):
            for est in model.estimators_:
                if hasattr(est, 'feature_names_in_'):
                    feature_names = est.feature_names_in_
                    print(f"✅ Found feature names in estimator: {len(feature_names)} features")
                    break
                elif hasattr(est, 'feature_name_'):
                    feature_names = est.feature_name_
                    print(f"✅ Found feature names in estimator: {len(feature_names)} features")
                    break
        
        # Method 3: From model data
        if feature_names is None and 'feature_names' in model_data:
            feature_names = model_data['feature_names']
            print(f"✅ Found feature names in model data: {len(feature_names)} features")
        
        if feature_names is not None:
            # Save feature names
            feature_names_list = list(feature_names) if hasattr(feature_names, '__iter__') else feature_names
            
            with open('model_feature_names.json', 'w') as f:
                json.dump({
                    'feature_count': len(feature_names_list),
                    'feature_names': feature_names_list
                }, f, indent=2)
            
            print(f"\n✅ Saved {len(feature_names_list)} feature names to model_feature_names.json")
            print(f"\nFirst 10 features:")
            for i, name in enumerate(feature_names_list[:10]):
                print(f"  {i+1}. {name}")
            
            # Create a fixed feature engineer that uses these exact features
            create_fixed_engineer(feature_names_list)
            
        else:
            print("❌ Could not extract feature names from model")
            print("\n🔧 Creating fallback solution...")
            
            # Try to infer from model structure
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
                print(f"✅ Model expects {n_features} features")
                
                with open('model_feature_count.json', 'w') as f:
                    json.dump({'expected_features': n_features}, f)
                
                print(f"✅ Saved expected feature count: {n_features}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def create_fixed_engineer(feature_names):
    """Create a feature engineer that produces exact features"""
    
    code = f'''#!/usr/bin/env python3
"""
Fixed Feature Engineer - Produces exactly {len(feature_names)} features
Auto-generated to match trained models
"""

import pandas as pd
import numpy as np
from feature_engineer_adaptive import AdaptiveFeatureEngineer

class FixedFeatureEngineer(AdaptiveFeatureEngineer):
    """Feature engineer that matches trained model features exactly"""
    
    def __init__(self):
        super().__init__(target_features={len(feature_names)})
        self.expected_features = {feature_names!r}
    
    def prepare_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with exact features expected by model"""
        # Create features
        df_features = self.create_features(df.copy())
        
        # Select only feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['target', 'target_binary', 'target_3class', 
                                     'future_return', 'time', 'open', 'high', 
                                     'low', 'close', 'volume', 'spread', 'datetime']]
        
        df_result = pd.DataFrame()
        
        # Add expected features in exact order
        for feat in self.expected_features:
            if feat in df_features.columns:
                df_result[feat] = df_features[feat]
            else:
                # Create missing feature with safe default
                if 'RSI' in feat or 'STOCH' in feat:
                    df_result[feat] = 50.0
                elif 'volume' in feat:
                    df_result[feat] = 0.0
                elif 'pattern' in feat:
                    df_result[feat] = 0
                else:
                    df_result[feat] = 0.0
        
        return df_result
'''
    
    with open('feature_engineer_fixed_final.py', 'w') as f:
        f.write(code)
    
    print("\n✅ Created feature_engineer_fixed_final.py")

print("\n🚀 Next steps:")
print("1. Check model_feature_names.json for extracted features")
print("2. Update server to use feature_engineer_fixed_final.py if created")
print("3. Or continue using adaptive engineer with target_features=68")