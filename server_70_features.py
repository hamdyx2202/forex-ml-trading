#!/usr/bin/env python3
"""
Server Wrapper with Feature Fix
غلاف الخادم مع إصلاح الميزات
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# تعديل AdaptiveFeatureEngineer ليُنتج 70 ميزة
import feature_engineer_adaptive

# تعديل __init__ ليستهدف 70
original_init = feature_engineer_adaptive.AdaptiveFeatureEngineer.__init__

def new_init(self, target_features=None):
    original_init(self, target_features=70)  # فرض 70 ميزة
    print("🎯 Forcing 70 features")

feature_engineer_adaptive.AdaptiveFeatureEngineer.__init__ = new_init

# تعديل create_features لإضافة padding
original_create = feature_engineer_adaptive.AdaptiveFeatureEngineer.create_features

def new_create_features(self, df, target_config=None):
    result = original_create(self, df, target_config)
    
    # التحقق من عدد الميزات
    feature_cols = [col for col in result.columns 
                   if col not in ['target', 'target_binary', 'target_3class', 
                                 'future_return', 'time', 'open', 'high', 
                                 'low', 'close', 'volume', 'spread', 'datetime']]
    
    current_count = len(feature_cols)
    
    if current_count < 70:
        print(f"📊 Adding {70 - current_count} padding features...")
        # إضافة أعمدة padding
        for i in range(current_count, 70):
            result[f'padding_feature_{i}'] = 0.0
    
    return result

feature_engineer_adaptive.AdaptiveFeatureEngineer.create_features = new_create_features

# الآن استيراد وتشغيل الخادم
print("🚀 Starting server with 70-feature fix...")
from src.mt5_bridge_server_advanced import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
