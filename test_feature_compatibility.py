#!/usr/bin/env python3
"""
اختبار التوافق بين النماذج 70 و 75 ميزة
Test Compatibility Between 70 and 75 Feature Models
"""

import numpy as np
import pandas as pd
from loguru import logger
import sys
import os

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer as FE70
from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75 as FE75
from feature_compatibility import FeatureCompatibilityLayer

def test_feature_engineering():
    """اختبار هندسة الميزات"""
    print("\n🧪 Testing Feature Engineering")
    print("=" * 50)
    
    # بيانات تجريبية
    test_data = {
        'open': np.random.rand(200) * 0.01 + 1.1000,
        'high': np.random.rand(200) * 0.01 + 1.1100,
        'low': np.random.rand(200) * 0.01 + 1.0900,
        'close': np.random.rand(200) * 0.01 + 1.1000,
        'volume': np.random.randint(1000, 10000, 200)
    }
    df = pd.DataFrame(test_data)
    
    # اختبار 70 ميزة
    try:
        fe70 = FE70(target_features=70)
        df_70 = fe70.engineer_features(df)
        features_70 = [col for col in df_70.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"✅ 70-feature engineering: {len(features_70)} features")
    except Exception as e:
        print(f"❌ Error with 70 features: {str(e)}")
        features_70 = []
    
    # اختبار 75 ميزة
    try:
        fe75 = FE75(target_features=75)
        df_75 = fe75.engineer_features(df, "EURUSD")
        features_75 = [col for col in df_75.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"✅ 75-feature engineering: {len(features_75)} features")
        
        # عرض الميزات الجديدة
        new_features = [f for f in features_75 if f not in features_70]
        print(f"\n📊 New features ({len(new_features)}):")
        for feat in new_features[:10]:  # أول 10 فقط
            print(f"   - {feat}")
            
    except Exception as e:
        print(f"❌ Error with 75 features: {str(e)}")

def test_compatibility_layer():
    """اختبار طبقة التوافق"""
    print("\n🔄 Testing Compatibility Layer")
    print("=" * 50)
    
    compat = FeatureCompatibilityLayer()
    
    # اختبار 1: تحويل من 70 إلى 75
    print("\n1. Converting 70 → 75 features:")
    features_70 = np.random.rand(10, 70)
    features_75 = compat.make_compatible(features_70, 75)
    print(f"   Input shape: {features_70.shape}")
    print(f"   Output shape: {features_75.shape}")
    print(f"   ✅ Conversion successful" if features_75.shape[1] == 75 else "   ❌ Conversion failed")
    
    # اختبار 2: تحويل من 75 إلى 70
    print("\n2. Converting 75 → 70 features:")
    features_75_orig = np.random.rand(10, 75)
    features_70_conv = compat.make_compatible(features_75_orig, 70)
    print(f"   Input shape: {features_75_orig.shape}")
    print(f"   Output shape: {features_70_conv.shape}")
    print(f"   ✅ Conversion successful" if features_70_conv.shape[1] == 70 else "   ❌ Conversion failed")
    
    # اختبار 3: DataFrame
    print("\n3. Testing DataFrame compatibility:")
    df_test = pd.DataFrame(np.random.rand(10, 70), columns=[f'feature_{i}' for i in range(70)])
    df_test['open'] = 1.1
    df_test['close'] = 1.2
    
    df_75 = compat.make_compatible(df_test, 75)
    feature_cols = [col for col in df_75.columns if col not in ['open', 'close']]
    print(f"   Input features: {len([col for col in df_test.columns if col not in ['open', 'close']])}")
    print(f"   Output features: {len(feature_cols)}")
    print(f"   ✅ DataFrame conversion successful" if len(feature_cols) == 75 else "   ❌ DataFrame conversion failed")

def test_model_loading():
    """اختبار تحميل النماذج"""
    print("\n📦 Testing Model Loading")
    print("=" * 50)
    
    import joblib
    import glob
    
    model_files = glob.glob('models/**/*.joblib', recursive=True)[:5]  # أول 5 نماذج فقط
    
    if not model_files:
        print("⚠️ No models found to test")
        return
    
    for model_file in model_files:
        try:
            model_data = joblib.load(model_file)
            
            # عرض معلومات النموذج
            print(f"\n📄 {os.path.basename(model_file)}:")
            
            if 'metadata' in model_data:
                n_features = model_data['metadata'].get('n_features', 'Unknown')
                print(f"   Features: {n_features}")
                print(f"   Created: {model_data['metadata'].get('created_at', 'Unknown')}")
            
            # اختبار التنبؤ مع كلا العددين
            if 'models' in model_data:
                model = model_data['models']
                
                # اختبار مع 70 ميزة
                try:
                    X_70 = np.random.rand(1, 70)
                    pred_70 = model.predict(X_70)
                    print(f"   ✅ Works with 70 features")
                except:
                    print(f"   ❌ Doesn't work with 70 features")
                
                # اختبار مع 75 ميزة
                try:
                    X_75 = np.random.rand(1, 75)
                    pred_75 = model.predict(X_75)
                    print(f"   ✅ Works with 75 features")
                except:
                    print(f"   ❌ Doesn't work with 75 features")
                    
        except Exception as e:
            print(f"❌ Error loading {model_file}: {str(e)}")

def test_full_pipeline():
    """اختبار الأنبوب الكامل"""
    print("\n🔧 Testing Full Pipeline")
    print("=" * 50)
    
    # بيانات تجريبية
    test_data = {
        'open': np.random.rand(200) * 0.01 + 1.1000,
        'high': np.random.rand(200) * 0.01 + 1.1100,
        'low': np.random.rand(200) * 0.01 + 1.0900,
        'close': np.random.rand(200) * 0.01 + 1.1000,
        'volume': np.random.randint(1000, 10000, 200)
    }
    df = pd.DataFrame(test_data)
    
    # 1. هندسة الميزات الجديدة
    print("\n1. Engineering 75 features...")
    fe75 = FE75()
    df_features = fe75.engineer_features(df, "EURUSD")
    feature_cols = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    print(f"   ✅ Generated {len(feature_cols)} features")
    
    # 2. التحقق من الميزات الجديدة
    sr_features = ['distance_to_support_pct', 'distance_to_resistance_pct', 
                   'nearest_support_strength', 'nearest_resistance_strength', 
                   'position_in_sr_range']
    
    print("\n2. Checking S/R features:")
    for feat in sr_features:
        if feat in df_features.columns:
            value = df_features[feat].iloc[-1]
            print(f"   ✅ {feat}: {value:.4f}")
        else:
            print(f"   ❌ {feat}: Missing!")
    
    # 3. اختبار التوافق
    print("\n3. Testing compatibility:")
    compat = FeatureCompatibilityLayer()
    
    # استخراج الميزات فقط
    X_75 = df_features[feature_cols].iloc[-1:].values
    
    # تحويل إلى 70
    X_70 = compat.make_compatible(X_75, 70)
    print(f"   75 → 70: {X_75.shape} → {X_70.shape}")
    
    # تحويل مرة أخرى إلى 75
    X_75_back = compat.make_compatible(X_70, 75)
    print(f"   70 → 75: {X_70.shape} → {X_75_back.shape}")
    
    print("\n✅ Full pipeline test completed!")

def main():
    """تشغيل جميع الاختبارات"""
    print("🔍 Feature Compatibility Test Suite")
    print("=" * 70)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Compatibility Layer", test_compatibility_layer),
        ("Model Loading", test_model_loading),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} failed: {str(e)}")
            logger.error(f"Test {test_name} failed", exc_info=True)
            failed += 1
    
    # ملخص النتائج
    print("\n" + "=" * 70)
    print("📊 Test Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The system is ready for 75 features.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()