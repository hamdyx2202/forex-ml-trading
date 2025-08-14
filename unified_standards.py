#!/usr/bin/env python3
"""
Unified Standards for All Systems
معايير موحدة لجميع الأنظمة
"""

# المعايير الثابتة
STANDARD_FEATURES = 70  # عدد الميزات الثابت
MODEL_NAME_FORMAT = "{symbol}_{timeframe}.pkl"  # تنسيق أسماء النماذج

# قائمة الميزات القياسية (70 ميزة)
STANDARD_FEATURE_LIST = [
    # Price features (10)
    'returns', 'log_returns', 'hl_ratio', 'co_ratio', 'body_size',
    'upper_shadow', 'lower_shadow', 'oc_ratio', 'range_ratio', 'body_to_range',
    
    # Technical indicators (40)
    'rsi_14', 'rsi_21', 'rsi_30',
    'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    'ema_9', 'ema_21', 'ema_50',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
    'macd', 'macd_signal', 'macd_diff',
    'stoch_k', 'stoch_d',
    'williams_r', 'cci', 'mfi', 'adx', 'plus_di', 'minus_di',
    'aroon_up', 'aroon_down', 'ultimate_osc',
    'roc_5', 'roc_10', 'roc_20',
    'stddev_10', 'stddev_20',
    'atr_14', 'atr_ratio',
    'volume_ratio', 'volume_sma_ratio', 'obv',
    
    # Pattern features (10)
    'trend_strength', 'support_distance', 'resistance_distance',
    'price_position', 'volatility_regime_code', 'volume_regime_code',
    'hour_sin', 'hour_cos', 'day_of_week', 'is_london_session',
    
    # Padding features (10) - للوصول إلى 70
    'padding_0', 'padding_1', 'padding_2', 'padding_3', 'padding_4',
    'padding_5', 'padding_6', 'padding_7', 'padding_8', 'padding_9'
]

def validate_features(features):
    """التحقق من توافق الميزات مع المعايير"""
    if len(features) != STANDARD_FEATURES:
        return False, f"Feature count mismatch: {len(features)} != {STANDARD_FEATURES}"
    return True, "Features are valid"

def get_model_filename(symbol, timeframe):
    """الحصول على اسم النموذج القياسي"""
    return MODEL_NAME_FORMAT.format(symbol=symbol, timeframe=timeframe)

def ensure_standard_features(df, feature_cols):
    """ضمان 70 ميزة دائماً"""
    current_count = len(feature_cols)
    
    if current_count < STANDARD_FEATURES:
        # إضافة padding
        for i in range(current_count, STANDARD_FEATURES):
            df[f'padding_{i-current_count}'] = 0.0
            feature_cols.append(f'padding_{i-current_count}')
    elif current_count > STANDARD_FEATURES:
        # قص الميزات الزائدة
        feature_cols = feature_cols[:STANDARD_FEATURES]
    
    return df, feature_cols

# معايير التدريب
TRAINING_STANDARDS = {
    'n_features': STANDARD_FEATURES,
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'early_stopping_rounds': 50,
    'models': {
        'lightgbm': {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'max_depth': 10,
            'num_leaves': 31
        },
        'xgboost': {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'max_depth': 10
        },
        'catboost': {
            'iterations': 500,
            'learning_rate': 0.01,
            'depth': 10
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20
        }
    }
}

# معايير حفظ النماذج
SAVING_STANDARDS = {
    'models_dir': 'models/unified',
    'backup_dir': 'models/backup',
    'versioning': False,  # لا timestamps
    'compression': 3,  # مستوى ضغط joblib
    'include_metadata': True
}

print("📊 Unified Standards Loaded")
print(f"✅ Standard Features: {STANDARD_FEATURES}")
print(f"✅ Model Name Format: {MODEL_NAME_FORMAT}")
print(f"✅ Features List Length: {len(STANDARD_FEATURE_LIST)}")