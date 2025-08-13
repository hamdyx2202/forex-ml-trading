#!/usr/bin/env python3
"""
Unified Feature Engineering Module
وحدة موحدة لهندسة الميزات - تُستخدم في التدريب والتنبؤ
Version: 2.0
"""

import talib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class UnifiedFeatureEngineer:
    """
    Feature engineer موحد يضمن نفس الميزات في التدريب والتنبؤ
    """
    
    # إصدار المميزات - يجب زيادته عند أي تغيير
    VERSION = "2.0"
    
    # قائمة المؤشرات المستخدمة
    INDICATORS_CONFIG = {
        'rsi_periods': [14, 21, 30],
        'ma_periods': [10, 20, 50, 100, 200],
        'bb_period': 20,
        'macd_params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'atr_period': 14,
        'adx_period': 14,
        'stoch_params': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}
    }
    
    @staticmethod
    def create_features(data: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
        """
        إنشاء المميزات بطريقة موحدة
        
        Parameters:
        -----------
        data : DataFrame
            بيانات OHLCV مع datetime index
        config : dict
            إعدادات المميزات (اختياري)
        
        Returns:
        --------
        features : array
            مصفوفة المميزات
        feature_names : list
            أسماء المميزات
        """
        
        if config is None:
            config = UnifiedFeatureEngineer.INDICATORS_CONFIG
            
        features = []
        feature_names = []
        
        # التحقق من البيانات
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(f"Creating features for {len(data)} rows")
        
        # 1. المؤشرات الفنية الأساسية
        # RSI - Relative Strength Index
        for period in config['rsi_periods']:
            rsi = talib.RSI(data['close'].values, timeperiod=period)
            features.append(rsi)
            feature_names.append(f'rsi_{period}')
        
        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = talib.MACD(
            data['close'].values, 
            **config['macd_params']
        )
        features.extend([macd, macd_signal, macd_hist])
        feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data['close'].values, 
            timeperiod=config['bb_period']
        )
        features.extend([bb_upper, bb_middle, bb_lower])
        feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower'])
        
        # BB Width and Position
        bb_width = bb_upper - bb_lower
        bb_position = (data['close'].values - bb_lower) / (bb_width + 0.0001)
        features.extend([bb_width, bb_position])
        feature_names.extend(['bb_width', 'bb_position'])
        
        # Moving Averages - SMA & EMA
        for period in config['ma_periods']:
            if len(data) >= period:
                sma = talib.SMA(data['close'].values, timeperiod=period)
                ema = talib.EMA(data['close'].values, timeperiod=period)
                features.extend([sma, ema])
                feature_names.extend([f'sma_{period}', f'ema_{period}'])
            else:
                # إضافة قيم افتراضية إذا لم تكن البيانات كافية
                features.extend([np.full(len(data), np.nan), np.full(len(data), np.nan)])
                feature_names.extend([f'sma_{period}', f'ema_{period}'])
        
        # 2. مؤشرات الحجم
        # OBV - On Balance Volume
        obv = talib.OBV(data['close'].values, data['volume'].values.astype(float))
        features.append(obv)
        feature_names.append('obv')
        
        # Volume SMA
        volume_sma = talib.SMA(data['volume'].values.astype(float), timeperiod=20)
        volume_ratio = data['volume'].values / (volume_sma + 1)
        features.extend([volume_sma, volume_ratio])
        feature_names.extend(['volume_sma_20', 'volume_ratio'])
        
        # 3. مؤشرات التذبذب والقوة
        # ATR - Average True Range
        atr = talib.ATR(
            data['high'].values, 
            data['low'].values, 
            data['close'].values, 
            timeperiod=config['atr_period']
        )
        features.append(atr)
        feature_names.append('atr')
        
        # ADX - Average Directional Index
        adx = talib.ADX(
            data['high'].values, 
            data['low'].values, 
            data['close'].values, 
            timeperiod=config['adx_period']
        )
        features.append(adx)
        feature_names.append('adx')
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            data['high'].values, 
            data['low'].values, 
            data['close'].values,
            **config['stoch_params']
        )
        features.extend([slowk, slowd])
        feature_names.extend(['stoch_k', 'stoch_d'])
        
        # 4. النسب والعلاقات السعرية
        # Price to Moving Averages
        for period in [20, 50]:
            sma = talib.SMA(data['close'].values, timeperiod=period)
            price_to_sma = data['close'].values / (sma + 0.0001)
            features.append(price_to_sma)
            feature_names.append(f'price_to_sma_{period}')
        
        # High-Low Ratio
        hl_ratio = data['high'].values / (data['low'].values + 0.0001)
        features.append(hl_ratio)
        feature_names.append('hl_ratio')
        
        # Close-Open Ratio
        co_ratio = data['close'].values / (data['open'].values + 0.0001)
        features.append(co_ratio)
        feature_names.append('co_ratio')
        
        # 5. مميزات الشموع اليابانية
        # Candle Range
        candle_range = data['high'].values - data['low'].values
        features.append(candle_range)
        feature_names.append('candle_range')
        
        # Candle Body
        candle_body = np.abs(data['close'].values - data['open'].values)
        features.append(candle_body)
        feature_names.append('candle_body')
        
        # Body Ratio
        body_ratio = candle_body / (candle_range + 0.0001)
        features.append(body_ratio)
        feature_names.append('body_ratio')
        
        # Upper Shadow
        upper_shadow = data['high'].values - np.maximum(data['open'].values, data['close'].values)
        features.append(upper_shadow)
        feature_names.append('upper_shadow')
        
        # Lower Shadow
        lower_shadow = np.minimum(data['open'].values, data['close'].values) - data['low'].values
        features.append(lower_shadow)
        feature_names.append('lower_shadow')
        
        # 6. مميزات زمنية (إذا كان لدينا datetime index)
        if isinstance(data.index, pd.DatetimeIndex):
            hour = data.index.hour
            day_of_week = data.index.dayofweek
            
            features.extend([hour, day_of_week])
            feature_names.extend(['hour', 'day_of_week'])
            
            # Trading Sessions
            asian_session = ((hour >= 0) & (hour < 8)).astype(int)
            london_session = ((hour >= 8) & (hour < 16)).astype(int)
            ny_session = ((hour >= 13) & (hour < 22)).astype(int)
            
            features.extend([asian_session, london_session, ny_session])
            feature_names.extend(['asian_session', 'london_session', 'ny_session'])
        
        # 7. تحويلات الأسعار
        # Returns
        returns_1 = data['close'].pct_change(1).values
        returns_5 = data['close'].pct_change(5).values
        returns_10 = data['close'].pct_change(10).values
        
        features.extend([returns_1, returns_5, returns_10])
        feature_names.extend(['returns_1', 'returns_5', 'returns_10'])
        
        # تحويل إلى numpy array
        features_array = np.column_stack(features)
        
        # معالجة NaN
        features_array = UnifiedFeatureEngineer._handle_nan(features_array)
        
        logger.info(f"Created {len(feature_names)} features")
        
        return features_array, feature_names
    
    @staticmethod
    def _handle_nan(features: np.ndarray) -> np.ndarray:
        """
        معالجة القيم المفقودة بطريقة ذكية
        """
        # Forward fill
        df_temp = pd.DataFrame(features)
        df_temp = df_temp.fillna(method='ffill', limit=5)
        
        # Backward fill
        df_temp = df_temp.fillna(method='bfill', limit=5)
        
        # Fill remaining with safe defaults
        for col in df_temp.columns:
            if df_temp[col].isna().any():
                # استخدام المتوسط للأعمدة العادية
                col_mean = df_temp[col].mean()
                if np.isnan(col_mean):
                    # إذا كان كل العمود NaN، استخدم 0
                    df_temp[col] = df_temp[col].fillna(0)
                else:
                    df_temp[col] = df_temp[col].fillna(col_mean)
        
        return df_temp.values
    
    @staticmethod
    def validate_features(features: np.ndarray, expected_names: List[str]) -> bool:
        """
        التحقق من صحة المميزات
        """
        if features.shape[1] != len(expected_names):
            raise ValueError(
                f"Feature count mismatch: "
                f"got {features.shape[1]}, "
                f"expected {len(expected_names)}"
            )
        return True
    
    @staticmethod
    def get_feature_info() -> Dict:
        """
        الحصول على معلومات المميزات
        """
        # إنشاء بيانات وهمية للحصول على أسماء المميزات
        dummy_data = pd.DataFrame({
            'open': np.random.rand(200),
            'high': np.random.rand(200),
            'low': np.random.rand(200),
            'close': np.random.rand(200),
            'volume': np.random.rand(200)
        })
        dummy_data.index = pd.date_range('2024-01-01', periods=200, freq='1H')
        
        _, feature_names = UnifiedFeatureEngineer.create_features(dummy_data)
        
        return {
            'version': UnifiedFeatureEngineer.VERSION,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'config': UnifiedFeatureEngineer.INDICATORS_CONFIG
        }

# دالة مساعدة للاستخدام المباشر
def create_unified_features(data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    دالة مساعدة لإنشاء المميزات
    """
    return UnifiedFeatureEngineer.create_features(data)

# للاختبار
if __name__ == "__main__":
    print("🔍 Testing Unified Feature Engineer...")
    
    # بيانات اختبار
    test_data = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    })
    test_data.index = pd.date_range('2024-01-01', periods=500, freq='1H')
    
    # إنشاء المميزات
    features, names = UnifiedFeatureEngineer.create_features(test_data)
    
    print(f"✅ Created {len(names)} features")
    print(f"📊 Feature shape: {features.shape}")
    print(f"\nFirst 10 features:")
    for i, name in enumerate(names[:10]):
        print(f"  {i+1}. {name}")
    
    # معلومات المميزات
    info = UnifiedFeatureEngineer.get_feature_info()
    print(f"\n📋 Feature Engineering Version: {info['version']}")
    print(f"📊 Total Features: {info['n_features']}")