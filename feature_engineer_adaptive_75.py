#!/usr/bin/env python3
"""
Adaptive Feature Engineering Module - 75 Features Version
يتضمن 70 ميزة أساسية + 5 ميزات جديدة للدعم والمقاومة
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# استيراد حاسب الدعم والمقاومة
from support_resistance import SupportResistanceCalculator

class AdaptiveFeatureEngineer75:
    """مهندس ميزات متطور مع 75 ميزة (70 أساسية + 5 دعم/مقاومة)"""
    
    def __init__(self, target_features: int = 75):
        """
        target_features: العدد المطلوب من الميزات (افتراضي 75)
        """
        self.target_features = target_features
        self.base_features = 70  # الميزات الأساسية الحالية
        self.sr_features = 5     # ميزات الدعم والمقاومة الجديدة
        self.feature_importance_order = None
        self.sr_calculator = SupportResistanceCalculator()
        
    def engineer_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        هندسة جميع الميزات مع ضمان 75 ميزة
        
        Args:
            df: DataFrame مع OHLCV
            symbol: رمز الزوج (مطلوب لحساب الدعم/المقاومة)
            
        Returns:
            DataFrame مع 75 ميزة بالضبط
        """
        try:
            # البدء بنسخة نظيفة
            df = df.copy()
            
            # 1. إضافة المؤشرات الفنية (حوالي 45-50 ميزة)
            df = self.add_technical_indicators(df)
            
            # 2. إضافة ميزات السعر (حوالي 15-20 ميزة)
            df = self.add_price_features(df)
            
            # 3. إضافة ميزات النمط (حوالي 5-10 ميزات)
            df = self.add_pattern_features(df)
            
            # 4. إضافة ميزات الدعم والمقاومة الجديدة (5 ميزات)
            if symbol:
                df = self.add_support_resistance_features(df, symbol)
            else:
                # قيم افتراضية إذا لم يتم توفير الرمز
                df = self.add_default_sr_features(df)
            
            # إزالة الأعمدة غير الرقمية
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # معالجة القيم الناقصة
            df[feature_cols] = df[feature_cols].fillna(0)
            
            # معالجة القيم اللانهائية
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
            
            # ضمان العدد الصحيح من الميزات
            current_features = len(feature_cols)
            
            if current_features < self.target_features:
                # إضافة ميزات padding
                logger.warning(f"Adding {self.target_features - current_features} padding features")
                for i in range(current_features, self.target_features):
                    df[f'padding_{i}'] = 0.0
                    feature_cols.append(f'padding_{i}')
            
            elif current_features > self.target_features:
                # اختيار أهم الميزات
                logger.warning(f"Reducing from {current_features} to {self.target_features} features")
                # الاحتفاظ بميزات الدعم/المقاومة دائماً
                sr_feature_names = [
                    'distance_to_support_pct',
                    'distance_to_resistance_pct',
                    'nearest_support_strength',
                    'nearest_resistance_strength',
                    'position_in_sr_range'
                ]
                
                # باقي الميزات
                other_features = [col for col in feature_cols if col not in sr_feature_names]
                
                # اختيار الميزات المطلوبة
                selected_features = sr_feature_names + other_features[:self.target_features - len(sr_feature_names)]
                feature_cols = selected_features
            
            # إعادة ترتيب الأعمدة
            final_cols = ['open', 'high', 'low', 'close', 'volume'] + sorted(feature_cols)
            df = df[final_cols]
            
            logger.info(f"✅ Engineered {len(feature_cols)} features (target: {self.target_features})")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة المؤشرات الفنية مع معالجة ذكية لـ NaN"""
        df = df.copy()
        n_bars = len(df)
        
        # Moving averages - only if enough data
        for period in [5, 10, 20, 50]:
            if n_bars >= period + 5:  # Add buffer
                df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Skip long period indicators if not enough data
        if n_bars >= 100:
            df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)
            df['EMA_100'] = talib.EMA(df['close'], timeperiod=100)
        
        if n_bars >= 200:
            df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
            df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # RSI
        if n_bars >= 20:
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        if n_bars >= 35:
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands
        if n_bars >= 25:
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_width'] + 0.0001)
        
        # ATR
        if n_bars >= 20:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Stochastic
        if n_bars >= 20:
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Volume indicators
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_SMA'] = df['volume'].rolling(window=min(20, n_bars-1), min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_SMA'].replace(0, 1)
        
        # ADX
        if n_bars >= 20:
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['DI_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['DI_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI
        if n_bars >= 20:
            df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Williams %R
        if n_bars >= 20:
            df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MFI (Money Flow Index)
        if n_bars >= 20 and 'volume' in df.columns:
            df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # SAR
        if n_bars >= 10:
            df['SAR'] = talib.SAR(df['high'], df['low'])
        
        # TEMA
        if n_bars >= 30:
            df['TEMA'] = talib.TEMA(df['close'], timeperiod=30)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات السعر"""
        df = df.copy()
        
        # Price ratios
        df['HL_ratio'] = df['high'] / df['low'].replace(0, 1)
        df['CO_ratio'] = df['close'] / df['open'].replace(0, 1)
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Returns over different periods
        for period in [5, 10, 20]:
            if len(df) >= period + 1:
                df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Candlestick features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, 1)
        
        # Support/Resistance levels (traditional)
        for period in [10, 20]:
            if len(df) >= period:
                df[f'dist_from_high_{period}'] = (df['close'] - df['high'].rolling(period, min_periods=1).max()) / df['close']
                df[f'dist_from_low_{period}'] = (df['close'] - df['low'].rolling(period, min_periods=1).min()) / df['close']
        
        # Volatility
        if len(df) >= 20:
            df['volatility'] = df['close'].rolling(20).std()
        
        # Price position
        if len(df) >= 50:
            df['price_position'] = (df['close'] - df['low'].rolling(50).min()) / (
                df['high'].rolling(50).max() - df['low'].rolling(50).min()
            ).replace(0, 1)
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات الأنماط"""
        df = df.copy()
        n_bars = len(df)
        
        # Consecutive movements
        df['consecutive_ups'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_ups'] = df['consecutive_ups'].groupby((df['consecutive_ups'] != df['consecutive_ups'].shift()).cumsum()).cumsum()
        
        df['consecutive_downs'] = (df['close'] < df['close'].shift(1)).astype(int)
        df['consecutive_downs'] = df['consecutive_downs'].groupby((df['consecutive_downs'] != df['consecutive_downs'].shift()).cumsum()).cumsum()
        
        # Doji detection
        df['is_doji'] = (abs(df['close'] - df['open']) < df['body_size'].rolling(20, min_periods=1).mean() * 0.1).astype(int)
        
        # Engulfing patterns
        if n_bars >= 2:
            # Bullish engulfing
            df['bullish_engulfing'] = (
                (df['close'] > df['open']) &  # Current candle is bullish
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous was bearish
                (df['open'] < df['close'].shift(1)) &  # Open below previous close
                (df['close'] > df['open'].shift(1))  # Close above previous open
            ).astype(int)
            
            # Bearish engulfing
            df['bearish_engulfing'] = (
                (df['close'] < df['open']) &  # Current candle is bearish
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous was bullish
                (df['open'] > df['close'].shift(1)) &  # Open above previous close
                (df['close'] < df['open'].shift(1))  # Close below previous open
            ).astype(int)
        
        # Hammer and shooting star
        df['hammer'] = (
            (df['lower_shadow'] > df['body_size'] * 2) &
            (df['upper_shadow'] < df['body_size'] * 0.5)
        ).astype(int)
        
        df['shooting_star'] = (
            (df['upper_shadow'] > df['body_size'] * 2) &
            (df['lower_shadow'] < df['body_size'] * 0.5)
        ).astype(int)
        
        return df
    
    def add_support_resistance_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        إضافة 5 ميزات الدعم والمقاومة الجديدة
        
        Args:
            df: DataFrame مع OHLCV
            symbol: رمز الزوج
            
        Returns:
            DataFrame مع ميزات S/R الجديدة
        """
        try:
            # حساب مستويات الدعم والمقاومة
            sr_levels = self.sr_calculator.calculate_all_levels(df, symbol)
            
            # استخراج الميزات
            features = sr_levels['features']
            
            # إضافة الميزات للـ DataFrame
            df['distance_to_support_pct'] = features['distance_to_support_pct']
            df['distance_to_resistance_pct'] = features['distance_to_resistance_pct']
            df['nearest_support_strength'] = features['nearest_support_strength']
            df['nearest_resistance_strength'] = features['nearest_resistance_strength']
            df['position_in_sr_range'] = features['position_in_sr_range']
            
            logger.info(f"✅ Added 5 S/R features for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding S/R features: {str(e)}")
            # إضافة قيم افتراضية في حالة الخطأ
            return self.add_default_sr_features(df)
    
    def add_default_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة قيم افتراضية لميزات الدعم والمقاومة"""
        df['distance_to_support_pct'] = 5.0
        df['distance_to_resistance_pct'] = 5.0
        df['nearest_support_strength'] = 0.5
        df['nearest_resistance_strength'] = 0.5
        df['position_in_sr_range'] = 0.5
        
        logger.warning("Using default S/R feature values")
        return df
    
    def get_feature_names(self) -> List[str]:
        """الحصول على أسماء جميع الميزات بالترتيب"""
        # قائمة الميزات المتوقعة (يجب تحديثها حسب الميزات الفعلية)
        feature_names = []
        
        # المؤشرات الفنية
        for period in [5, 10, 20, 50, 100, 200]:
            feature_names.extend([f'SMA_{period}', f'EMA_{period}'])
        
        feature_names.extend([
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position',
            'ATR', 'STOCH_K', 'STOCH_D', 'volume_SMA', 'volume_ratio',
            'ADX', 'DI_plus', 'DI_minus', 'CCI', 'WILLR', 'MFI', 'SAR', 'TEMA'
        ])
        
        # ميزات السعر
        feature_names.extend([
            'HL_ratio', 'CO_ratio', 'price_change', 'price_change_abs',
            'return_5', 'return_10', 'return_20',
            'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
            'dist_from_high_10', 'dist_from_low_10',
            'dist_from_high_20', 'dist_from_low_20',
            'volatility', 'price_position'
        ])
        
        # ميزات الأنماط
        feature_names.extend([
            'consecutive_ups', 'consecutive_downs', 'is_doji',
            'bullish_engulfing', 'bearish_engulfing',
            'hammer', 'shooting_star'
        ])
        
        # ميزات الدعم والمقاومة (الجديدة)
        feature_names.extend([
            'distance_to_support_pct',
            'distance_to_resistance_pct',
            'nearest_support_strength',
            'nearest_resistance_strength',
            'position_in_sr_range'
        ])
        
        # إضافة padding إذا لزم الأمر
        while len(feature_names) < self.target_features:
            feature_names.append(f'padding_{len(feature_names)}')
        
        return feature_names[:self.target_features]


# دالة مساعدة للاستخدام المباشر
def engineer_features_75(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    دالة سريعة لهندسة 75 ميزة
    
    Args:
        df: DataFrame مع OHLCV
        symbol: رمز الزوج (مطلوب لحساب الدعم/المقاومة)
        
    Returns:
        DataFrame مع 75 ميزة بالضبط
    """
    engineer = AdaptiveFeatureEngineer75()
    return engineer.engineer_features(df, symbol)


if __name__ == "__main__":
    # مثال للاختبار
    print("🧪 Testing 75-feature engineering...")
    
    # بيانات تجريبية
    test_data = {
        'open': np.random.rand(200) * 0.01 + 1.1000,
        'high': np.random.rand(200) * 0.01 + 1.1100,
        'low': np.random.rand(200) * 0.01 + 1.0900,
        'close': np.random.rand(200) * 0.01 + 1.1000,
        'volume': np.random.randint(1000, 10000, 200)
    }
    
    df = pd.DataFrame(test_data)
    
    # هندسة الميزات
    engineer = AdaptiveFeatureEngineer75()
    df_features = engineer.engineer_features(df, "EURUSD")
    
    # عرض النتائج
    feature_cols = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    print(f"\n✅ Total features: {len(feature_cols)}")
    print(f"📊 Feature columns: {feature_cols[:10]}...")
    print(f"\n🎯 S/R Features:")
    sr_features = ['distance_to_support_pct', 'distance_to_resistance_pct', 
                   'nearest_support_strength', 'nearest_resistance_strength', 
                   'position_in_sr_range']
    for feat in sr_features:
        if feat in df_features.columns:
            print(f"   {feat}: {df_features[feat].iloc[-1]:.4f}")