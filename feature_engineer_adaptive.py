#!/usr/bin/env python3
"""
Adaptive Feature Engineering Module
يتكيف مع عدد الميزات المطلوب من النموذج
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AdaptiveFeatureEngineer:
    """مهندس ميزات تكيفي"""
    
    def __init__(self, target_features: Optional[int] = None):
        """
        target_features: العدد المطلوب من الميزات (68-69 للنماذج المدربة)
        """
        self.target_features = target_features
        self.feature_importance_order = None
        
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
        
        # Candlestick features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, 1)
        
        # Support/Resistance
        for period in [10, 20]:
            if len(df) >= period:
                df[f'dist_from_high_{period}'] = (df['close'] - df['high'].rolling(period, min_periods=1).max()) / df['close']
                df[f'dist_from_low_{period}'] = (df['close'] - df['low'].rolling(period, min_periods=1).min()) / df['close']
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة أنماط الشموع الأساسية فقط"""
        df = df.copy()
        
        if len(df) >= 10:
            # Only most important patterns
            patterns = ['CDLDOJI', 'CDLHAMMER', 'CDLENGULFING']
            
            for pattern in patterns:
                try:
                    pattern_func = getattr(talib, pattern)
                    df[f'pattern_{pattern}'] = pattern_func(
                        df['open'], df['high'], df['low'], df['close']
                    )
                except:
                    pass
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات زمنية"""
        df = df.copy()
        
        # Handle datetime
        if isinstance(df.index, pd.DatetimeIndex):
            time_idx = df.index
        elif 'datetime' in df.columns:
            time_idx = pd.to_datetime(df['datetime'])
        elif 'time' in df.columns:
            time_idx = pd.to_datetime(df['time'], unit='s')
        else:
            return df
        
        # Extract time components
        df['hour'] = time_idx.hour
        df['day_of_week'] = time_idx.dayofweek
        
        # Trading sessions
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        return df
    
    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة بنية السوق الأساسية"""
        df = df.copy()
        
        # Simple trend
        if 'SMA_20' in df.columns:
            df['trend_sma20'] = (df['close'] > df['SMA_20']).astype(int)
        
        if 'SMA_50' in df.columns:
            df['trend_sma50'] = (df['close'] > df['SMA_50']).astype(int)
        
        # Volatility
        if len(df) >= 20:
            df['volatility'] = df['close'].pct_change().rolling(20, min_periods=2).std()
        
        return df
    
    def create_features(self, df: pd.DataFrame, target_config: Optional[Dict] = None) -> pd.DataFrame:
        """إنشاء الميزات مع التكيف لعدد الميزات المطلوب"""
        logger.info(f"Starting adaptive feature engineering for {len(df)} rows")
        
        # Keep essential columns
        essential_cols = ['open', 'high', 'low', 'close', 'volume', 'time', 'datetime']
        df_essential = df[essential_cols].copy() if 'volume' in df else df[essential_cols[:-1]].copy()
        
        # Add features progressively
        df = self.add_technical_indicators(df)
        df = self.add_price_features(df)
        df = self.add_time_features(df)
        df = self.add_market_structure(df)
        
        # Only add patterns if we need more features
        current_features = [col for col in df.columns if col not in essential_cols]
        if len(current_features) < 70:
            df = self.add_pattern_features(df)
        
        # Smart NaN handling
        initial_rows = len(df)
        
        # Fill NaN intelligently
        for col in df.columns:
            if col in essential_cols:
                continue
                
            if df[col].isna().all():
                df.drop(columns=[col], inplace=True)
                continue
            
            # Forward fill then backward fill
            df[col] = df[col].fillna(method='ffill', limit=5)
            df[col] = df[col].fillna(method='bfill', limit=5)
            
            # Fill remaining with mean or safe default
            if df[col].isna().any():
                if 'RSI' in col or 'STOCH' in col:
                    df[col] = df[col].fillna(50)
                elif 'volume' in col:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # Drop rows with too many NaN (more than 20% of features)
        nan_threshold = 0.2 * len(df.columns)
        df = df.dropna(thresh=len(df.columns) - nan_threshold)
        
        logger.info(f"After NaN handling: {len(df)} rows remain (removed {initial_rows - len(df)} rows)")
        
        # Add target if requested
        if target_config and 'target' not in df.columns:
            df = self.add_target_variable(df, **target_config)
        
        # Select features (exclude non-feature columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['target', 'target_binary', 'target_3class', 
                                     'future_return', 'time', 'open', 'high', 
                                     'low', 'close', 'volume', 'spread', 'datetime']]
        
        # Adaptive feature selection
        if self.target_features and len(feature_cols) > self.target_features:
            # Select most important features based on variance
            variances = df[feature_cols].var()
            selected_features = variances.nlargest(self.target_features).index.tolist()
            
            # Keep only selected features plus essential columns
            keep_cols = essential_cols + selected_features
            if 'target' in df.columns:
                keep_cols.append('target')
            if 'target_binary' in df.columns:
                keep_cols.append('target_binary')
                
            df = df[keep_cols]
            logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)} available")
        
        logger.info(f"Feature engineering completed. Features: {len(feature_cols)}")
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001) -> pd.DataFrame:
        """إضافة متغير الهدف"""
        df = df.copy()
        
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['target'] = np.where(df['future_return'] > threshold, 1, 
                               np.where(df['future_return'] < -threshold, -1, 0))
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        # Remove last rows without target
        df = df[:-lookahead]
        
        return df
    
    def prepare_for_prediction(self, df: pd.DataFrame, expected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """تحضير البيانات للتنبؤ مع مطابقة الميزات المتوقعة"""
        df_features = self.create_features(df.copy())
        
        # Remove target columns
        target_cols = ['target', 'target_3class', 'target_binary', 'future_return']
        df_features = df_features.drop(columns=[col for col in target_cols if col in df_features.columns])
        
        # Match expected features if provided
        if expected_features:
            # Add missing features with 0
            for feat in expected_features:
                if feat not in df_features.columns:
                    df_features[feat] = 0
            
            # Select only expected features in correct order
            df_features = df_features[expected_features]
        
        return df_features

# للاستخدام المباشر
def create_adaptive_engineer(target_features: int = 68) -> AdaptiveFeatureEngineer:
    """إنشاء مهندس ميزات يستهدف 68 ميزة"""
    return AdaptiveFeatureEngineer(target_features=target_features)