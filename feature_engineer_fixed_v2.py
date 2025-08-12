#!/usr/bin/env python3
"""
Fixed Feature Engineering Module V2
نسخة محسنة تتعامل مع datetime بشكل صحيح
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering for forex data - Fixed version V2"""
    
    def __init__(self):
        logger.add("logs/feature_engineer.log", rotation="1 day", retention="30 days")
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة المؤشرات الفنية"""
        df = df.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # RSI
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = hist
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period)
            df[f'BB_upper_{period}'] = upper
            df[f'BB_middle_{period}'] = middle
            df[f'BB_lower_{period}'] = lower
            df[f'BB_width_{period}'] = upper - lower
            df[f'BB_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-10)
        
        # ATR
        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Stochastic
        for period in [5, 14]:
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                                       fastk_period=period, slowk_period=3, slowd_period=3)
            df[f'STOCH_K_{period}'] = slowk
            df[f'STOCH_D_{period}'] = slowd
        
        # ADX
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
        df['DI_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
        df['DI_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'])
        
        # CCI
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # OBV
        df['OBV'] = talib.OBV(df['close'], df['volume'].astype(float))
        
        # MFI
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'].astype(float))
        
        logger.info(f"Added {len([col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'STOCH', 'ADX', 'CCI', 'WILLR', 'OBV', 'MFI'])])} technical indicators")
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات الأسعار"""
        df = df.copy()
        
        # Price changes
        for period in [1, 5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # High-Low features
        df['HL_ratio'] = df['high'] / df['low']
        df['HL_spread'] = df['high'] - df['low']
        df['OC_ratio'] = df['open'] / df['close']
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_candle_ratio'] = df['body_size'] / (df['high'] - df['low'] + 1e-10)
        
        # Support/Resistance levels
        for period in [20, 50, 100]:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'price_to_resistance_{period}'] = df['close'] / df[f'resistance_{period}']
            df[f'price_to_support_{period}'] = df['close'] / df[f'support_{period}']
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة أنماط الشموع"""
        df = df.copy()
        
        # Candlestick patterns using TA-Lib
        pattern_functions = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'ENGULFING': talib.CDLENGULFING,
            'STAR': talib.CDLMORNINGSTAR,
            'HARAMI': talib.CDLHARAMI,
            'THREEWHITESOLDIERS': talib.CDL3WHITESOLDIERS,
            'THREEBLACKCROWS': talib.CDL3BLACKCROWS,
            'SHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
            'INVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
            'HANGINGMAN': talib.CDLHANGINGMAN
        }
        
        for name, func in pattern_functions.items():
            try:
                df[f'pattern_{name}'] = func(df['open'], df['high'], df['low'], df['close'])
            except:
                df[f'pattern_{name}'] = 0
        
        # Count patterns
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        df['bullish_patterns'] = (df[pattern_cols] > 0).sum(axis=1)
        df['bearish_patterns'] = (df[pattern_cols] < 0).sum(axis=1)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات زمنية - Fixed version V2"""
        df = df.copy()
        
        # الحصول على datetime series
        if isinstance(df.index, pd.DatetimeIndex):
            # إذا كان الفهرس datetime
            time_series = df.index
        elif 'datetime' in df.columns:
            # إذا كان هناك عمود datetime
            time_series = pd.to_datetime(df['datetime'])
        elif 'time' in df.columns:
            # إذا كان هناك عمود time (timestamp)
            time_series = pd.to_datetime(df['time'], unit='s')
        else:
            logger.warning("No time information found, skipping time features")
            return df
        
        # Extract time components
        df['hour'] = time_series.hour
        df['day_of_week'] = time_series.dayofweek
        df['day_of_month'] = time_series.day
        df['month'] = time_series.month
        
        # Trading sessions
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة بنية السوق"""
        df = df.copy()
        
        # Trend identification
        for period in [20, 50, 100]:
            if f'SMA_{period}' in df.columns:
                df[f'trend_{period}'] = np.where(df['close'] > df[f'SMA_{period}'], 1, -1)
                
                # Trend strength
                if 'ATR_14' in df.columns:
                    df[f'trend_strength_{period}'] = abs(df['close'] - df[f'SMA_{period}']) / (df['ATR_14'] + 1e-10)
        
        # Market regime
        if 'ATR_14' in df.columns:
            try:
                df['volatility_regime'] = pd.qcut(df['ATR_14'].dropna(), q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                df['volatility_regime'] = df['volatility_regime'].map({'low': 0, 'medium': 1, 'high': 2})
            except:
                df['volatility_regime'] = 0
                
        if 'volume' in df.columns:
            try:
                df['volume_regime'] = pd.qcut(df['volume'].dropna(), q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                df['volume_regime'] = df['volume_regime'].map({'low': 0, 'medium': 1, 'high': 2})
            except:
                df['volume_regime'] = 0
        
        # Pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['R1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['S1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['R2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['S2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001) -> pd.DataFrame:
        """إضافة متغير الهدف"""
        df = df.copy()
        
        # Future return
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Binary classification
        df['target'] = np.where(df['future_return'] > threshold, 1, 
                               np.where(df['future_return'] < -threshold, -1, 0))
        
        # Multi-class target
        df['target_3class'] = df['target']
        
        # Remove last rows without target
        df = df[:-lookahead]
        
        return df
    
    def create_features(self, df: pd.DataFrame, target_config: Optional[Dict] = None) -> pd.DataFrame:
        """إنشاء جميع الميزات"""
        logger.info(f"Starting feature engineering for {len(df)} rows")
        
        # Add all features
        df = self.add_technical_indicators(df)
        df = self.add_price_features(df)
        df = self.add_pattern_features(df)
        df = self.add_time_features(df)
        df = self.add_market_structure(df)
        
        # Add target variable
        if target_config:
            df = self.add_target_variable(df, **target_config)
        
        # Remove NaN values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
        
        # Feature selection (remove highly correlated features)
        df = self._remove_correlated_features(df)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """إزالة الميزات المترابطة بشدة"""
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['time', 'target', 'target_3class']]
        
        if len(numeric_cols) < 2:
            return df
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)
        
        return df