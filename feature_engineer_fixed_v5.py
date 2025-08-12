#!/usr/bin/env python3
"""
Feature Engineering Module - Fixed V5
هندسة الميزات - الإصدار المحسن
Fixed: DatetimeIndex.dt error and improved NaN handling
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """مهندس الميزات المحسن"""
    
    def __init__(self, min_periods_factor: float = 0.5):
        """
        min_periods_factor: Factor to reduce required periods for indicators
        0.5 means use half the standard period (e.g., SMA_100 needs only 50 bars)
        """
        self.feature_cols = []
        self.min_periods_factor = min_periods_factor
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة المؤشرات الفنية مع تحسين معالجة NaN"""
        df = df.copy()
        
        # Adjust periods based on available data
        n_bars = len(df)
        
        # Moving averages with adaptive periods
        for period in [5, 10, 20, 50, 100, 200]:
            min_period = max(2, int(period * self.min_periods_factor))
            if n_bars >= min_period:
                df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
            else:
                df[f'SMA_{period}'] = np.nan
                df[f'EMA_{period}'] = np.nan
        
        # RSI with fallback
        if n_bars >= 7:
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        else:
            df['RSI'] = 50.0  # Neutral value
        
        # MACD with minimum periods
        if n_bars >= 26:
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            df['MACD'] = df['MACD_signal'] = df['MACD_hist'] = 0.0
        
        # Bollinger Bands
        if n_bars >= 10:
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
        else:
            df['BB_upper'] = df['close'] * 1.02
            df['BB_middle'] = df['close']
            df['BB_lower'] = df['close'] * 0.98
        
        # ATR
        if n_bars >= 7:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            df['ATR'] = df['high'] - df['low']
        
        # Stochastic
        if n_bars >= 9:
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
        else:
            df['STOCH_K'] = df['STOCH_D'] = 50.0
        
        # Volume indicators
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_SMA'] = df['volume'].rolling(window=min(20, n_bars-1), min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_SMA'].replace(0, 1)
        else:
            df['volume_SMA'] = 0
            df['volume_ratio'] = 1
        
        # Smart NaN filling - forward fill then backward fill then use mean
        indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'time', 'datetime']]
        for col in indicator_cols:
            if df[col].isna().all():
                # If all NaN, use a sensible default
                if 'RSI' in col or 'STOCH' in col:
                    df[col] = 50.0
                elif 'volume' in col:
                    df[col] = 0.0
                else:
                    df[col] = df['close'].mean() if 'close' in df else 0.0
            else:
                # Forward fill, then backward fill, then use mean
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات السعر"""
        df = df.copy()
        
        # Price ratios
        df['HL_ratio'] = df['high'] / df['low'].replace(0, 1)
        df['CO_ratio'] = df['close'] / df['open'].replace(0, 1)
        
        # Price changes
        df['price_change'] = df['close'].pct_change().fillna(0)
        df['price_change_abs'] = df['price_change'].abs()
        
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, 1)
        
        # Support/Resistance
        df['dist_from_high_20'] = (df['close'] - df['high'].rolling(20, min_periods=1).max()) / df['close']
        df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20, min_periods=1).min()) / df['close']
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة أنماط الشموع"""
        df = df.copy()
        
        # Only add patterns if we have enough data
        if len(df) >= 5:
            # Candlestick patterns
            pattern_functions = [
                'CDLDOJI', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 
                'CDLENGULFING', 'CDLHARAMI', 'CDLMORNINGSTAR',
                'CDLEVENINGSTAR', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS'
            ]
            
            for pattern in pattern_functions:
                try:
                    pattern_func = getattr(talib, pattern)
                    df[f'pattern_{pattern}'] = pattern_func(
                        df['open'], df['high'], df['low'], df['close']
                    ).fillna(0)
                except:
                    df[f'pattern_{pattern}'] = 0
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات زمنية - Fixed V5"""
        df = df.copy()
        
        # Handle different datetime formats
        if 'datetime' in df.columns:
            if isinstance(df['datetime'].iloc[0], pd.Timestamp):
                time_series = df['datetime']
            else:
                time_series = pd.to_datetime(df['datetime'])
        elif isinstance(df.index, pd.DatetimeIndex):
            # If index is already DatetimeIndex, use it directly
            time_series = df.index.to_series()
        elif df.index.name == 'datetime':
            # Convert index to datetime if needed
            time_series = pd.to_datetime(df.index).to_series()
        elif 'time' in df.columns:
            # Create datetime from timestamp
            time_series = pd.to_datetime(df['time'], unit='s')
        else:
            logger.warning("No time column found, skipping time features")
            return df
        
        # Extract time components (works for both Series and DatetimeIndex)
        df['hour'] = time_series.dt.hour
        df['day_of_week'] = time_series.dt.dayofweek
        df['day_of_month'] = time_series.dt.day
        df['month'] = time_series.dt.month
        
        # Trading sessions
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة بنية السوق"""
        df = df.copy()
        
        # Trend detection
        if len(df) >= 20:
            df['trend_sma20'] = np.where(df['close'] > df['SMA_20'], 1, -1) if 'SMA_20' in df else 0
            df['trend_sma50'] = np.where(df['close'] > df['SMA_50'], 1, -1) if 'SMA_50' in df else 0
        else:
            df['trend_sma20'] = df['trend_sma50'] = 0
        
        # Higher highs and lower lows
        if len(df) >= 10:
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['higher_high_count'] = df['higher_high'].rolling(10, min_periods=1).sum()
            df['lower_low_count'] = df['lower_low'].rolling(10, min_periods=1).sum()
        else:
            df['higher_high'] = df['lower_low'] = 0
            df['higher_high_count'] = df['lower_low_count'] = 0
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=2).std().fillna(0)
        
        # Pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['R1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['S1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['R2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['S2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        # Fill NaN in market structure features
        market_cols = ['pivot', 'R1', 'S1', 'R2', 'S2']
        for col in market_cols:
            df[col] = df[col].fillna(df['close'])
        
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
        """إنشاء جميع الميزات مع معالجة محسنة لـ NaN"""
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
        
        # Smart NaN handling - keep as many rows as possible
        initial_rows = len(df)
        
        # First, try to fill NaN values intelligently
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                # Try forward fill first
                df[col] = df[col].fillna(method='ffill')
                # Then backward fill
                df[col] = df[col].fillna(method='bfill')
                # Finally use mean for any remaining NaN
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        
        # Only drop rows that still have NaN after filling
        df = df.dropna()
        
        logger.info(f"After smart NaN handling: {len(df)} rows remain (removed {initial_rows - len(df)} rows)")
        
        # Feature selection (remove highly correlated features)
        if len(df) > 0:
            corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            df = df.drop(columns=to_drop, errors='ignore')
            if to_drop:
                logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return df
    
    def prepare_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحضير البيانات للتنبؤ (بدون target)"""
        df = self.create_features(df.copy())
        
        # Remove any target-related columns if they exist
        target_cols = ['target', 'target_3class', 'target_binary', 'future_return']
        df = df.drop(columns=[col for col in target_cols if col in df.columns])
        
        return df

# Test the module
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })
    sample_data.set_index('datetime', inplace=True)
    
    # Test feature engineering
    engineer = FeatureEngineer(min_periods_factor=0.5)
    features_df = engineer.create_features(sample_data.copy())
    
    print(f"Original data: {len(sample_data)} rows")
    print(f"After feature engineering: {len(features_df)} rows")
    print(f"Number of features: {len(features_df.columns)}")
    print(f"\nFeature columns: {list(features_df.columns[:10])}...")