import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
from loguru import logger
import json
from scipy import stats


class FeatureEngineer:
    """استخراج المؤشرات الفنية والميزات للتعلم الآلي"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.add("logs/feature_engineer.log", rotation="1 day", retention="30 days")
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة المؤشرات الفنية"""
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        # ATR
        df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['adx_pos'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_pos()
        df['adx_neg'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_neg()
        
        # Volume indicators
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            df['volume_obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        logger.info(f"Added {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'time', 'symbol', 'timeframe']])} technical indicators")
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات السعر"""
        df = df.copy()
        
        # Price changes
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # High-Low percentage
        df['hl_pct'] = (df['high'] - df['low']) / df['close']
        df['co_pct'] = (df['close'] - df['open']) / df['open']
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Support and Resistance
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['res_distance'] = (df['resistance_20'] - df['close']) / df['close']
        df['sup_distance'] = (df['close'] - df['support_20']) / df['close']
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة أنماط الشموع"""
        df = df.copy()
        
        # Candlestick patterns
        df['doji'] = ta.volatility.keltner_channel_hband_indicator(
            df['high'], df['low'], df['close'], window=20, window_atr=10
        ).astype(int)
        
        # Hammer pattern
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)).astype(int)
        
        # Engulfing pattern
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات زمنية"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_month'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        
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
        
        return df
    
    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات هيكل السوق"""
        df = df.copy()
        
        # Trend identification
        df['trend_20'] = np.where(df['sma_20'] > df['sma_20'].shift(1), 1, -1)
        df['trend_50'] = np.where(df['sma_50'] > df['sma_50'].shift(1), 1, -1)
        
        # Higher highs and lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Swing points
        df['swing_high'] = ((df['high'] > df['high'].shift(1)) & 
                           (df['high'] > df['high'].shift(-1))).astype(int)
        df['swing_low'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['low'] < df['low'].shift(-1))).astype(int)
        
        # Market regime
        volatility = df['close'].rolling(window=20).std()
        df['high_volatility'] = (volatility > volatility.rolling(window=50).mean()).astype(int)
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, target_type: str = 'classification', 
                          lookahead: int = 5, threshold: float = 0.001) -> pd.DataFrame:
        """إضافة متغير الهدف للتنبؤ"""
        df = df.copy()
        
        if target_type == 'classification':
            # Binary classification: 1 for up, 0 for down
            future_return = df['close'].shift(-lookahead) / df['close'] - 1
            df['target'] = (future_return > threshold).astype(int)
            
            # Multi-class classification
            df['target_3class'] = pd.cut(future_return, 
                                        bins=[-np.inf, -threshold, threshold, np.inf],
                                        labels=[0, 1, 2])  # 0: down, 1: neutral, 2: up
        
        elif target_type == 'regression':
            # Regression: predict actual return
            df['target'] = df['close'].shift(-lookahead) / df['close'] - 1
        
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
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """حساب أهمية الميزات"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['time', 'symbol', 'timeframe', target_col, 'target_3class']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    # مثال على الاستخدام
    engineer = FeatureEngineer()
    
    # Load sample data
    # df = pd.read_csv("data/sample_data.csv")
    # df['time'] = pd.to_datetime(df['time'])
    
    # Create features
    # df_features = engineer.create_features(df, target_config={'lookahead': 5, 'threshold': 0.001})
    
    # Get feature importance
    # importance = engineer.get_feature_importance(df_features)
    # print(importance.head(20))