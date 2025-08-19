#!/usr/bin/env python3
"""
🚀 Force Train Models - تدريب إجباري للنماذج
📊 يتخطى فحوصات الجودة ويدرب بالبيانات المتاحة
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_basic_features(df):
    """حساب الميزات الأساسية"""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    features['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    features['bb_upper'] = sma + (std * 2)
    features['bb_lower'] = sma - (std * 2)
    features['bb_width'] = features['bb_upper'] - features['bb_lower']
    
    # Volume features
    features['volume_sma'] = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma']
    
    # Price position
    features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Volatility
    features['volatility'] = df['close'].rolling(20).std()
    
    # Support/Resistance
    features['resistance'] = df['high'].rolling(20).max()
    features['support'] = df['low'].rolling(20).min()
    features['sr_ratio'] = (df['close'] - features['support']) / (features['resistance'] - features['support'])
    
    return features

def prepare_data_for_training(symbol, timeframe, min_samples=1000):
    """تحضير البيانات للتدريب"""
    try:
        conn = sqlite3.connect('./data/forex_ml.db')
        
        # جلب البيانات
        query = f"""
        SELECT time, open, high, low, close, volume 
        FROM price_data 
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        ORDER BY time
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < min_samples:
            logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} samples")
            return None, None
        
        logger.info(f"Loaded {len(df)} samples for {symbol} {timeframe}")
        
        # تحويل الوقت
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # حساب الميزات
        features = calculate_basic_features(df)
        
        # إنشاء الهدف (1 للصعود، 0 للهبوط)
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # حذف الصفوف الفارغة
        features = features.dropna()
        
        if len(features) < min_samples:
            return None, None
        
        # فصل الميزات والهدف
        X = features.drop('target', axis=1)
        y = features['target']
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None

def train_models(symbol, timeframe):
    """تدريب النماذج"""
    logger.info(f"\n🤖 Training models for {symbol} {timeframe}...")
    
    # تحضير البيانات
    X, y = prepare_data_for_training(symbol, timeframe, min_samples=500)
    
    if X is None or y is None:
        logger.error("Failed to prepare data")
        return False
    
    try:
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تحجيم البيانات
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # إنشاء مجلد النماذج
        os.makedirs('./trained_models', exist_ok=True)
        
        # النماذج
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'neural_network': MLPClassifier(hidden_layers=(100, 50), max_iter=500, random_state=42)
        }
        
        trained_count = 0
        
        for model_name, model in models.items():
            try:
                logger.info(f"   Training {model_name}...")
                
                # تدريب
                model.fit(X_train_scaled, y_train)
                
                # تقييم
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                logger.info(f"   ✅ {model_name} - Accuracy: {accuracy:.2%}")
                
                # حفظ النموذج
                model_path = f'./trained_models/{symbol}_{timeframe}_{model_name}_forced.pkl'
                joblib.dump(model, model_path)
                trained_count += 1
                
            except Exception as e:
                logger.error(f"   ❌ Error training {model_name}: {e}")
        
        # حفظ الscaler
        if trained_count > 0:
            scaler_path = f'./trained_models/{symbol}_{timeframe}_scaler_forced.pkl'
            joblib.dump(scaler, scaler_path)
            logger.info(f"✅ Saved {trained_count} models for {symbol} {timeframe}")
            return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        
    return False

def main():
    logger.info("="*60)
    logger.info("🚀 Force Training Models")
    logger.info("="*60)
    
    # الأزواج والأطر الزمنية
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'XAUUSD']
    timeframes = ['M15', 'H1']
    
    total_success = 0
    total_failed = 0
    
    for symbol in pairs:
        for timeframe in timeframes:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {symbol} {timeframe}")
            
            if train_models(symbol, timeframe):
                total_success += 1
            else:
                total_failed += 1
    
    # التقرير النهائي
    logger.info("\n" + "="*60)
    logger.info("🏁 Training Complete")
    logger.info(f"✅ Successful: {total_success}")
    logger.info(f"❌ Failed: {total_failed}")
    
    # عرض النماذج المحفوظة
    if os.path.exists('./trained_models'):
        models = [f for f in os.listdir('./trained_models') if f.endswith('.pkl') and 'scaler' not in f]
        logger.info(f"\n🤖 Total models saved: {len(models)}")
        
        if models:
            logger.info("\nSaved models:")
            for model in models[:10]:
                logger.info(f"   - {model}")

if __name__ == "__main__":
    main()