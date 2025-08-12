#!/usr/bin/env python3
"""
Simple training script without TA-Lib
تدريب بسيط بدون مكتبات معقدة
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from loguru import logger

# Setup logging
logger.remove()
logger.add("logs/simple_train.log", rotation="1 day")

def create_simple_features(df):
    """إنشاء مؤشرات بسيطة بدون TA-Lib"""
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Price changes
    for period in [1, 5, 10]:
        df[f'returns_{period}'] = df['close'].pct_change(period)
    
    # RSI simple
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = calculate_rsi(df['close'])
    
    # Bollinger Bands
    for period in [20]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'BB_upper_{period}'] = sma + (std * 2)
        df[f'BB_lower_{period}'] = sma - (std * 2)
        df[f'BB_position_{period}'] = (df['close'] - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'] + 1e-10)
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
    
    # High-Low features
    df['HL_ratio'] = df['high'] / (df['low'] + 1e-10)
    df['body_size'] = abs(df['close'] - df['open'])
    
    # Time features (from timestamp)
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['day_of_week'] = pd.to_datetime(df['time'], unit='s').dt.dayofweek
    
    # Target - future price direction
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = np.where(df['future_return'] > 0.001, 1, 
                           np.where(df['future_return'] < -0.001, -1, 0))
    
    return df

def train_simple_model(symbol, timeframe):
    """تدريب نموذج بسيط"""
    logger.info(f"Training {symbol} {timeframe}")
    
    # Load data
    conn = sqlite3.connect("data/forex_ml.db")
    query = """
        SELECT time, open, high, low, close, volume
        FROM price_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY time
        LIMIT 10000
    """
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    conn.close()
    
    if len(df) < 1000:
        logger.warning(f"Not enough data for {symbol} {timeframe}")
        return False
    
    # Create features
    df = create_simple_features(df)
    
    # Remove NaN
    df = df.dropna()
    
    if len(df) < 500:
        logger.warning(f"Not enough data after feature creation for {symbol} {timeframe}")
        return False
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    # Remove samples with target = 0 (no clear direction)
    mask = y != 0
    X = X[mask]
    y = y[mask]
    
    if len(y) < 100:
        logger.warning(f"Not enough samples for {symbol} {timeframe}")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Accuracy for {symbol} {timeframe}: {accuracy:.2%}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/{symbol}_{timeframe}_simple.pkl"
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'accuracy': accuracy
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return True

def main():
    """التدريب الرئيسي"""
    logger.info("="*60)
    logger.info("🚀 Simple Training Started")
    logger.info("="*60)
    
    # Get available data
    conn = sqlite3.connect("data/forex_ml.db")
    cursor = conn.execute("""
        SELECT DISTINCT symbol, timeframe, COUNT(*) as count
        FROM price_data
        GROUP BY symbol, timeframe
        HAVING count > 1000
        ORDER BY count DESC
        LIMIT 10
    """)
    
    pairs = []
    for row in cursor:
        pairs.append((row[0], row[1], row[2]))
    conn.close()
    
    logger.info(f"Found {len(pairs)} pairs with enough data")
    
    # Train models
    success = 0
    for symbol, timeframe, count in pairs:
        logger.info(f"\nTraining {symbol} {timeframe} ({count} bars)")
        if train_simple_model(symbol, timeframe):
            success += 1
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ Training completed: {success}/{len(pairs)} models")
    logger.info("="*60)

if __name__ == "__main__":
    main()