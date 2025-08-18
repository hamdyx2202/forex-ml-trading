#!/usr/bin/env python3
"""
🚀 Production Server for Forex ML Trading
📊 Complete system without MT5 dependency
🌐 Server: 69.62.121.53:5000
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import sqlite3
import joblib
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('production_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global system instance
class UnifiedSystemServer:
    def __init__(self):
        self.historical_db = './data/forex_ml.db'
        self.trading_db = './trading_performance.db'
        self.unified_db = './unified_forex_system.db'
        self.models = {}
        self.model_performance = {}
        self.last_train_time = {}
        self.min_confidence = 0.65
        self.feature_importance = {}
        self.current_features = []
        
        logger.info("🚀 Initializing Unified System Server")
        self.init_databases()
        self.load_existing_models()
        
    def init_databases(self):
        """Initialize databases"""
        try:
            # Create unified DB if not exists
            if not os.path.exists(self.unified_db):
                conn = sqlite3.connect(self.unified_db)
                cursor = conn.cursor()
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    feature_name TEXT,
                    importance_score REAL,
                    last_updated TIMESTAMP
                )''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_key TEXT,
                    training_date TIMESTAMP,
                    historical_accuracy REAL,
                    total_trades INTEGER
                )''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS profitable_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE,
                    pattern_description TEXT,
                    feature_conditions TEXT,
                    avg_profit_pips REAL,
                    occurrences INTEGER,
                    last_seen TIMESTAMP
                )''')
                
                conn.commit()
                conn.close()
            
            logger.info("✅ Databases initialized")
            
        except Exception as e:
            logger.error(f"Database init error: {str(e)}")
            
    def load_existing_models(self):
        """Load existing models"""
        model_dir = 'unified_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        for file in model_files:
            try:
                model_path = os.path.join(model_dir, file)
                model_data = joblib.load(model_path)
                model_key = file.replace('_unified.pkl', '')
                self.models[model_key] = model_data
                logger.info(f"✅ Loaded model: {model_key}")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                
        logger.info(f"📊 Total models loaded: {len(self.models)}")
        
    def load_historical_data(self, symbol, timeframe, limit=50000):
        """Load historical data from database"""
        try:
            conn = sqlite3.connect(self.historical_db)
            
            # Map timeframes
            tf_map = {
                'M5': 'PERIOD_M5',
                'M15': 'PERIOD_M15',
                'M30': 'PERIOD_M30',
                'H1': 'PERIOD_H1',
                'H4': 'PERIOD_H4'
            }
            
            db_timeframe = tf_map.get(timeframe, timeframe)
            
            query = f"""
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = ? AND (timeframe = ? OR timeframe = ?)
            ORDER BY time DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, db_timeframe, limit))
            conn.close()
            
            if len(df) == 0:
                return None
                
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            logger.info(f"✅ Loaded {len(df)} historical records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def calculate_features(self, df):
        """Calculate all features"""
        features = pd.DataFrame(index=df.index)
        
        # Basic features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        features['rsi_28'] = 100 - (100 / (1 + rs.rolling(28).mean()))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean()
        features['natr_14'] = (features['atr_14'] / df['close']) * 100
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (2 * std_20)
        features['bb_lower'] = sma_20 - (2 * std_20)
        features['bb_middle'] = sma_20
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_width'] + 0.0001)
        
        # ADX (simplified)
        features['adx_14'] = features['atr_14'].rolling(14).mean() / df['close'].rolling(14).std()
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
        # Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        features['is_tokyo'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        
        # Pattern features
        features['body_size'] = np.abs(df['close'] - df['open'])
        features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        features['shadow_body_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / (features['body_size'] + 0.0001)
        
        # Statistical features
        for window in [10, 20, 50]:
            features[f'rolling_std_{window}'] = df['close'].rolling(window).std()
            features[f'rolling_skew_{window}'] = df['close'].rolling(window).skew()
            features[f'rolling_kurt_{window}'] = df['close'].rolling(window).kurt()
            
        # Clean data
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        self.current_features = features.columns.tolist()
        
        return features
        
    def train_model(self, symbol, timeframe):
        """Train a model for symbol/timeframe"""
        logger.info(f"🧠 Training model for {symbol} {timeframe}...")
        
        # Load data
        df = self.load_historical_data(symbol, timeframe)
        if df is None or len(df) < 1000:
            logger.error(f"Not enough data for {symbol} {timeframe}")
            return None
            
        # Calculate features
        features = self.calculate_features(df)
        
        # Create targets
        targets = self.create_targets(df)
        
        # Align data
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets[:min_len]
        
        # Remove NaN
        start_idx = 100
        features = features.iloc[start_idx:]
        targets = targets[start_idx:]
        
        # Train simple model
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"✅ Model trained with {accuracy:.2%} accuracy")
        
        # Save
        model_key = f"{symbol}_{timeframe}"
        self.models[model_key] = {
            'model': model,
            'scaler': scaler,
            'features': features.columns.tolist(),
            'accuracy': accuracy
        }
        
        # Save to disk
        os.makedirs('unified_models', exist_ok=True)
        joblib.dump(self.models[model_key], f'unified_models/{model_key}_unified.pkl')
        
        return self.models[model_key]
        
    def create_targets(self, df, min_pips=15, future_candles=20):
        """Create targets"""
        targets = []
        
        for i in range(len(df) - future_candles):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+future_candles+1]
            
            if len(future_prices) == 0:
                continue
                
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            pip_value = 0.01 if 'JPY' in str(df.index.name) else 0.0001
            
            long_profit = (max_price - current_price) / pip_value
            short_profit = (current_price - min_price) / pip_value
            
            if long_profit > min_pips and long_profit > short_profit * 1.2:
                target = 0  # Buy
            elif short_profit > min_pips and short_profit > long_profit * 1.2:
                target = 1  # Sell
            else:
                target = 2  # Hold
                
            targets.append(target)
            
        targets.extend([2] * future_candles)
        
        return np.array(targets)
        
    def predict(self, symbol, timeframe, features):
        """Make prediction"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            # Train if not exists
            self.train_model(symbol, timeframe)
            
        if model_key not in self.models:
            return None, 0
            
        model_data = self.models[model_key]
        
        # Scale features
        features_scaled = model_data['scaler'].transform(features)
        
        # Predict
        prediction = model_data['model'].predict(features_scaled)[0]
        proba = model_data['model'].predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        return prediction, confidence

# Create system instance
system = UnifiedSystemServer()

@app.route('/status', methods=['GET'])
def status():
    """Server status"""
    return jsonify({
        'status': 'running',
        'server': '69.62.121.53:5000',
        'models_loaded': len(system.models),
        'version': '3.0-production'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        symbol = data['symbol']
        timeframe = data['timeframe']
        candles = data['candles']
        
        logger.info(f"📊 Prediction request: {symbol} {timeframe}")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Calculate features
        features = system.calculate_features(df)
        latest_features = features.iloc[-1:].copy()
        
        # Predict
        prediction, confidence = system.predict(symbol, timeframe, latest_features)
        
        # Generate signal
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
            
        # Calculate SL/TP
        atr = features.get('atr_14', pd.Series([50 * pip_value])).iloc[-1]
        sl_pips = max(min(atr / pip_value * 1.5, 100), 20)
        tp1_pips = sl_pips * 2.0
        tp2_pips = sl_pips * 3.0
        
        if action == 'BUY':
            sl_price = current_price - (sl_pips * pip_value)
            tp1_price = current_price + (tp1_pips * pip_value)
            tp2_price = current_price + (tp2_pips * pip_value)
        elif action == 'SELL':
            sl_price = current_price + (sl_pips * pip_value)
            tp1_price = current_price - (tp1_pips * pip_value)
            tp2_price = current_price - (tp2_pips * pip_value)
        else:
            sl_price = tp1_price = tp2_price = current_price
            
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_pips),
            'tp1_pips': float(tp1_pips),
            'tp2_pips': float(tp2_pips),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ {action} signal with {confidence:.1%} confidence")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/trade_result', methods=['POST'])
def trade_result():
    """Trade result endpoint"""
    try:
        data = request.json
        logger.info(f"📈 Trade result: {data['symbol']} - {data['result']}")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain endpoint"""
    try:
        data = request.json
        symbol = data.get('symbol', 'EURUSDm')
        timeframe = data.get('timeframe', 'M15')
        
        logger.info(f"🔄 Retraining {symbol} {timeframe}")
        system.train_model(symbol, timeframe)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML PRODUCTION SERVER")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("📊 Complete system ready")
    logger.info("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()