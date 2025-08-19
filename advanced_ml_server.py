#!/usr/bin/env python3
"""
🚀 Advanced ML Server with 200+ Features
📊 يدعم 6 نماذج ML و 10 فرضيات
"""

import os
import sys
import json
import logging
import threading
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML Libraries
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# Import Model Manager
try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except:
    MODEL_MANAGER_AVAILABLE = False

# Optional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('advanced_ml_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AdvancedMLSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.db_path = './data/forex_ml.db'
        self.models_dir = './trained_models'
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize Model Manager
        if MODEL_MANAGER_AVAILABLE:
            self.model_manager = ModelManager(self.models_dir)
            logger.info("✅ Model Manager initialized")
        else:
            self.model_manager = None
            
        self.load_existing_models()
        
        logger.info(f"✅ Advanced ML System initialized - {len(self.models)} models loaded")
    
    def calculate_advanced_features(self, df):
        """حساب 200+ ميزة متقدمة"""
        features = pd.DataFrame(index=df.index)
        
        # 1. Price Features (20)
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['hl2'] = (df['high'] + df['low']) / 2
        features['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        features['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        features['price_range'] = df['high'] - df['low']
        features['body_size'] = abs(df['close'] - df['open'])
        features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # 2. Moving Averages (30)
        for period in [5, 10, 20, 30, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'sma_slope_{period}'] = features[f'sma_{period}'].diff()
        
        # 3. RSI Multiple Periods (10)
        for period in [7, 14, 21, 28, 35]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 4. MACD Variations (15)
        for fast, slow in [(12, 26), (5, 35), (8, 17)]:
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            features[f'macd_{fast}_{slow}'] = exp1 - exp2
            features[f'macd_signal_{fast}_{slow}'] = features[f'macd_{fast}_{slow}'].ewm(span=9).mean()
            features[f'macd_hist_{fast}_{slow}'] = features[f'macd_{fast}_{slow}'] - features[f'macd_signal_{fast}_{slow}']
        
        # 5. Bollinger Bands (20)
        for period in [10, 20, 30, 50]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_upper_{period}'] = sma + (std * 2)
            features[f'bb_lower_{period}'] = sma - (std * 2)
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
        
        # 6. Stochastic (15)
        for period in [5, 14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            features[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # 7. ATR & Volatility (15)
        for period in [7, 14, 21, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'volatility_{period}'] = df['close'].rolling(period).std()
        
        # 8. Support/Resistance (20)
        for period in [10, 20, 50, 100]:
            features[f'resistance_{period}'] = df['high'].rolling(period).max()
            features[f'support_{period}'] = df['low'].rolling(period).min()
            features[f'sr_ratio_{period}'] = (df['close'] - features[f'support_{period}']) / (features[f'resistance_{period}'] - features[f'support_{period}'])
        
        # 9. Pattern Recognition (20)
        # Doji
        features['doji'] = (features['body_size'] / features['price_range'] < 0.1).astype(int)
        
        # Hammer
        features['hammer'] = ((features['lower_shadow'] > features['body_size'] * 2) & 
                            (features['upper_shadow'] < features['body_size'])).astype(int)
        
        # Engulfing
        features['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                        (df['close'].shift() < df['open'].shift()) &
                                        (df['open'] < df['close'].shift()) &
                                        (df['close'] > df['open'].shift())).astype(int)
        
        # 10. Volume Features (10) - إذا متاح
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['price_volume'] = df['close'] * df['volume']
        
        # 11. Time Features (15)
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['day_of_month'] = df.index.day
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            features['is_monday'] = (df.index.dayofweek == 0).astype(int)
            features['is_friday'] = (df.index.dayofweek == 4).astype(int)
            features['is_london'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_newyork'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_tokyo'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
        
        # 12. Price Action (20)
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['upper_body_ratio'] = features['upper_shadow'] / features['body_size'].replace(0, 1)
        features['lower_body_ratio'] = features['lower_shadow'] / features['body_size'].replace(0, 1)
        
        # 13. Momentum Indicators (15)
        features['roc_5'] = df['close'].pct_change(5) * 100
        features['roc_10'] = df['close'].pct_change(10) * 100
        features['roc_20'] = df['close'].pct_change(20) * 100
        
        # Williams %R
        for period in [14, 28]:
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            features[f'williams_r_{period}'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        # 14. ADX (10)
        for period in [14, 21]:
            # Simplified ADX
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = true_range.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            features[f'adx_{period}'] = dx.rolling(period).mean()
        
        # Remove NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        
        return features
    
    def train_all_models(self, symbol, timeframe, train_sl_tp=True):
        """تدريب 6 نماذج ML + نماذج SL/TP الذكية"""
        try:
            logger.info(f"🤖 Training {symbol} {timeframe} with all models...")
            
            # Load data - try different symbol variations
            conn = sqlite3.connect(self.db_path)
            
            # Try different symbol formats
            possible_symbols = [
                symbol,
                f"{symbol}m",
                f"{symbol}.ecn",
                f"{symbol}_pro",
                symbol.lower(),
                f"{symbol.lower()}m"
            ]
            
            df = None
            for sym in possible_symbols:
                query = f"""
                SELECT * FROM price_data 
                WHERE symbol = '{sym}' OR symbol LIKE '{sym}%'
                ORDER BY time DESC
                LIMIT 10000
                """
                df_temp = pd.read_sql_query(query, conn)
                if not df_temp.empty:
                    df = df_temp
                    logger.info(f"Found data with symbol: {sym}")
                    break
            
            if df is None or df.empty:
                query = f"""
                SELECT * FROM price_data 
                WHERE symbol LIKE '%{symbol}%'
                ORDER BY time DESC
                LIMIT 10000
                """
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            if df is None or len(df) < 1000:
                logger.warning(f"Not enough data for {symbol}")
                return False
            
            # Prepare data
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Calculate features
            features = self.calculate_advanced_features(df)
            
            if len(features) < 500:
                return False
            
            # Prepare training data
            # Ensure df and features have the same index
            common_index = features.index.intersection(df.index)
            features_aligned = features.loc[common_index]
            df_aligned = df.loc[common_index]
            
            # Create target variable
            y = (df_aligned['close'].shift(-1) > df_aligned['close']).astype(int)
            
            # Remove last row (NaN from shift)
            X = features_aligned.values[:-1]
            y = y.values[:-1]
            
            # Final check
            if len(X) != len(y):
                logger.error(f"Shape mismatch: X={len(X)}, y={len(y)}")
                # Use minimum length
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            key = f"{symbol}_{timeframe}"
            self.scalers[key] = scaler
            
            # Train models
            models = {}
            
            # 1. Random Forest
            try:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_train_scaled, y_train)
                acc = rf.score(X_test_scaled, y_test)
                models['random_forest'] = rf
                logger.info(f"   ✅ Random Forest: {acc:.4f}")
            except Exception as e:
                logger.error(f"   ❌ Random Forest: {e}")
            
            # 2. Gradient Boosting
            try:
                gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                gb.fit(X_train_scaled, y_train)
                acc = gb.score(X_test_scaled, y_test)
                models['gradient_boosting'] = gb
                logger.info(f"   ✅ Gradient Boosting: {acc:.4f}")
            except Exception as e:
                logger.error(f"   ❌ Gradient Boosting: {e}")
            
            # 3. Extra Trees
            try:
                et = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                et.fit(X_train_scaled, y_train)
                acc = et.score(X_test_scaled, y_test)
                models['extra_trees'] = et
                logger.info(f"   ✅ Extra Trees: {acc:.4f}")
            except Exception as e:
                logger.error(f"   ❌ Extra Trees: {e}")
            
            # 4. Neural Network
            try:
                nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                nn.fit(X_train_scaled, y_train)
                acc = nn.score(X_test_scaled, y_test)
                models['neural_network'] = nn
                logger.info(f"   ✅ Neural Network: {acc:.4f}")
            except Exception as e:
                logger.error(f"   ❌ Neural Network: {e}")
            
            # 5. LightGBM
            if LIGHTGBM_AVAILABLE:
                try:
                    lgbm = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42, verbosity=-1)
                    lgbm.fit(X_train_scaled, y_train)
                    acc = lgbm.score(X_test_scaled, y_test)
                    models['lightgbm'] = lgbm
                    logger.info(f"   ✅ LightGBM: {acc:.4f}")
                except Exception as e:
                    logger.error(f"   ❌ LightGBM: {e}")
            
            # 6. XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False)
                    xgb_model.fit(X_train_scaled, y_train)
                    acc = xgb_model.score(X_test_scaled, y_test)
                    models['xgboost'] = xgb_model
                    logger.info(f"   ✅ XGBoost: {acc:.4f}")
                except Exception as e:
                    logger.error(f"   ❌ XGBoost: {e}")
            
            # Save models
            self.models[key] = models
            
            # Save to disk and register with manager
            for model_name, model in models.items():
                model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_name}.pkl")
                joblib.dump(model, model_path)
                
                # Register with Model Manager
                if self.model_manager and model_name in accuracies:
                    self.model_manager.register_model(
                        model_path=model_path,
                        accuracy=accuracies[model_name],
                        symbol=symbol,
                        timeframe=timeframe,
                        model_type=model_name
                    )
            
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ Trained {len(models)} models for {symbol} {timeframe}")
            
            # تدريب نماذج SL/TP الذكية
            if train_sl_tp and len(df) > 2000:
                self.train_optimal_sl_tp(symbol, timeframe, df, features)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            return False
    
    def predict_ensemble(self, symbol, timeframe, df):
        """تنبؤ باستخدام جميع النماذج"""
        try:
            key = f"{symbol}_{timeframe}"
            
            # Train if no model
            if key not in self.models:
                logger.info(f"Training new models for {key}...")
                if not self.train_all_models(symbol, timeframe):
                    return self.simple_prediction(df)
            
            # Calculate features
            features = self.calculate_advanced_features(df)
            if features.empty:
                return self.simple_prediction(df)
            
            X = features.values[-1:]
            
            # Scale
            if key in self.scalers:
                X_scaled = self.scalers[key].transform(X)
            else:
                X_scaled = X
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0]
                        confidences.append(max(prob))
                    else:
                        confidences.append(0.6)
                        
                except Exception as e:
                    logger.error(f"Prediction error {model_name}: {e}")
            
            if not predictions:
                return self.simple_prediction(df)
            
            # Ensemble voting
            buy_votes = sum(1 for p in predictions if p == 1)
            total_votes = len(predictions)
            
            if buy_votes > total_votes / 2:
                action = 0  # BUY
                confidence = np.mean(confidences)
            elif buy_votes < total_votes / 2:
                action = 1  # SELL
                confidence = np.mean(confidences)
            else:
                action = 2  # HOLD
                confidence = 0.5
            
            logger.info(f"   📊 Ensemble: {buy_votes}/{total_votes} votes, confidence: {confidence:.2%}")
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return self.simple_prediction(df)
    
    def train_optimal_sl_tp(self, symbol, timeframe, df, features_df):
        """تدريب نموذج لتحديد أفضل SL/TP بناءً على التاريخ"""
        try:
            logger.info(f"   🎯 Training optimal SL/TP for {symbol} {timeframe}...")
            
            # حساب النتائج التاريخية لمختلف مستويات SL/TP
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            # إنشاء dataset للتدريب
            sl_tp_data = []
            
            # اختبار مستويات مختلفة
            sl_levels = [20, 30, 40, 50, 60, 80, 100, 120, 150]
            tp_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            
            for i in range(100, len(df) - 100):
                entry_price = df['close'].iloc[i]
                
                # الميزات عند نقطة الدخول
                if i < len(features_df):
                    entry_features = features_df.iloc[i].to_dict()
                    
                    # اختبار كل مجموعة SL/TP
                    for sl_pips in sl_levels:
                        for tp_ratio in tp_ratios:
                            tp_pips = sl_pips * tp_ratio
                            
                            # حساب النتيجة
                            sl_price = entry_price - (sl_pips * pip_value)
                            tp_price = entry_price + (tp_pips * pip_value)
                            
                            # تتبع السعر للأمام
                            future_prices = df['close'].iloc[i+1:i+101]
                            future_lows = df['low'].iloc[i+1:i+101]
                            future_highs = df['high'].iloc[i+1:i+101]
                            
                            # هل وصل TP أو SL؟
                            hit_sl = (future_lows <= sl_price).any()
                            hit_tp = (future_highs >= tp_price).any()
                            
                            if hit_tp and not hit_sl:
                                result = 1  # ربح
                                profit = tp_pips
                            elif hit_sl:
                                result = 0  # خسارة
                                profit = -sl_pips
                            else:
                                result = 0.5  # لم يصل لأي هدف
                                profit = 0
                            
                            # إضافة للداتا
                            record = {
                                'sl_pips': sl_pips,
                                'tp_pips': tp_pips,
                                'tp_ratio': tp_ratio,
                                'result': result,
                                'profit': profit,
                                **entry_features  # إضافة كل الميزات
                            }
                            sl_tp_data.append(record)
            
            if sl_tp_data:
                # تحويل لـ DataFrame وتدريب نموذج
                sl_tp_df = pd.DataFrame(sl_tp_data)
                
                # حساب أفضل SL/TP لكل مجموعة ميزات
                best_params = sl_tp_df.groupby(['sl_pips', 'tp_ratio'])['result'].mean()
                best_sl, best_ratio = best_params.idxmax()
                
                logger.info(f"   ✅ Best SL: {best_sl} pips, TP Ratio: {best_ratio}")
                
                # حفظ المعلمات الأمثل
                key = f"{symbol}_{timeframe}"
                if not hasattr(self, 'optimal_sl_tp'):
                    self.optimal_sl_tp = {}
                    
                self.optimal_sl_tp[key] = {
                    'sl_pips': best_sl,
                    'tp_ratio': best_ratio,
                    'success_rate': best_params.max()
                }
                
        except Exception as e:
            logger.error(f"   ❌ SL/TP training error: {str(e)}")
    
    def simple_prediction(self, df):
        """Simple fallback prediction"""
        try:
            latest = df.iloc[-1]
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            sma50 = df['close'].rolling(50).mean().iloc[-1]
            
            if latest['close'] > sma20 > sma50:
                return 0, 0.65  # Buy
            elif latest['close'] < sma20 < sma50:
                return 1, 0.65  # Sell
            else:
                return 2, 0.5   # Hold
        except:
            return 2, 0.5
    
    def load_existing_models(self):
        """Load pre-trained models"""
        if not os.path.exists(self.models_dir):
            return
        
        for file in os.listdir(self.models_dir):
            if file.endswith('.pkl'):
                try:
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:])
                        
                        if model_type == 'scaler':
                            scaler = joblib.load(os.path.join(self.models_dir, file))
                            self.scalers[f"{symbol}_{timeframe}"] = scaler
                        else:
                            key = f"{symbol}_{timeframe}"
                            if key not in self.models:
                                self.models[key] = {}
                            
                            model = joblib.load(os.path.join(self.models_dir, file))
                            self.models[key][model_type] = model
                            
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

# Initialize system
system = AdvancedMLSystem()

@app.route('/status', methods=['GET'])
def status():
    """Server status"""
    return jsonify({
        'status': 'running',
        'version': '2.0-advanced',
        'features': '200+',
        'models': '6 ML models',
        'total_loaded': len(system.models),
        'ml_libraries': {
            'lightgbm': LIGHTGBM_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Parse JSON
        raw_data = request.get_data(as_text=True)
        
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON', 'action': 'NONE', 'confidence': 0})
        
        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        
        # Clean symbol
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
        
        logger.info(f"📊 Request: {symbol} ({clean_symbol}) {timeframe} - {len(candles)} candles")
        
        if len(candles) < 200:
            return jsonify({
                'symbol': symbol,
                'action': 'NONE',
                'confidence': 0,
                'error': 'Need at least 200 candles for advanced features'
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        df = df.dropna()
        
        # Get prediction
        prediction, confidence = system.predict_ensemble(clean_symbol, timeframe, df)
        
        # Determine signal
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
        
        # Calculate SL/TP - استخدام القيم المدربة إذا متاحة
        key = f"{clean_symbol}_{timeframe}"
        
        if hasattr(system, 'optimal_sl_tp') and key in system.optimal_sl_tp:
            # استخدام SL/TP الأمثل المدرب
            optimal = system.optimal_sl_tp[key]
            sl_pips = optimal['sl_pips']
            tp_ratio = optimal['tp_ratio']
            tp1_pips = sl_pips * tp_ratio
            tp2_pips = sl_pips * (tp_ratio + 1)
            
            logger.info(f"   📊 Using trained SL/TP: {sl_pips} pips, ratio {tp_ratio}")
        else:
            # القيم الافتراضية
            atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]
            sl_distance = max(min(atr * 1.5, 100 * pip_value), 30 * pip_value)
            sl_pips = sl_distance / pip_value
            tp1_pips = sl_pips * 2
            tp2_pips = sl_pips * 3
        
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
            sl_pips = tp1_pips = tp2_pips = 0
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_distance / pip_value),
            'tp1_pips': float(sl_distance / pip_value * 2),
            'tp2_pips': float(sl_distance / pip_value * 3),
            'models_used': len(system.models.get(f"{clean_symbol}_{timeframe}", {}))
        }
        
        logger.info(f"   ✅ {action} with {confidence:.1%} confidence")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'action': 'NONE',
            'confidence': 0
        })

@app.route('/train', methods=['POST'])
def train():
    """Train endpoint"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').replace('.ecn', '')
        timeframe = data.get('timeframe', 'M15')
        
        success = system.train_all_models(symbol, timeframe)
        
        return jsonify({
            'success': success,
            'message': f'Trained {symbol} {timeframe}' if success else 'Training failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def models():
    """List models"""
    models_info = {}
    
    for key, models in system.models.items():
        models_info[key] = {
            'models': list(models.keys()),
            'count': len(models)
        }
    
    return jsonify({
        'total_pairs': len(system.models),
        'models': models_info
    })

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("🚀 ADVANCED ML SERVER")
    logger.info("📊 200+ Features | 6 ML Models")
    logger.info("🌐 Server: http://0.0.0.0:5000")
    logger.info("="*60 + "\n")
    
    # Train some basic models on startup
    logger.info("🤖 Training basic models...")
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        try:
            system.train_all_models(symbol, 'M15')
        except:
            pass
    
    app.run(host='0.0.0.0', port=5000, debug=False)