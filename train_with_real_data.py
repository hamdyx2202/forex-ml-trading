#!/usr/bin/env python3
"""
🚀 Advanced Training System with Real Data
📊 يستخدم 7.8 مليون سجل من البيانات الحقيقية
✨ جميع الميزات المتقدمة والنماذج
"""

import os
import sys
import gc
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available, using basic indicators")

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training_real_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealDataTrainingSystem:
    """نظام التدريب باستخدام البيانات الحقيقية"""
    
    def __init__(self):
        logger.info("="*100)
        logger.info("🚀 Advanced Forex ML System - Real Data")
        logger.info("📊 Database: ./data/forex_ml.db (7.8M records)")
        logger.info("🤖 6 ML Models | 200+ Features | Multiple Targets")
        logger.info("="*100)
        
        # Database path
        self.db_path = './data/forex_ml.db'
        
        # Get available symbols and timeframes
        self.symbols, self.timeframes = self.get_available_data()
        
        # ML Models configuration
        self.model_configs = {
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'num_leaves': 63,
                    'min_child_samples': 30,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 30,
                    'min_samples_leaf': 15,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'min_samples_split': 30,
                    'random_state': 42
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 30,
                    'min_samples_leaf': 15,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': 42
                }
            }
        }
        
        # Target configurations
        self.target_configs = [
            {'name': 'scalp', 'candles': 10, 'min_pips': 10},
            {'name': 'intraday', 'candles': 30, 'min_pips': 20},
            {'name': 'swing', 'candles': 120, 'min_pips': 40}
        ]
        
        # Results storage
        self.results = {}
        
    def get_available_data(self):
        """الحصول على الرموز والأطر الزمنية المتاحة"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # الحصول على الرموز الفريدة
            symbols_query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
            symbols = pd.read_sql_query(symbols_query, conn)['symbol'].tolist()
            
            # الحصول على الأطر الزمنية
            timeframes_query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
            timeframes = pd.read_sql_query(timeframes_query, conn)['timeframe'].tolist()
            
            conn.close()
            
            logger.info(f"✅ Found {len(symbols)} symbols: {symbols}")
            logger.info(f"✅ Found {len(timeframes)} timeframes: {timeframes}")
            
            return symbols, timeframes
            
        except Exception as e:
            logger.error(f"Error getting available data: {str(e)}")
            return ['EURUSDm'], ['PERIOD_M15']
            
    def load_data(self, symbol, timeframe, limit=50000):
        """تحميل البيانات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            ORDER BY time DESC
            LIMIT {limit}
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) == 0:
                return None
                
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            # Rename volume column
            df.rename(columns={'volume': 'tick_volume'}, inplace=True)
            
            logger.info(f"✅ Loaded {len(df)} records for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def calculate_features(self, df):
        """حساب الميزات المتقدمة"""
        features = pd.DataFrame(index=df.index)
        
        # 1. Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # 2. Moving averages
        for period in [5, 10, 20, 50, 100]:
            if TALIB_AVAILABLE:
                features[f'sma_{period}'] = talib.SMA(df['close'], period)
                features[f'ema_{period}'] = talib.EMA(df['close'], period)
            else:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                
            features[f'price_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            
        # 3. Technical indicators
        if TALIB_AVAILABLE:
            # Momentum
            features['rsi_14'] = talib.RSI(df['close'], 14)
            features['rsi_28'] = talib.RSI(df['close'], 28)
            
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            
            features['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], 14)
            features['mom_10'] = talib.MOM(df['close'], 10)
            features['roc_10'] = talib.ROC(df['close'], 10)
            
            # Volatility
            features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            features['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], 14)
            
            # Trend
            features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], 14)
            features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
            features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = upper - lower
            features['bb_position'] = (df['close'] - lower) / (upper - lower + 0.0001)
            
        else:
            # Basic indicators without TA-Lib
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
            
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
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (2 * std_20)
            features['bb_lower'] = sma_20 - (2 * std_20)
            features['bb_middle'] = sma_20
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_width'] + 0.0001)
            
        # 4. Volume features
        if 'tick_volume' in df.columns:
            features['volume_sma'] = df['tick_volume'].rolling(10).mean()
            features['volume_ratio'] = df['tick_volume'] / features['volume_sma']
            
        # 5. Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        features['is_tokyo'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        
        # 6. Price patterns
        features['body_size'] = np.abs(df['close'] - df['open'])
        features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        features['shadow_body_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / (features['body_size'] + 0.0001)
        
        # 7. Statistical features
        for window in [10, 20, 50]:
            features[f'rolling_std_{window}'] = df['close'].rolling(window).std()
            features[f'rolling_skew_{window}'] = df['close'].rolling(window).skew()
            features[f'rolling_kurt_{window}'] = df['close'].rolling(window).kurt()
            
        # Clean data
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
        
    def create_targets(self, df, config):
        """إنشاء الأهداف للتدريب"""
        targets = []
        
        for i in range(len(df) - config['candles']):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+config['candles']+1]
            
            if len(future_prices) == 0:
                continue
                
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            # Calculate in pips (for forex, 1 pip = 0.0001 for most pairs)
            if 'JPY' in df.index.name if hasattr(df.index, 'name') else '':
                pip_value = 0.01
            else:
                pip_value = 0.0001
                
            long_profit = (max_price - current_price) / pip_value
            short_profit = (current_price - min_price) / pip_value
            
            if long_profit > config['min_pips'] and long_profit > short_profit:
                target = 0  # Buy
            elif short_profit > config['min_pips']:
                target = 1  # Sell
            else:
                target = 2  # Hold
                
            targets.append(target)
            
        # Pad the end
        targets.extend([2] * config['candles'])
        
        return np.array(targets)
        
    def train_models(self, X_train, X_test, y_train, y_test, model_key):
        """تدريب جميع النماذج"""
        results = {}
        models = {}
        
        logger.info(f"  Training 6 ML models...")
        
        for name, config in self.model_configs.items():
            try:
                logger.info(f"    Training {name}...")
                
                model = config['model'](**config['params'])
                
                if name in ['lightgbm', 'xgboost']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = accuracy
                models[name] = model
                
                logger.info(f"      {name}: {accuracy:.2%}")
                
            except Exception as e:
                logger.error(f"      Error in {name}: {str(e)}")
                
        return models, results
        
    def train_symbol_timeframe(self, symbol, timeframe):
        """تدريب رمز وإطار زمني واحد"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol} {timeframe}")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_data(symbol, timeframe)
        
        if df is None or len(df) < 1000:
            logger.warning(f"Not enough data for {symbol} {timeframe}")
            return None
            
        # Calculate features
        logger.info("Calculating features...")
        features = self.calculate_features(df)
        
        all_results = {}
        
        # Train for each target configuration
        for target_config in self.target_configs:
            logger.info(f"\n  Strategy: {target_config['name']}")
            
            # Create targets
            targets = self.create_targets(df, target_config)
            
            # Align data
            min_len = min(len(features), len(targets))
            features_aligned = features.iloc[:min_len]
            targets_aligned = targets[:min_len]
            
            # Remove NaN rows
            start_idx = 100
            features_final = features_aligned.iloc[start_idx:]
            targets_final = targets_aligned[start_idx:]
            
            # Check target distribution
            unique, counts = np.unique(targets_final, return_counts=True)
            logger.info(f"  Target distribution: Buy={counts[0]}, Sell={counts[1]}, Hold={counts[2]}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_final, targets_final,
                test_size=0.2, random_state=42,
                stratify=targets_final
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            model_key = f"{symbol}_{timeframe}_{target_config['name']}"
            models, results = self.train_models(
                X_train_scaled, X_test_scaled,
                y_train, y_test,
                model_key
            )
            
            # Ensemble accuracy
            if models:
                ensemble_preds = []
                for model in models.values():
                    ensemble_preds.append(model.predict(X_test_scaled))
                    
                ensemble_pred = stats.mode(np.array(ensemble_preds), axis=0)[0].flatten()
                ensemble_acc = accuracy_score(y_test, ensemble_pred)
                
                logger.info(f"  ✨ Ensemble accuracy: {ensemble_acc:.2%}")
                
                all_results[target_config['name']] = {
                    'models': models,
                    'scaler': scaler,
                    'results': results,
                    'ensemble_accuracy': ensemble_acc
                }
                
                # Save models
                self.save_models(models, scaler, model_key)
                
        return all_results
        
    def save_models(self, models, scaler, model_key):
        """حفظ النماذج"""
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/{model_key}_ensemble.pkl'
        joblib.dump({
            'models': models,
            'scaler': scaler
        }, model_path)
        
        logger.info(f"  💾 Saved: {model_path}")
        
    def train_all(self):
        """تدريب جميع الرموز والأطر الزمنية"""
        total = len(self.symbols) * len(self.timeframes)
        completed = 0
        successful = 0
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                completed += 1
                logger.info(f"\n[{completed}/{total}] Processing {symbol} {timeframe}")
                
                result = self.train_symbol_timeframe(symbol, timeframe)
                
                if result:
                    successful += 1
                    self.results[f"{symbol}_{timeframe}"] = result
                    
                # Memory cleanup
                gc.collect()
                
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Training Complete!")
        logger.info(f"   Successful: {successful}/{total}")
        logger.info(f"   Success Rate: {successful/total*100:.1f}%")
        logger.info(f"{'='*80}")

def main():
    system = RealDataTrainingSystem()
    system.train_all()

if __name__ == "__main__":
    main()