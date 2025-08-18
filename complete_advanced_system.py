#!/usr/bin/env python3
"""
🚀 النظام المتقدم الكامل - Complete Advanced System
✨ يشمل جميع الميزات والنماذج والفرضيات
📊 200+ ميزة، 6 نماذج ML، 10 فرضيات، أهداف متعددة
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

# Technical Analysis
import talib

# Import our systems
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing advanced systems
try:
    from ultimate_forex_system_v2 import AdvancedForexSystem
    from advanced_hypothesis_system import HypothesisEngine
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️ Advanced systems not found, using basic features")

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('complete_advanced_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteAdvancedSystem:
    """النظام المتقدم الكامل مع جميع الميزات"""
    
    def __init__(self):
        logger.info("="*100)
        logger.info("🚀 Complete Advanced Forex ML System")
        logger.info("📊 200+ Features | 6 ML Models | 10 Hypotheses | Multi-Targets")
        logger.info("="*100)
        
        # Initialize components
        if ADVANCED_AVAILABLE:
            self.forex_system = AdvancedForexSystem()
            self.hypothesis_engine = HypothesisEngine()
        else:
            self.forex_system = None
            self.hypothesis_engine = None
            
        # Trading symbols
        self.symbols = [
            'EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 'AUDUSDm',
            'USDCADm', 'NZDUSDm', 'EURGBPm', 'EURJPYm', 'GBPJPYm',
            'AUDJPYm', 'EURCHFm', 'GBPCHFm', 'CADJPYm', 'AUDCADm',
            'EURAUDm', 'EURCADm', 'GBPAUDm', 'GBPCADm', 'AUDCHFm',
            'CADCHFm', 'NZDJPYm', 'NZDCADm', 'NZDCHFm', 'CHFJPYm',
            'AUDNZDm', 'EURNZDm', 'GBPNZDm', 'BTCUSDm', 'ETHUSDm'
        ]
        
        # Timeframes
        self.timeframes = {
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240
        }
        
        # ML Models configuration
        self.model_configs = {
            'lightgbm': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'learning_rate': 0.05,
                    'num_leaves': 127,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'random_state': 42
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (200, 150, 100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': 42
                }
            }
        }
        
        # Multiple targets configuration
        self.target_configs = [
            {'name': 'quick_scalp', 'candles': 5, 'min_pips': 5, 'tp_ratios': [1.5, 2.0]},
            {'name': 'scalp', 'candles': 10, 'min_pips': 10, 'tp_ratios': [2.0, 3.0]},
            {'name': 'intraday', 'candles': 20, 'min_pips': 15, 'tp_ratios': [2.5, 3.5]},
            {'name': 'swing', 'candles': 60, 'min_pips': 30, 'tp_ratios': [3.0, 5.0]},
            {'name': 'position', 'candles': 240, 'min_pips': 50, 'tp_ratios': [4.0, 8.0]}
        ]
        
        # Storage
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
    def calculate_advanced_features(self, df):
        """حساب جميع الميزات المتقدمة (200+)"""
        if ADVANCED_AVAILABLE and self.forex_system:
            return self.forex_system.calculate_all_features(df)
        else:
            # Basic features if advanced not available
            return self.calculate_basic_features(df)
            
    def calculate_basic_features(self, df):
        """حساب الميزات الأساسية"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = talib.SMA(df['close'], period)
            features[f'ema_{period}'] = talib.EMA(df['close'], period)
            
        # Technical indicators
        features['rsi_14'] = talib.RSI(df['close'], 14)
        features['rsi_28'] = talib.RSI(df['close'], 28)
        
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        
        features['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], 14)
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = upper - lower
        features['bb_position'] = (df['close'] - lower) / (upper - lower + 0.0001)
        
        # Volume features if available
        if 'tick_volume' in df.columns:
            features['volume_sma'] = talib.SMA(df['tick_volume'], 10)
            features['volume_ratio'] = df['tick_volume'] / features['volume_sma']
            features['obv'] = talib.OBV(df['close'], df['tick_volume'])
        
        # Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        # Clean data
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        
        return features
        
    def create_multiple_targets(self, df, config):
        """إنشاء أهداف متعددة للتداول"""
        targets = []
        target_details = []
        
        for i in range(len(df) - config['candles']):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+config['candles']+1]
            
            if len(future_prices) == 0:
                continue
                
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            # Calculate potential profits
            long_profit = (max_price - current_price) / current_price * 10000  # in pips
            short_profit = (current_price - min_price) / current_price * 10000
            
            # Determine signal
            if long_profit > config['min_pips'] and long_profit > short_profit:
                target = 0  # Buy
                tp1 = current_price * (1 + config['tp_ratios'][0] * config['min_pips'] / 10000)
                tp2 = current_price * (1 + config['tp_ratios'][1] * config['min_pips'] / 10000)
                sl = current_price * (1 - config['min_pips'] / 10000)
            elif short_profit > config['min_pips']:
                target = 1  # Sell
                tp1 = current_price * (1 - config['tp_ratios'][0] * config['min_pips'] / 10000)
                tp2 = current_price * (1 - config['tp_ratios'][1] * config['min_pips'] / 10000)
                sl = current_price * (1 + config['min_pips'] / 10000)
            else:
                target = 2  # Hold
                tp1 = tp2 = sl = current_price
                
            targets.append(target)
            target_details.append({
                'target': target,
                'tp1': tp1,
                'tp2': tp2,
                'sl': sl,
                'expected_profit': max(long_profit, short_profit)
            })
            
        # Pad the end
        padding = config['candles']
        targets.extend([2] * padding)  # Hold
        
        return np.array(targets), target_details
        
    def train_all_models(self, X_train, X_test, y_train, y_test, model_key):
        """تدريب جميع النماذج الستة"""
        results = {}
        models = {}
        
        logger.info(f"Training 6 ML models for {model_key}...")
        
        for name, config in self.model_configs.items():
            try:
                logger.info(f"  Training {name}...")
                
                # Create model
                model = config['model'](**config['params'])
                
                # Train
                if name in ['lightgbm', 'xgboost']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                models[name] = model
                
                logger.info(f"    {name} Accuracy: {accuracy:.2%}")
                
            except Exception as e:
                logger.error(f"    Error training {name}: {str(e)}")
                
        return models, results
        
    def ensemble_predictions(self, models, X):
        """الجمع بين تنبؤات جميع النماذج"""
        all_predictions = []
        all_probabilities = []
        
        for name, model in models.items():
            try:
                pred = model.predict(X)
                prob = model.predict_proba(X)
                
                all_predictions.append(pred)
                all_probabilities.append(prob)
            except:
                pass
                
        if all_predictions:
            # Majority voting
            ensemble_pred = stats.mode(np.array(all_predictions), axis=0)[0].flatten()
            # Average probabilities
            ensemble_prob = np.mean(all_probabilities, axis=0)
            
            return ensemble_pred, ensemble_prob
        else:
            return None, None
            
    def evaluate_with_hypotheses(self, df, features, predictions, probabilities):
        """تقييم مع دمج الفرضيات"""
        if ADVANCED_AVAILABLE and self.hypothesis_engine:
            # Get hypothesis signals
            hypothesis_results = self.hypothesis_engine.evaluate_all(df, features)
            
            # Combine with ML predictions
            final_results = self.hypothesis_engine.integrate_with_ml_predictions(
                hypothesis_results,
                predictions[-1] if len(predictions) > 0 else 2,
                probabilities[-1] if len(probabilities) > 0 else [0.33, 0.33, 0.34]
            )
            
            return final_results
        else:
            return {
                'final_decision': ['BUY', 'SELL', 'HOLD'][predictions[-1]] if len(predictions) > 0 else 'HOLD',
                'confidence': np.max(probabilities[-1]) if len(probabilities) > 0 else 0.5
            }
            
    def train_symbol(self, symbol, timeframe, data_path='data/forex_data.db'):
        """تدريب رمز واحد مع جميع الميزات"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {symbol} {timeframe}")
        logger.info(f"{'='*50}")
        
        try:
            # Load data
            import sqlite3
            conn = sqlite3.connect(data_path)
            
            query = f"""
            SELECT * FROM {symbol}_{timeframe}
            ORDER BY timestamp DESC
            LIMIT 50000
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 1000:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return None
                
            # Prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Calculate features
            logger.info(f"Calculating features...")
            features = self.calculate_advanced_features(df)
            
            # Train for each target configuration
            all_results = {}
            
            for target_config in self.target_configs:
                logger.info(f"\nTraining for {target_config['name']} strategy...")
                
                # Create targets
                targets, target_details = self.create_multiple_targets(df, target_config)
                
                # Align data
                min_len = min(len(features), len(targets))
                features_aligned = features.iloc[:min_len]
                targets_aligned = targets[:min_len]
                
                # Remove NaN rows
                start_idx = 200  # After longest indicator
                features_final = features_aligned.iloc[start_idx:]
                targets_final = targets_aligned[start_idx:]
                
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
                
                # Train all models
                model_key = f"{symbol}_{timeframe}_{target_config['name']}"
                models, results = self.train_all_models(
                    X_train_scaled, X_test_scaled,
                    y_train, y_test,
                    model_key
                )
                
                # Ensemble predictions
                ensemble_pred, ensemble_prob = self.ensemble_predictions(models, X_test_scaled)
                
                if ensemble_pred is not None:
                    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                    logger.info(f"  Ensemble Accuracy: {ensemble_accuracy:.2%}")
                    
                    # Evaluate with hypotheses
                    final_eval = self.evaluate_with_hypotheses(
                        df.iloc[-len(X_test):],
                        features_final.iloc[-len(X_test):],
                        ensemble_pred,
                        ensemble_prob
                    )
                    
                    all_results[target_config['name']] = {
                        'models': models,
                        'scaler': scaler,
                        'results': results,
                        'ensemble_accuracy': ensemble_accuracy,
                        'final_evaluation': final_eval
                    }
                    
                    # Save models
                    self.save_models(models, scaler, model_key)
                    
            return all_results
            
        except Exception as e:
            logger.error(f"Error training {symbol} {timeframe}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def save_models(self, models, scaler, model_key):
        """حفظ النماذج"""
        os.makedirs('models', exist_ok=True)
        
        # Save ensemble
        ensemble_path = f'models/{model_key}_ensemble.pkl'
        joblib.dump({
            'models': models,
            'scaler': scaler,
            'feature_names': self.feature_names
        }, ensemble_path)
        
        logger.info(f"  💾 Saved ensemble: {ensemble_path}")
        
    def train_all_symbols(self):
        """تدريب جميع الرموز"""
        total = len(self.symbols) * len(self.timeframes)
        completed = 0
        successful = 0
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Starting Complete Advanced Training")
        logger.info(f"Total combinations: {total}")
        logger.info(f"{'='*100}")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                completed += 1
                
                logger.info(f"\n[{completed}/{total}] Training {symbol} {timeframe}")
                
                result = self.train_symbol(symbol, timeframe)
                
                if result:
                    successful += 1
                    self.results[f"{symbol}_{timeframe}"] = result
                    
                # Memory cleanup
                gc.collect()
                
        logger.info(f"\n{'='*100}")
        logger.info(f"Training Complete!")
        logger.info(f"Successful: {successful}/{total}")
        logger.info(f"Success Rate: {successful/total*100:.1f}%")
        logger.info(f"{'='*100}")
        
        # Save summary
        self.save_training_summary()
        
    def save_training_summary(self):
        """حفظ ملخص التدريب"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(self.symbols),
            'total_timeframes': len(self.timeframes),
            'total_models': len(self.model_configs),
            'total_features': len(self.feature_names),
            'results': {}
        }
        
        for key, result in self.results.items():
            summary['results'][key] = {
                strategy: {
                    'ensemble_accuracy': res.get('ensemble_accuracy', 0),
                    'confidence': res.get('final_evaluation', {}).get('confidence', 0)
                }
                for strategy, res in result.items()
            }
            
        with open('training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📄 Training summary saved to training_summary.json")

def main():
    system = CompleteAdvancedSystem()
    system.train_all_symbols()

if __name__ == "__main__":
    main()