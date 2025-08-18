#!/usr/bin/env python3
"""
🚀 النظام الموحد للتدريب والتداول والتعلم المستمر
📊 يدمج التدريب على البيانات التاريخية مع التعلم من الصفقات الحية
🧠 نظام واحد متكامل يتحسن باستمرار
"""

import os
import sys
import gc
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import joblib
import time
import threading
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('unified_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedTradingLearningSystem:
    """النظام الموحد الذي يجمع كل شيء"""
    
    def __init__(self):
        logger.info("="*100)
        logger.info("🚀 Unified Trading & Learning System")
        logger.info("📊 Historical Training + Live Trading + Continuous Learning")
        logger.info("🧠 One System, Complete Intelligence")
        logger.info("="*100)
        
        # Database paths
        self.historical_db = './data/forex_ml.db'  # 7.8M records
        self.trading_db = './trading_performance.db'
        self.unified_db = './unified_forex_system.db'
        
        # Initialize unified database
        self.init_unified_database()
        
        # System parameters
        self.min_confidence = 0.65
        self.retrain_interval = 24  # hours
        self.min_new_trades = 100
        self.performance_threshold = 0.60  # 60% win rate minimum
        
        # Models storage
        self.models = {}
        self.model_performance = {}
        self.last_train_time = {}
        
        # Features configuration
        self.feature_importance = {}
        self.adaptive_features = True
        
        # Initialize MT5
        self.mt5_connected = self.init_mt5()
        
    def init_unified_database(self):
        """إنشاء قاعدة بيانات موحدة"""
        conn = sqlite3.connect(self.unified_db)
        cursor = conn.cursor()
        
        # جدول الميزات المهمة
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_importance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timeframe TEXT,
            feature_name TEXT,
            importance_score REAL,
            success_correlation REAL,
            last_updated TIMESTAMP
        )''')
        
        # جدول أداء النماذج
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT,
            training_date TIMESTAMP,
            historical_accuracy REAL,
            live_accuracy REAL,
            total_trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            sharpe_ratio REAL
        )''')
        
        # جدول التعلم التكيفي
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS adaptive_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            learning_type TEXT,
            symbol TEXT,
            timeframe TEXT,
            old_parameters TEXT,
            new_parameters TEXT,
            improvement_metrics TEXT,
            market_conditions TEXT
        )''')
        
        # جدول الأنماط المربحة
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS profitable_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT UNIQUE,
            pattern_description TEXT,
            feature_conditions TEXT,
            avg_profit_pips REAL,
            success_rate REAL,
            occurrences INTEGER,
            last_seen TIMESTAMP,
            market_type TEXT
        )''')
        
        conn.commit()
        conn.close()
        
    def init_mt5(self):
        """تهيئة MT5"""
        if mt5.initialize():
            logger.info("✅ MT5 connected successfully")
            return True
        else:
            logger.warning("⚠️ MT5 not connected - will work in training mode only")
            return False
            
    def merge_historical_and_live_data(self, symbol, timeframe):
        """دمج البيانات التاريخية مع البيانات الحية"""
        all_data = []
        
        # 1. البيانات التاريخية من قاعدة البيانات
        try:
            conn = sqlite3.connect(self.historical_db)
            query = f"""
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = '{symbol}' 
            AND (timeframe = '{timeframe}' OR timeframe = 'PERIOD_{timeframe}')
            ORDER BY time
            """
            historical_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(historical_df) > 0:
                historical_df['source'] = 'historical'
                all_data.append(historical_df)
                logger.info(f"  Loaded {len(historical_df)} historical records")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            
        # 2. البيانات الحية من MT5
        if self.mt5_connected:
            try:
                # تحويل الإطار الزمني
                tf_map = {
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4
                }
                
                if timeframe in tf_map:
                    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, 10000)
                    
                    if rates is not None and len(rates) > 0:
                        live_df = pd.DataFrame(rates)
                        live_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                        live_df['source'] = 'live'
                        all_data.append(live_df)
                        logger.info(f"  Loaded {len(live_df)} live records")
                        
            except Exception as e:
                logger.error(f"Error loading live data: {str(e)}")
                
        # 3. دمج البيانات
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # إزالة المكرر
            combined_df['time'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
            combined_df = combined_df.sort_values('time')
            combined_df.set_index('time', inplace=True)
            
            logger.info(f"  ✅ Total merged data: {len(combined_df)} records")
            
            return combined_df
        else:
            return pd.DataFrame()
            
    def calculate_adaptive_features(self, df, symbol, timeframe):
        """حساب الميزات مع التكيف بناءً على الأهمية"""
        features = pd.DataFrame(index=df.index)
        
        # الميزات الأساسية (دائماً)
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # جلب أهمية الميزات
        feature_importance = self.get_feature_importance(symbol, timeframe)
        
        # الميزات المتقدمة (حسب الأهمية)
        if self.should_include_feature('moving_averages', feature_importance, 0.1):
            for period in [5, 10, 20, 50, 100]:
                if TALIB_AVAILABLE:
                    features[f'sma_{period}'] = talib.SMA(df['close'], period)
                    features[f'ema_{period}'] = talib.EMA(df['close'], period)
                else:
                    features[f'sma_{period}'] = df['close'].rolling(period).mean()
                    features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                    
        if self.should_include_feature('momentum', feature_importance, 0.15):
            if TALIB_AVAILABLE:
                features['rsi_14'] = talib.RSI(df['close'], 14)
                features['rsi_28'] = talib.RSI(df['close'], 28)
                
                macd, macd_signal, macd_hist = talib.MACD(df['close'])
                features['macd'] = macd
                features['macd_signal'] = macd_signal
                features['macd_hist'] = macd_hist
                
        if self.should_include_feature('volatility', feature_importance, 0.12):
            if TALIB_AVAILABLE:
                features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                features['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], 14)
                
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                features['bb_width'] = upper - lower
                features['bb_position'] = (df['close'] - lower) / (upper - lower + 0.0001)
                
        if self.should_include_feature('trend', feature_importance, 0.18):
            if TALIB_AVAILABLE:
                features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                
        if self.should_include_feature('volume', feature_importance, 0.08) and 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
        if self.should_include_feature('time', feature_importance, 0.05):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['is_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
            
        if self.should_include_feature('patterns', feature_importance, 0.20):
            # أنماط الشموع
            features['body_size'] = np.abs(df['close'] - df['open'])
            features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # نسب مهمة
            features['shadow_body_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / (features['body_size'] + 0.0001)
            
        # تنظيف البيانات
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # حفظ أسماء الميزات
        self.current_features = features.columns.tolist()
        
        return features
        
    def should_include_feature(self, feature_group, importance_dict, min_importance):
        """تحديد ما إذا كانت الميزة مهمة بما يكفي"""
        if not self.adaptive_features:
            return True  # استخدم كل الميزات
            
        avg_importance = np.mean([v for k, v in importance_dict.items() if feature_group in k])
        return avg_importance >= min_importance if avg_importance > 0 else True
        
    def get_feature_importance(self, symbol, timeframe):
        """جلب أهمية الميزات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.unified_db)
            query = """
            SELECT feature_name, importance_score 
            FROM feature_importance
            WHERE symbol = ? AND timeframe = ?
            ORDER BY importance_score DESC
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            return dict(zip(df['feature_name'], df['importance_score']))
        except:
            return {}
            
    def train_unified_model(self, symbol, timeframe, force_retrain=False):
        """تدريب نموذج موحد يجمع البيانات التاريخية والحية"""
        model_key = f"{symbol}_{timeframe}"
        
        # التحقق من الحاجة لإعادة التدريب
        if not force_retrain and model_key in self.last_train_time:
            hours_since_train = (datetime.now() - self.last_train_time[model_key]).total_seconds() / 3600
            if hours_since_train < self.retrain_interval:
                logger.info(f"  Model {model_key} recently trained ({hours_since_train:.1f}h ago)")
                return self.models.get(model_key)
                
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Unified Model: {symbol} {timeframe}")
        logger.info(f"{'='*60}")
        
        # دمج البيانات
        df = self.merge_historical_and_live_data(symbol, timeframe)
        
        if len(df) < 1000:
            logger.warning(f"Not enough data for {model_key}")
            return None
            
        # حساب الميزات التكيفية
        features = self.calculate_adaptive_features(df, symbol, timeframe)
        
        # إنشاء الأهداف
        targets = self.create_advanced_targets(df)
        
        # محاذاة البيانات
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets[:min_len]
        
        # إزالة NaN
        start_idx = 100
        features = features.iloc[start_idx:]
        targets = targets[start_idx:]
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=0.2,
            random_state=42,
            stratify=targets
        )
        
        # المعايرة
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب نماذج متعددة
        models = {}
        results = {}
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test_scaled))
        models['lightgbm'] = lgb_model
        results['lightgbm'] = lgb_acc
        logger.info(f"  LightGBM: {lgb_acc:.2%}")
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train, verbose=False)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
        models['xgboost'] = xgb_model
        results['xgboost'] = xgb_acc
        logger.info(f"  XGBoost: {xgb_acc:.2%}")
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=30,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
        models['random_forest'] = rf_model
        results['random_forest'] = rf_acc
        logger.info(f"  Random Forest: {rf_acc:.2%}")
        
        # Ensemble
        ensemble_preds = []
        for model in models.values():
            ensemble_preds.append(model.predict(X_test_scaled))
        ensemble_pred = stats.mode(np.array(ensemble_preds), axis=0)[0].flatten()
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        logger.info(f"  ✨ Ensemble: {ensemble_acc:.2%}")
        
        # حفظ أهمية الميزات
        self.save_feature_importance(symbol, timeframe, models, features.columns)
        
        # حفظ النموذج
        model_data = {
            'models': models,
            'scaler': scaler,
            'features': features.columns.tolist(),
            'accuracy': ensemble_acc,
            'results': results
        }
        
        self.models[model_key] = model_data
        self.last_train_time[model_key] = datetime.now()
        
        # حفظ الأداء
        self.save_model_performance(model_key, ensemble_acc, len(X_train))
        
        # حفظ النموذج على القرص
        self.save_model_to_disk(model_key, model_data)
        
        return model_data
        
    def create_advanced_targets(self, df, min_pips=15, future_candles=20):
        """إنشاء أهداف متقدمة"""
        targets = []
        
        for i in range(len(df) - future_candles):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+future_candles+1]
            
            if len(future_prices) == 0:
                continue
                
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            # حساب النقاط
            if 'JPY' in df.index.name if hasattr(df.index, 'name') else '':
                pip_value = 0.01
            else:
                pip_value = 0.0001
                
            long_profit = (max_price - current_price) / pip_value
            short_profit = (current_price - min_price) / pip_value
            
            if long_profit > min_pips and long_profit > short_profit * 1.2:
                target = 0  # Buy
            elif short_profit > min_pips and short_profit > long_profit * 1.2:
                target = 1  # Sell
            else:
                target = 2  # Hold
                
            targets.append(target)
            
        # Padding
        targets.extend([2] * future_candles)
        
        return np.array(targets)
        
    def save_feature_importance(self, symbol, timeframe, models, feature_names):
        """حفظ أهمية الميزات"""
        conn = sqlite3.connect(self.unified_db)
        cursor = conn.cursor()
        
        # حذف القديم
        cursor.execute("""
        DELETE FROM feature_importance 
        WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))
        
        # حساب الأهمية من كل نموذج
        importance_scores = {}
        
        # LightGBM
        if 'lightgbm' in models:
            lgb_importance = models['lightgbm'].feature_importances_
            for i, feature in enumerate(feature_names):
                importance_scores[feature] = importance_scores.get(feature, 0) + lgb_importance[i]
                
        # Random Forest
        if 'random_forest' in models:
            rf_importance = models['random_forest'].feature_importances_
            for i, feature in enumerate(feature_names):
                importance_scores[feature] = importance_scores.get(feature, 0) + rf_importance[i]
                
        # تطبيع وحفظ
        total = sum(importance_scores.values())
        
        for feature, score in importance_scores.items():
            normalized_score = score / total if total > 0 else 0
            
            cursor.execute("""
            INSERT INTO feature_importance 
            (symbol, timeframe, feature_name, importance_score, last_updated)
            VALUES (?, ?, ?, ?, ?)
            """, (symbol, timeframe, feature, normalized_score, datetime.now()))
            
        conn.commit()
        conn.close()
        
    def save_model_performance(self, model_key, accuracy, training_samples):
        """حفظ أداء النموذج"""
        conn = sqlite3.connect(self.unified_db)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO model_performance 
        (model_key, training_date, historical_accuracy, total_trades)
        VALUES (?, ?, ?, ?)
        """, (model_key, datetime.now(), accuracy, training_samples))
        
        conn.commit()
        conn.close()
        
    def save_model_to_disk(self, model_key, model_data):
        """حفظ النموذج على القرص"""
        os.makedirs('unified_models', exist_ok=True)
        
        model_path = f'unified_models/{model_key}_unified.pkl'
        joblib.dump(model_data, model_path)
        
        logger.info(f"  💾 Saved: {model_path}")
        
    def analyze_and_learn_from_trades(self):
        """تحليل الصفقات والتعلم منها"""
        conn = sqlite3.connect(self.trading_db)
        
        # جلب الصفقات الأخيرة
        query = """
        SELECT * FROM trades 
        WHERE exit_time IS NOT NULL 
        AND learning_notes IS NULL
        ORDER BY exit_time DESC 
        LIMIT 100
        """
        
        trades_df = pd.read_sql_query(query, conn)
        
        if len(trades_df) == 0:
            conn.close()
            return
            
        # تحليل الأنماط
        patterns = self.discover_profitable_patterns(trades_df)
        
        # حفظ الأنماط المربحة
        self.save_profitable_patterns(patterns)
        
        # تحديث النماذج بناءً على التعلم
        symbols_to_update = trades_df['symbol'].unique()
        
        for symbol in symbols_to_update:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            
            # حساب معدل النجاح
            win_rate = len(symbol_trades[symbol_trades['result'] == 'WIN']) / len(symbol_trades)
            
            # إذا كان الأداء ضعيف، أعد التدريب
            if win_rate < self.performance_threshold:
                logger.info(f"⚠️ Low win rate for {symbol}: {win_rate:.1%} - Retraining...")
                
                for timeframe in ['M15', 'H1']:
                    self.train_unified_model(symbol, timeframe, force_retrain=True)
                    
        conn.close()
        
    def discover_profitable_patterns(self, trades_df):
        """اكتشاف الأنماط المربحة"""
        patterns = []
        
        # تحليل الصفقات الرابحة
        winning_trades = trades_df[trades_df['result'] == 'WIN']
        
        for _, trade in winning_trades.iterrows():
            if trade['features_snapshot']:
                features = json.loads(trade['features_snapshot'])
                
                # البحث عن أنماط مشتركة
                pattern = {
                    'rsi_range': self.get_range(features.get('rsi_14', 50)),
                    'adx_range': self.get_range(features.get('adx_14', 25)),
                    'bb_position': self.get_range(features.get('bb_position', 0.5)),
                    'hour': features.get('hour', -1),
                    'profit_pips': trade['pnl_pips']
                }
                
                patterns.append(pattern)
                
        return patterns
        
    def get_range(self, value, step=10):
        """تحويل القيمة إلى نطاق"""
        return f"{int(value//step)*step}-{int(value//step)*step+step}"
        
    def save_profitable_patterns(self, patterns):
        """حفظ الأنماط المربحة"""
        conn = sqlite3.connect(self.unified_db)
        cursor = conn.cursor()
        
        # تجميع الأنماط المتشابهة
        pattern_groups = {}
        
        for pattern in patterns:
            pattern_hash = json.dumps(pattern, sort_keys=True)
            
            if pattern_hash not in pattern_groups:
                pattern_groups[pattern_hash] = {
                    'count': 0,
                    'total_pips': 0,
                    'pattern': pattern
                }
                
            pattern_groups[pattern_hash]['count'] += 1
            pattern_groups[pattern_hash]['total_pips'] += pattern['profit_pips']
            
        # حفظ الأنماط المربحة
        for pattern_hash, data in pattern_groups.items():
            avg_pips = data['total_pips'] / data['count']
            
            if avg_pips > 20 and data['count'] >= 3:  # نمط مربح ومتكرر
                cursor.execute("""
                INSERT OR REPLACE INTO profitable_patterns 
                (pattern_hash, pattern_description, feature_conditions, 
                 avg_profit_pips, occurrences, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern_hash,
                    f"Profitable pattern with {avg_pips:.1f} pips average",
                    json.dumps(data['pattern']),
                    avg_pips,
                    data['count'],
                    datetime.now()
                ))
                
        conn.commit()
        conn.close()
        
    def predict_with_pattern_matching(self, symbol, timeframe, features):
        """التنبؤ مع مطابقة الأنماط"""
        # التنبؤ الأساسي من النموذج
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            return None, 0
            
        model_data = self.models[model_key]
        
        # تطبيق المعايرة
        features_scaled = model_data['scaler'].transform(features)
        
        # التنبؤ من كل نموذج
        predictions = []
        for model in model_data['models'].values():
            predictions.append(model.predict(features_scaled)[0])
            
        # القرار الأساسي
        base_prediction = max(set(predictions), key=predictions.count)
        
        # البحث عن أنماط مربحة مطابقة
        pattern_boost = self.check_profitable_patterns(features.iloc[-1])
        
        # دمج التنبؤ مع الأنماط
        if pattern_boost > 0:
            confidence = 0.8  # ثقة عالية
        else:
            confidence = 0.65
            
        return base_prediction, confidence
        
    def check_profitable_patterns(self, current_features):
        """التحقق من الأنماط المربحة"""
        conn = sqlite3.connect(self.unified_db)
        
        query = """
        SELECT * FROM profitable_patterns
        WHERE occurrences >= 5
        AND avg_profit_pips > 25
        """
        
        patterns_df = pd.read_sql_query(query, conn)
        conn.close()
        
        boost = 0
        
        for _, pattern in patterns_df.iterrows():
            conditions = json.loads(pattern['feature_conditions'])
            
            # مطابقة الشروط
            match = True
            
            if 'rsi_range' in conditions:
                rsi_value = current_features.get('rsi_14', 50)
                rsi_range = conditions['rsi_range'].split('-')
                if not (float(rsi_range[0]) <= rsi_value <= float(rsi_range[1])):
                    match = False
                    
            if match:
                boost += pattern['avg_profit_pips'] / 100  # تعزيز بناءً على الربحية
                
        return boost
        
    def run_unified_system(self):
        """تشغيل النظام الموحد"""
        logger.info("\n" + "="*80)
        logger.info("🚀 Starting Unified Trading & Learning System")
        logger.info("="*80)
        
        # تدريب أولي على البيانات التاريخية
        initial_symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
        initial_timeframes = ['M15', 'H1']
        
        logger.info("\n📊 Initial training on historical data...")
        
        for symbol in initial_symbols:
            for timeframe in initial_timeframes:
                self.train_unified_model(symbol, timeframe)
                
        # بدء حلقة التداول والتعلم
        logger.info("\n🔄 Starting live trading and continuous learning...")
        
        while True:
            try:
                # 1. التداول (إذا كان MT5 متصل)
                if self.mt5_connected:
                    self.execute_trading_cycle()
                    
                # 2. التعلم من الصفقات
                self.analyze_and_learn_from_trades()
                
                # 3. تحديث النماذج دورياً
                self.periodic_model_update()
                
                # 4. عرض الإحصائيات
                self.display_unified_stats()
                
                # الانتظار
                time.sleep(60)  # دقيقة واحدة
                
            except KeyboardInterrupt:
                logger.info("\nSystem stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)
                
    def execute_trading_cycle(self):
        """دورة التداول"""
        # هنا يمكن إضافة منطق التداول الفعلي
        pass
        
    def periodic_model_update(self):
        """تحديث النماذج دورياً"""
        for model_key, last_time in self.last_train_time.items():
            hours_passed = (datetime.now() - last_time).total_seconds() / 3600
            
            if hours_passed >= self.retrain_interval:
                symbol, timeframe = model_key.split('_')[:2]
                logger.info(f"🔄 Periodic retrain for {model_key}")
                self.train_unified_model(symbol, timeframe, force_retrain=True)
                
    def display_unified_stats(self):
        """عرض إحصائيات النظام الموحد"""
        logger.info("\n📊 System Statistics:")
        logger.info(f"  Active Models: {len(self.models)}")
        
        # إحصائيات الأداء
        conn = sqlite3.connect(self.unified_db)
        
        avg_accuracy = pd.read_sql_query("""
        SELECT AVG(historical_accuracy) as avg_acc
        FROM model_performance
        WHERE training_date > datetime('now', '-7 days')
        """, conn).iloc[0]['avg_acc']
        
        pattern_count = pd.read_sql_query("""
        SELECT COUNT(*) as count
        FROM profitable_patterns
        WHERE avg_profit_pips > 20
        """, conn).iloc[0]['count']
        
        conn.close()
        
        if avg_accuracy:
            logger.info(f"  Average Model Accuracy: {avg_accuracy:.1%}")
        logger.info(f"  Profitable Patterns Found: {pattern_count}")

def main():
    """تشغيل النظام الموحد"""
    system = UnifiedTradingLearningSystem()
    system.run_unified_system()
    
    # إغلاق MT5
    if system.mt5_connected:
        mt5.shutdown()

if __name__ == "__main__":
    main()