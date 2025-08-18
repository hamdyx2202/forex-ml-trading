#!/usr/bin/env python3
"""
Ultimate Advanced Training System - النظام المتقدم النهائي
يحتوي على:
- 200+ ميزة متقدمة كاملة
- 5 نماذج ذكاء اصطناعي
- حسابات SL/TP ديناميكية متقدمة
- 5 استراتيجيات تداول
- دعم ومقاومة مع Fibonacci
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from ta import add_all_ta_features
from ta.utils import dropna
import talib
import time
import gc
import psutil

# مكتبات المعالجة المتوازية
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/ultimate_training_system.log", rotation="1 day", retention="30 days")

class UltimateAdvancedTrainer:
    """النظام المتقدم النهائي بجميع الميزات والنماذج"""
    
    def __init__(self, use_all_features=True, use_all_models=True):
        self.min_data_points = 10000
        self.test_size = 0.2
        self.validation_split = 0.15
        self.random_state = 42
        self.use_all_features = use_all_features
        self.use_all_models = use_all_models
        
        # عدد العمليات المتوازية
        self.max_workers = min(cpu_count() - 1, 4)
        
        # عرض الإعدادات
        logger.info("="*100)
        logger.info("🚀 النظام المتقدم النهائي للتدريب")
        logger.info("="*100)
        logger.info(f"📊 الميزات: {'200+ ميزة متقدمة' if use_all_features else 'ميزات أساسية'}")
        logger.info(f"🤖 النماذج: {'5 نماذج ذكاء اصطناعي' if use_all_models else 'LightGBM فقط'}")
        logger.info(f"⚡ المعالجة المتوازية: {self.max_workers} عمليات")
        
        # مراقبة الذاكرة
        self.monitor_memory()
        
        # 5 استراتيجيات تداول مختلفة
        self.training_strategies = {
            'ultra_short': {
                'lookahead': 5,
                'min_pips': 3,
                'confidence_threshold': 0.95,
                'description': 'سكالبينج فائق السرعة'
            },
            'scalping': {
                'lookahead': 15,
                'min_pips': 5,
                'confidence_threshold': 0.92,
                'description': 'سكالبينج عادي'
            },
            'short_term': {
                'lookahead': 30,
                'min_pips': 10,
                'confidence_threshold': 0.90,
                'description': 'تداول قصير المدى'
            },
            'medium_term': {
                'lookahead': 60,
                'min_pips': 20,
                'confidence_threshold': 0.88,
                'description': 'تداول متوسط المدى'
            },
            'long_term': {
                'lookahead': 240,
                'min_pips': 40,
                'confidence_threshold': 0.85,
                'description': 'تداول طويل المدى'
            }
        }
        
        # إعدادات SL/TP ديناميكية لكل استراتيجية
        self.sl_tp_settings = {
            'ultra_short': {
                'stop_loss_atr': 0.5,
                'take_profit_ratios': [1.0, 1.5, 2.0],
                'trailing_stop_atr': 0.3,
                'breakeven_pips': 5
            },
            'scalping': {
                'stop_loss_atr': 1.0,
                'take_profit_ratios': [1.0, 2.0, 3.0],
                'trailing_stop_atr': 0.5,
                'breakeven_pips': 10
            },
            'short_term': {
                'stop_loss_atr': 1.5,
                'take_profit_ratios': [1.5, 2.5, 3.5],
                'trailing_stop_atr': 0.7,
                'breakeven_pips': 15
            },
            'medium_term': {
                'stop_loss_atr': 2.0,
                'take_profit_ratios': [2.0, 3.0, 4.0],
                'trailing_stop_atr': 1.0,
                'breakeven_pips': 20
            },
            'long_term': {
                'stop_loss_atr': 2.5,
                'take_profit_ratios': [2.5, 4.0, 6.0],
                'trailing_stop_atr': 1.5,
                'breakeven_pips': 30
            }
        }
        
        # إعدادات 5 نماذج ذكاء اصطناعي
        self.model_configs = {
            'lightgbm': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'dart',
                'num_leaves': 127,
                'max_depth': -1,
                'learning_rate': 0.01,
                'n_estimators': 2000,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'min_child_samples': 20,
                'verbosity': -1,
                'n_jobs': 1
            },
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 10,
                'learning_rate': 0.01,
                'n_estimators': 2000,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'tree_method': 'hist',
                'n_jobs': 1
            },
            'catboost': {
                'loss_function': 'MultiClass',
                'classes_count': 3,
                'iterations': 2000,
                'depth': 10,
                'learning_rate': 0.01,
                'l2_leaf_reg': 5,
                'border_count': 254,
                'random_strength': 0.1,
                'bagging_temperature': 0.1,
                'od_type': 'Iter',
                'od_wait': 50,
                'verbose': False,
                'thread_count': 1
            },
            'extra_trees': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'n_jobs': 1
            },
            'neural_network': {
                'hidden_layer_sizes': (200, 150, 100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20
            }
        }
    
    def monitor_memory(self):
        """مراقبة استخدام الذاكرة"""
        memory = psutil.virtual_memory()
        logger.info(f"💾 الذاكرة: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        if memory.percent > 80:
            logger.warning("⚠️ استخدام الذاكرة مرتفع!")
    
    def check_database(self):
        """التحقق من قاعدة البيانات"""
        db_path = Path("data/forex_ml.db")
        
        if not db_path.exists():
            logger.error("❌ قاعدة البيانات غير موجودة!")
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"✅ قاعدة البيانات: {count:,} سجل")
            return count > 0
            
        except Exception as e:
            logger.error(f"❌ خطأ: {e}")
            return False
    
    def load_data_advanced(self, symbol, timeframe, limit=100000):
        """تحميل البيانات"""
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time ASC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if len(df) < self.min_data_points:
                logger.warning(f"⚠️ بيانات غير كافية: {len(df)} سجل")
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ: {e}")
            return None
    
    def create_ultra_advanced_features(self, df, symbol):
        """إنشاء 200+ ميزة متقدمة"""
        if not self.use_all_features:
            return self.create_basic_features(df)
        
        logger.info("🔬 إنشاء 200+ ميزة متقدمة...")
        start_time = time.time()
        
        features = pd.DataFrame(index=df.index)
        
        # 1. ميزات الأسعار الأساسية
        logger.info("  • ميزات الأسعار...")
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # 2. المتوسطات المتحركة المتعددة (14 فترة)
        logger.info("  • المتوسطات المتحركة...")
        ma_periods = [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233]
        for period in ma_periods:
            sma = df['close'].rolling(period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            
            features[f'sma_{period}'] = (df['close'] - sma) / sma
            features[f'ema_{period}'] = (df['close'] - ema) / ema
            features[f'sma_{period}_slope'] = sma.pct_change(5)
            features[f'ema_{period}_slope'] = ema.pct_change(5)
        
        # 3. مؤشرات TA-Lib المتقدمة
        logger.info("  • مؤشرات فنية...")
        try:
            # RSI متعدد (6 فترات)
            for period in [5, 7, 9, 14, 21, 28]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            
            # Stochastic (3 فترات)
            for period in [5, 9, 14]:
                slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                          fastk_period=period, slowk_period=3, slowd_period=3)
                features[f'stoch_k_{period}'] = slowk
                features[f'stoch_d_{period}'] = slowd
            
            # MACD متعدد (3 إعدادات)
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (3, 10, 16)]:
                macd, macdsignal, macdhist = talib.MACD(df['close'].values, 
                                                        fastperiod=fast, 
                                                        slowperiod=slow, 
                                                        signalperiod=signal)
                features[f'macd_{fast}_{slow}'] = macd
                features[f'macd_signal_{fast}_{slow}'] = macdsignal
                features[f'macd_hist_{fast}_{slow}'] = macdhist
            
            # Bollinger Bands متعدد (9 إعدادات)
            for period in [10, 20, 30]:
                for dev in [1.5, 2.0, 2.5]:
                    upper, middle, lower = talib.BBANDS(df['close'].values,
                                                       timeperiod=period,
                                                       nbdevup=dev,
                                                       nbdevdn=dev)
                    features[f'bb_upper_{period}_{dev}'] = (upper - df['close']) / df['close']
                    features[f'bb_lower_{period}_{dev}'] = (df['close'] - lower) / df['close']
                    features[f'bb_width_{period}_{dev}'] = (upper - lower) / middle
            
            # ATR و Volatility (5 فترات)
            for period in [7, 10, 14, 20, 30]:
                features[f'atr_{period}'] = talib.ATR(df['high'].values, 
                                                     df['low'].values, 
                                                     df['close'].values, 
                                                     timeperiod=period) / df['close']
                features[f'natr_{period}'] = talib.NATR(df['high'].values, 
                                                       df['low'].values, 
                                                       df['close'].values, 
                                                       timeperiod=period)
            
            # ADX و DI (3 فترات)
            for period in [7, 14, 21]:
                features[f'adx_{period}'] = talib.ADX(df['high'].values, 
                                                     df['low'].values, 
                                                     df['close'].values, 
                                                     timeperiod=period)
                features[f'plus_di_{period}'] = talib.PLUS_DI(df['high'].values, 
                                                              df['low'].values, 
                                                              df['close'].values, 
                                                              timeperiod=period)
                features[f'minus_di_{period}'] = talib.MINUS_DI(df['high'].values, 
                                                               df['low'].values, 
                                                               df['close'].values, 
                                                               timeperiod=period)
            
            # مؤشرات إضافية
            features['cci_14'] = talib.CCI(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=14)
            features['mfi_14'] = talib.MFI(df['high'].values, df['low'].values, 
                                          df['close'].values, df['volume'].values, timeperiod=14)
            features['roc_10'] = talib.ROC(df['close'].values, timeperiod=10)
            features['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, 
                                                df['close'].values, timeperiod=14)
            features['ultimate_osc'] = talib.ULTOSC(df['high'].values, df['low'].values, 
                                                   df['close'].values)
            
            # أنماط الشموع اليابانية (8 أنماط)
            patterns = {
                'cdl_doji': talib.CDLDOJI,
                'cdl_hammer': talib.CDLHAMMER,
                'cdl_shooting_star': talib.CDLSHOOTINGSTAR,
                'cdl_engulfing': talib.CDLENGULFING,
                'cdl_morning_star': talib.CDLMORNINGSTAR,
                'cdl_evening_star': talib.CDLEVENINGSTAR,
                'cdl_3_black_crows': talib.CDL3BLACKCROWS,
                'cdl_3_white_soldiers': talib.CDL3WHITESOLDIERS
            }
            
            for name, func in patterns.items():
                features[name] = func(df['open'].values, df['high'].values, 
                                    df['low'].values, df['close'].values)
                
        except Exception as e:
            logger.warning(f"⚠️ بعض المؤشرات فشلت: {e}")
        
        # 4. ميزات الحجم المتقدمة
        logger.info("  • ميزات الحجم...")
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_sma_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        
        # Force Index
        features['force_index'] = df['close'].diff() * df['volume']
        features['force_index_ema'] = features['force_index'].ewm(span=13).mean()
        
        # 5. ميزات التقلب المتقدمة
        logger.info("  • ميزات التقلب...")
        for period in [5, 10, 20, 30, 50]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / returns.rolling(period * 2).std()
            
            # Parkinson volatility
            features[f'parkinson_vol_{period}'] = np.sqrt(
                np.log(df['high']/df['low']).pow(2).rolling(period).mean() / (4 * np.log(2))
            )
            
            # Garman-Klass volatility
            features[f'gk_vol_{period}'] = np.sqrt(
                0.5 * np.log(df['high']/df['low']).pow(2).rolling(period).mean() -
                (2*np.log(2)-1) * np.log(df['close']/df['open']).pow(2).rolling(period).mean()
            )
        
        # 6. ميزات السوق والوقت
        logger.info("  • ميزات الوقت...")
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # جلسات التداول
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
        features['london_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        features['sydney_session'] = ((features['hour'] >= 22) | (features['hour'] < 7)).astype(int)
        
        # تداخل الجلسات
        features['london_ny_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 17)).astype(int)
        features['asian_london_overlap'] = ((features['hour'] >= 8) & (features['hour'] < 9)).astype(int)
        
        # 7. ميزات الارتباط والعلاقات
        logger.info("  • ميزات العلاقات...")
        for period in [5, 10, 20]:
            features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            features[f'close_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'close_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()
        
        # 8. ميزات الزخم المتقدمة
        logger.info("  • ميزات الزخم...")
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features[f'acceleration_{period}'] = features[f'momentum_{period}'].diff()
        
        # 9. مستويات الدعم والمقاومة الديناميكية
        logger.info("  • الدعم والمقاومة...")
        for period in [20, 50, 100]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            features[f'distance_from_high_{period}'] = (rolling_high - df['close']) / df['close']
            features[f'distance_from_low_{period}'] = (df['close'] - rolling_low) / df['close']
            features[f'position_in_range_{period}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)
        
        # 10. ميزات إحصائية متقدمة
        logger.info("  • ميزات إحصائية...")
        for period in [10, 20, 50]:
            rolling_returns = df['close'].pct_change().rolling(period)
            features[f'skewness_{period}'] = rolling_returns.skew()
            features[f'kurtosis_{period}'] = rolling_returns.kurt()
            features[f'mean_return_{period}'] = rolling_returns.mean()
            features[f'median_return_{period}'] = rolling_returns.median()
        
        # تنظيف البيانات
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        end_time = time.time()
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة في {end_time - start_time:.2f} ثانية")
        
        return features
    
    def create_basic_features(self, df):
        """إنشاء ميزات أساسية فقط (للمقارنة)"""
        logger.info("🔬 إنشاء ميزات أساسية...")
        features = pd.DataFrame(index=df.index)
        
        # أساسيات فقط
        features['returns'] = df['close'].pct_change()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        
        # متوسطات بسيطة
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = (df['close'] - sma) / sma
        
        # RSI
        features['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'].values)
        features['macd'] = macd
        features['macd_signal'] = signal
        
        # الحجم
        features['volume_sma'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # الوقت
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        features = features.fillna(0)
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة أساسية")
        
        return features
    
    def calculate_pip_value(self, symbol):
        """حساب قيمة النقطة حسب العملة"""
        if 'JPY' in symbol or 'XAG' in symbol:
            return 0.01
        elif 'XAU' in symbol:
            return 0.1
        else:
            return 0.0001
    
    def find_support_resistance_levels(self, df, current_idx, lookback=150):
        """إيجاد مستويات الدعم والمقاومة مع Fibonacci"""
        start_idx = max(0, current_idx - lookback)
        price_data = df[['high', 'low', 'close']].iloc[start_idx:current_idx]
        
        highs = []
        lows = []
        
        # إيجاد القمم والقيعان
        window = 5
        for i in range(window, len(price_data) - window):
            # قمة محلية
            if (price_data['high'].iloc[i] == price_data['high'].iloc[i-window:i+window+1].max()):
                highs.append(price_data['high'].iloc[i])
            
            # قاع محلي
            if (price_data['low'].iloc[i] == price_data['low'].iloc[i-window:i+window+1].min()):
                lows.append(price_data['low'].iloc[i])
        
        # مستويات Fibonacci
        if len(price_data) > 0:
            recent_high = price_data['high'].max()
            recent_low = price_data['low'].min()
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for level in fib_levels:
                fib_price = recent_low + (recent_high - recent_low) * level
                if fib_price > price_data['close'].iloc[-1]:
                    highs.append(fib_price)
                else:
                    lows.append(fib_price)
        
        # تجميع المستويات القريبة
        def cluster_levels(levels, threshold=0.002):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = [[levels[0]]]
            
            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            result = []
            for cluster in clusters:
                weight = len(cluster)
                avg_level = sum(cluster) / len(cluster)
                result.append({'level': avg_level, 'strength': weight})
            
            result.sort(key=lambda x: x['strength'], reverse=True)
            return [item['level'] for item in result[:10]]
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def calculate_dynamic_sl_tp(self, df, position_type, entry_idx, strategy_name):
        """حساب SL/TP ديناميكي متقدم مع الدعم والمقاومة"""
        sl_tp_config = self.sl_tp_settings[strategy_name]
        
        # ATR
        atr_period = 14
        if f'atr_{atr_period}' in df.columns:
            current_atr = df.iloc[entry_idx][f'atr_{atr_period}']
        else:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            current_atr = true_range.rolling(atr_period).mean().iloc[entry_idx]
        
        current_price = df['close'].iloc[entry_idx]
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        # حساب المسافات
        sl_distance = current_atr * sl_tp_config['stop_loss_atr']
        tp_distances = [sl_distance * ratio for ratio in sl_tp_config['take_profit_ratios']]
        
        # الدعم والمقاومة
        support_resistance = self.find_support_resistance_levels(df, entry_idx)
        
        if position_type == 'long':
            # Stop Loss
            stop_loss = current_price - sl_distance
            
            # تعديل بناءً على الدعم
            nearest_support = min([s for s in support_resistance['support'] 
                                 if s < current_price and s > stop_loss], 
                                default=stop_loss)
            if nearest_support > stop_loss:
                stop_loss = nearest_support - (2 * pip_value)
            
            # Take Profit متعدد
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price + tp_distance
                
                # تعديل بناءً على المقاومة
                nearest_resistance = min([r for r in support_resistance['resistance'] 
                                        if r > current_price and r < tp], 
                                       default=tp)
                if nearest_resistance < tp:
                    tp = nearest_resistance - (2 * pip_value)
                
                take_profits.append(tp)
            
        else:  # Short
            # Stop Loss
            stop_loss = current_price + sl_distance
            
            # تعديل بناءً على المقاومة
            nearest_resistance = max([r for r in support_resistance['resistance'] 
                                    if r > current_price and r < stop_loss], 
                                   default=stop_loss)
            if nearest_resistance < stop_loss:
                stop_loss = nearest_resistance + (2 * pip_value)
            
            # Take Profit متعدد
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price - tp_distance
                
                # تعديل بناءً على الدعم
                nearest_support = max([s for s in support_resistance['support'] 
                                     if s < current_price and s > tp], 
                                    default=tp)
                if nearest_support > tp:
                    tp = nearest_support + (2 * pip_value)
                
                take_profits.append(tp)
        
        return {
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'trailing_stop_distance': current_atr * sl_tp_config['trailing_stop_atr'],
            'breakeven_pips': sl_tp_config['breakeven_pips'],
            'risk_amount': abs(current_price - stop_loss),
            'reward_ratios': sl_tp_config['take_profit_ratios'],
            'atr_used': current_atr,
            'support_levels': support_resistance['support'][:3],
            'resistance_levels': support_resistance['resistance'][:3]
        }
    
    def create_advanced_targets_with_sl_tp(self, df, strategy_name):
        """إنشاء أهداف متقدمة مع SL/TP كاملة"""
        strategy = self.training_strategies[strategy_name]
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        targets = []
        confidences = []
        sl_tp_info = []
        trade_quality = []
        
        logger.info(f"    • إنشاء أهداف {strategy_name} ({strategy['description']})...")
        
        # معالجة على دفعات لتحسين الأداء
        batch_size = 5000
        total_samples = len(df) - lookahead
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            
            # عرض التقدم
            if batch_start % 10000 == 0:
                progress = (batch_start / total_samples) * 100
                logger.info(f"      Progress: {progress:.1f}% ({batch_start}/{total_samples})")
            
            for i in range(batch_start, batch_end):
                future_prices = df['close'].iloc[i+1:i+lookahead+1].values
                future_highs = df['high'].iloc[i+1:i+lookahead+1].values
                future_lows = df['low'].iloc[i+1:i+lookahead+1].values
                current_price = df['close'].iloc[i]
            
                # حركة مع الـ wicks
                max_up = (future_highs.max() - current_price) / pip_value
                max_down = (current_price - future_lows.min()) / pip_value
                
                # حركة الإغلاق
                close_up = (future_prices.max() - current_price) / pip_value
                close_down = (current_price - future_prices.min()) / pip_value
                
                # تحديد نوع الصفقة
                if max_up >= min_pips * 2 and close_up >= min_pips * 1.5:
                    # Long
                    targets.append(2)
                
                confidence = min(
                    0.5 + (close_up / (min_pips * 4)) * 0.3 +
                    (1 - max_down / max_up) * 0.2,
                    1.0
                )
                confidences.append(confidence)
                
                sl_tp = self.calculate_dynamic_sl_tp(df, 'long', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                quality = self.evaluate_trade_quality(df, i, 'long', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            elif max_down >= min_pips * 2 and close_down >= min_pips * 1.5:
                # Short
                targets.append(0)
                
                confidence = min(
                    0.5 + (close_down / (min_pips * 4)) * 0.3 +
                    (1 - max_up / max_down) * 0.2,
                    1.0
                )
                confidences.append(confidence)
                
                sl_tp = self.calculate_dynamic_sl_tp(df, 'short', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                quality = self.evaluate_trade_quality(df, i, 'short', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            else:
                # Hold
                targets.append(1)
                confidences.append(0.5)
                sl_tp_info.append(None)
                trade_quality.append(0)
        
        # ملء الأخير
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        sl_tp_info.extend([None] * lookahead)
        trade_quality.extend([0] * lookahead)
        
        # إحصائيات
        buy_count = targets.count(2)
        sell_count = targets.count(0)
        hold_count = targets.count(1)
        
        logger.info(f"      ✓ Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")
        
        return (np.array(targets), np.array(confidences), 
                sl_tp_info, np.array(trade_quality))
    
    def evaluate_trade_quality(self, df, idx, position_type, sl_tp, max_up, max_down):
        """تقييم جودة الصفقة"""
        quality_score = 0.5
        
        # Risk/Reward
        if sl_tp and sl_tp['take_profits']:
            avg_rr = np.mean([abs(tp - df['close'].iloc[idx]) / sl_tp['risk_amount'] 
                            for tp in sl_tp['take_profits']])
            if avg_rr >= 3:
                quality_score += 0.2
            elif avg_rr >= 2:
                quality_score += 0.1
        
        # وضوح الاتجاه
        if position_type == 'long':
            trend_clarity = max_up / (max_down + 1)
        else:
            trend_clarity = max_down / (max_up + 1)
        
        if trend_clarity >= 3:
            quality_score += 0.15
        elif trend_clarity >= 2:
            quality_score += 0.1
        
        # RSI
        rsi_col = 'rsi_14'
        if rsi_col in df.columns:
            rsi = df[rsi_col].iloc[idx]
            if position_type == 'long' and 30 < rsi < 70:
                quality_score += 0.1
            elif position_type == 'short' and 30 < rsi < 70:
                quality_score += 0.1
        
        # التقلب
        if sl_tp and 'atr_used' in sl_tp:
            atr_ratio = sl_tp['atr_used'] / df['close'].iloc[idx]
            if 0.001 < atr_ratio < 0.02:
                quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    def balance_data(self, X, y, confidence, quality):
        """موازنة البيانات مع فلترة الجودة"""
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # فلترة عالية الجودة والثقة
        high_quality = (confidence > 0.7) & (quality > 0.7)
        X_high = X[high_quality]
        y_high = y[high_quality]
        
        if len(X_high) < 100:
            logger.warning("⚠️ عينات قليلة، استخدام كل البيانات")
            X_high = X[confidence > 0.6]
            y_high = y[confidence > 0.6]
        
        try:
            over = SMOTE(sampling_strategy=0.8, random_state=42)
            under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            pipeline = Pipeline([('o', over), ('u', under)])
            
            X_balanced, y_balanced = pipeline.fit_resample(X_high, y_high)
            logger.info(f"      ✓ تمت الموازنة: {len(X_balanced)} عينة")
            return X_balanced, y_balanced
        except:
            return X_high, y_high
    
    def train_single_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """تدريب نموذج واحد"""
        try:
            if model_name == 'lightgbm':
                model = lgb.LGBMClassifier(**model_config)
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                         
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(**model_config)
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=False)
                         
            elif model_name == 'catboost':
                model = CatBoostClassifier(**model_config)
                model.fit(X_train, y_train,
                         eval_set=(X_val, y_val),
                         early_stopping_rounds=50,
                         verbose=False)
                         
            elif model_name == 'extra_trees':
                model = ExtraTreesClassifier(**model_config)
                model.fit(X_train, y_train)
                
            elif model_name == 'neural_network':
                model = MLPClassifier(**model_config)
                model.fit(X_train, y_train)
            
            val_score = model.score(X_val, y_val)
            return model_name, model, val_score
            
        except Exception as e:
            logger.error(f"      ✗ فشل {model_name}: {e}")
            return model_name, None, 0
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """تدريب جميع النماذج"""
        logger.info("    🎯 تدريب النماذج...")
        
        if not self.use_all_models:
            # LightGBM فقط
            model = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return model, {'accuracy': accuracy, 'models_count': 1}
        
        # تدريب 5 نماذج
        trained_models = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for model_name, config in self.model_configs.items():
                future = executor.submit(
                    self.train_single_model,
                    model_name, config,
                    X_train, y_train,
                    X_val, y_val
                )
                futures.append(future)
            
            for future in as_completed(futures):
                model_name, model, score = future.result()
                if model is not None:
                    trained_models.append((model_name, model))
                    logger.info(f"      ✓ {model_name}: {score:.4f}")
        
        if len(trained_models) < 2:
            # استخدام أفضل نموذج
            if trained_models:
                best_model = trained_models[0][1]
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                return best_model, {'accuracy': accuracy, 'models_count': 1}
            else:
                return None, {'accuracy': 0, 'models_count': 0}
        
        # نموذج مجمع
        logger.info("    🔗 إنشاء النموذج المجمع...")
        ensemble = VotingClassifier(trained_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # تقييم
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # دقة الصفقات
        trade_mask = y_pred != 1
        trade_accuracy = accuracy_score(y_test[trade_mask], y_pred[trade_mask]) if trade_mask.sum() > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'trade_accuracy': trade_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'models_count': len(trained_models)
        }
        
        logger.info(f"    ✅ النموذج المجمع: دقة {accuracy:.4f} ({len(trained_models)} نماذج)")
        
        return ensemble, results
    
    def train_strategy(self, X, y, confidence, quality, sl_tp_info, feature_names, 
                      strategy_name, strategy, symbol, timeframe):
        """تدريب استراتيجية واحدة"""
        logger.info(f"\n  📊 استراتيجية {strategy_name} - {strategy['description']}")
        
        # موازنة البيانات
        X_balanced, y_balanced = self.balance_data(X, y, confidence, quality)
        
        # تقسيم
        split1 = int(len(X_balanced) * 0.7)
        split2 = int(len(X_balanced) * 0.85)
        
        X_train = X_balanced[:split1]
        X_val = X_balanced[split1:split2]
        X_test = X_balanced[split2:]
        
        y_train = y_balanced[:split1]
        y_val = y_balanced[split1:split2]
        y_test = y_balanced[split2:]
        
        # معايرة
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب
        model, results = self.train_ensemble_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        if model is None:
            return None
        
        # حفظ النموذج بغض النظر عن الدقة
        model_dir = Path(f"models/{symbol}_{timeframe}/{strategy_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'strategy': strategy,
            'results': results,
            'sl_tp_settings': self.sl_tp_settings[strategy_name],
            'training_date': datetime.now(),
            'use_all_features': self.use_all_features,
            'use_all_models': self.use_all_models
        }
        
        # حفظ النموذج الرئيسي
        model_path = model_dir / 'model_ultimate.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"    💾 تم حفظ النموذج في: {model_path}")
        
        # حفظ نسخة احتياطية مع التاريخ
        backup_path = model_dir / f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        joblib.dump(model_data, backup_path)
        
        # إضافة مسار النموذج للنتائج
        results['model_path'] = str(model_path)
        results['model'] = model
        results['scaler'] = scaler
        
        # حفظ ملف معلومات النموذج
        info_path = model_dir / 'model_info.json'
        info_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy_name': strategy_name,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'models_count': results['models_count'],
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_names),
            'training_samples': len(X_train)
        }
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        logger.info(f"    ✅ النموذج محفوظ: دقة {results['accuracy']:.4f}")
        
        return results
    
    def train_symbol(self, symbol, timeframe):
        """تدريب عملة واحدة"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 تدريب {symbol} {timeframe}")
        logger.info(f"{'='*80}")
        
        # تحميل البيانات
        df = self.load_data_advanced(symbol, timeframe)
        if df is None:
            return None
        
        # إنشاء الميزات
        features = self.create_ultra_advanced_features(df, symbol)
        X = features.values
        feature_names = features.columns.tolist()
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategies': {},
            'best_accuracy': 0,
            'best_strategy': None
        }
        
        # تدريب كل استراتيجية
        for strategy_name, strategy in self.training_strategies.items():
            try:
                # إنشاء الأهداف مع SL/TP
                y, confidence, sl_tp_info, quality = self.create_advanced_targets_with_sl_tp(df, strategy_name)
                
                # تدريب
                strategy_results = self.train_strategy(
                    X, y, confidence, quality, sl_tp_info, feature_names,
                    strategy_name, strategy, symbol, timeframe
                )
                
                if strategy_results:
                    results['strategies'][strategy_name] = strategy_results
                    
                    if strategy_results['accuracy'] > results['best_accuracy']:
                        results['best_accuracy'] = strategy_results['accuracy']
                        results['best_strategy'] = strategy_name
                    
                    logger.info(f"  ✅ {strategy_name}: {strategy_results['accuracy']:.4f}")
                
                gc.collect()
                
            except Exception as e:
                logger.error(f"  ❌ خطأ في {strategy_name}: {e}")
        
        # ملخص
        logger.info(f"\n📊 ملخص {symbol} {timeframe}:")
        logger.info(f"🏆 أفضل: {results['best_strategy']} - {results['best_accuracy']:.4f}")
        
        return results
    
    def train_all(self):
        """تدريب جميع العملات"""
        logger.info("\n" + "="*100)
        logger.info("🚀 بدء التدريب الشامل النهائي")
        logger.info("="*100)
        
        if not self.check_database():
            return
        
        # الحصول على العملات
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            query = """
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY symbol, timeframe
            """
            available = pd.read_sql_query(query, conn, params=(self.min_data_points,))
            conn.close()
            
            logger.info(f"📊 عدد الأزواج: {len(available)}")
            
        except Exception as e:
            logger.error(f"❌ خطأ: {e}")
            return
        
        # النتائج
        excellent = []  # 90%+
        good = []       # 85-90%
        acceptable = [] # 80-85%
        failed = []
        
        # تدريب
        for idx, row in available.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            logger.info(f"\n📈 معالجة {idx+1}/{len(available)}: {symbol} {timeframe}")
            
            try:
                result = self.train_symbol(symbol, timeframe)
                
                if result:
                    acc = result['best_accuracy']
                    
                    model_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': acc,
                        'strategy': result['best_strategy'],
                        'strategies': result['strategies']
                    }
                    
                    if acc >= 0.90:
                        excellent.append(model_info)
                    elif acc >= 0.85:
                        good.append(model_info)
                    elif acc >= 0.80:
                        acceptable.append(model_info)
                    else:
                        failed.append(f"{symbol} {timeframe}")
                
                self.monitor_memory()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ: {e}")
                failed.append(f"{symbol} {timeframe}")
        
        # التقرير
        self._print_report(excellent, good, acceptable, failed)
    
    def _print_report(self, excellent, good, acceptable, failed):
        """طباعة التقرير النهائي"""
        logger.info("\n" + "="*100)
        logger.info("📊 التقرير النهائي - النظام المتقدم النهائي")
        logger.info("="*100)
        
        total = len(excellent) + len(good) + len(acceptable) + len(failed)
        
        logger.info(f"\n📈 الإحصائيات:")
        logger.info(f"  🌟 ممتاز (90%+): {len(excellent)}")
        logger.info(f"  ✅ جيد (85-90%): {len(good)}")
        logger.info(f"  👍 مقبول (80-85%): {len(acceptable)}")
        logger.info(f"  ❌ فشل (<80%): {len(failed)}")
        
        if total > 0:
            success_rate = (len(excellent) + len(good)) / total * 100
            logger.info(f"\n🎯 معدل النجاح (85%+): {success_rate:.1f}%")
        
        # أفضل النماذج
        if excellent:
            logger.info(f"\n🌟 أفضل النماذج:")
            for m in sorted(excellent, key=lambda x: x['accuracy'], reverse=True)[:10]:
                logger.info(f"  • {m['symbol']} {m['timeframe']}: {m['accuracy']:.4f} ({m['strategy']})")
        
        # حفظ التقرير
        report = {
            'date': datetime.now().isoformat(),
            'configuration': {
                'use_all_features': self.use_all_features,
                'use_all_models': self.use_all_models,
                'features_count': '200+' if self.use_all_features else 'basic',
                'models_count': 5 if self.use_all_models else 1
            },
            'results': {
                'total': total,
                'excellent': len(excellent),
                'good': len(good),
                'acceptable': len(acceptable),
                'failed': len(failed),
                'success_rate': success_rate if total > 0 else 0
            },
            'best_models': excellent[:20],
            'strategies_used': list(self.training_strategies.keys()),
            'sl_tp_settings': self.sl_tp_settings
        }
        
        report_path = Path("models/ultimate_training_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n💾 تم حفظ التقرير: {report_path}")

def main():
    """الدالة الرئيسية"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Advanced Training System')
    parser.add_argument('--basic-features', action='store_true', help='Use basic features only')
    parser.add_argument('--single-model', action='store_true', help='Use LightGBM only')
    parser.add_argument('--symbol', type=str, help='Train specific symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    args = parser.parse_args()
    
    # الإعدادات
    use_all_features = not args.basic_features
    use_all_models = not args.single_model
    
    trainer = UltimateAdvancedTrainer(
        use_all_features=use_all_features,
        use_all_models=use_all_models
    )
    
    if args.symbol:
        # تدريب عملة واحدة
        trainer.train_symbol(args.symbol, args.timeframe)
    else:
        # تدريب شامل
        trainer.train_all()

if __name__ == "__main__":
    main()