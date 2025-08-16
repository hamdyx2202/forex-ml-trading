#!/usr/bin/env python3
"""
Advanced Complete Training System - نظام التدريب المتقدم الكامل
هدف: تحقيق نسبة نجاح 95%+ باستخدام جميع التقنيات المتقدمة
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from ta import add_all_ta_features
from ta.utils import dropna
import talib

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/advanced_training.log", rotation="1 day", retention="30 days")

class AdvancedCompleteTrainer:
    """نظام التدريب المتقدم الكامل للوصول لنسبة نجاح 95%+"""
    
    def __init__(self):
        self.min_data_points = 10000  # بيانات أكثر للدقة العالية
        self.test_size = 0.2
        self.validation_split = 0.15
        self.random_state = 42
        
        # استراتيجيات متعددة للتدريب
        self.training_strategies = {
            'ultra_short': {'lookahead': 5, 'min_pips': 3, 'confidence_threshold': 0.95},
            'scalping': {'lookahead': 15, 'min_pips': 5, 'confidence_threshold': 0.92},
            'short_term': {'lookahead': 30, 'min_pips': 10, 'confidence_threshold': 0.90},
            'medium_term': {'lookahead': 60, 'min_pips': 20, 'confidence_threshold': 0.88},
            'long_term': {'lookahead': 240, 'min_pips': 40, 'confidence_threshold': 0.85}
        }
        
        # إعدادات Stop Loss و Take Profit الديناميكية
        self.sl_tp_settings = {
            'ultra_short': {
                'stop_loss_atr': 0.5,      # نصف ATR للسكالبينج السريع
                'take_profit_ratios': [1.0, 1.5, 2.0],  # TP متعدد
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

        
        # إعدادات النماذج المتقدمة
        self.ensemble_models = {
            'lightgbm_primary': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'dart',  # DART للدقة العالية
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
                'subsample_for_bin': 200000,
                'class_weight': 'balanced',
                'verbosity': -1
            },
            'xgboost_primary': {
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
                'scale_pos_weight': 1,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'tree_method': 'hist'
            },
            'catboost_primary': {
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
                'class_weights': [1, 1, 1],
                'verbose': False
            },
            'extra_trees': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'n_jobs': -1
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
        
    def check_database(self):
        """التحقق من وجود قاعدة البيانات والبيانات"""
        db_path = Path("data/forex_ml.db")
        
        if not db_path.exists():
            logger.error("❌ قاعدة البيانات غير موجودة!")
            logger.info("🔧 يرجى تشغيل: python setup_database.py")
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0:
                logger.error("❌ لا توجد بيانات في قاعدة البيانات!")
                logger.info("📊 يرجى جمع البيانات من MT5 أولاً")
                return False
            
            logger.info(f"✅ قاعدة البيانات جاهزة: {count:,} سجل")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في قاعدة البيانات: {e}")
            return False
    
    def load_data_advanced(self, symbol, timeframe, limit=100000):
        """تحميل البيانات مع معالجة متقدمة"""
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
                logger.warning(f"⚠️ بيانات غير كافية لـ {symbol} {timeframe}: {len(df)} سجل")
                return None
            
            # تحويل الوقت
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
            # التأكد من عدم وجود قيم مفقودة
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل لـ {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def create_ultra_advanced_features(self, df, symbol):
        """إنشاء ميزات متقدمة جداً - 200+ ميزة"""
        logger.info("🔬 إنشاء ميزات متقدمة...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. ميزات الأسعار الأساسية
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # 2. المتوسطات المتحركة المتعددة
        ma_periods = [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233]
        for period in ma_periods:
            sma = df['close'].rolling(period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            
            features[f'sma_{period}'] = (df['close'] - sma) / sma
            features[f'ema_{period}'] = (df['close'] - ema) / ema
            features[f'sma_{period}_slope'] = sma.pct_change(5)
            features[f'ema_{period}_slope'] = ema.pct_change(5)
        
        # 3. مؤشرات TA-Lib المتقدمة
        try:
            # RSI متعدد
            for period in [5, 7, 9, 14, 21, 28]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            
            # Stochastic
            for period in [5, 9, 14]:
                slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                          fastk_period=period, slowk_period=3, slowd_period=3)
                features[f'stoch_k_{period}'] = slowk
                features[f'stoch_d_{period}'] = slowd
            
            # MACD متعدد
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (3, 10, 16)]:
                macd, macdsignal, macdhist = talib.MACD(df['close'].values, 
                                                        fastperiod=fast, 
                                                        slowperiod=slow, 
                                                        signalperiod=signal)
                features[f'macd_{fast}_{slow}'] = macd
                features[f'macd_signal_{fast}_{slow}'] = macdsignal
                features[f'macd_hist_{fast}_{slow}'] = macdhist
            
            # Bollinger Bands متعدد
            for period in [10, 20, 30]:
                for dev in [1.5, 2.0, 2.5]:
                    upper, middle, lower = talib.BBANDS(df['close'].values,
                                                       timeperiod=period,
                                                       nbdevup=dev,
                                                       nbdevdn=dev)
                    features[f'bb_upper_{period}_{dev}'] = (upper - df['close']) / df['close']
                    features[f'bb_lower_{period}_{dev}'] = (df['close'] - lower) / df['close']
                    features[f'bb_width_{period}_{dev}'] = (upper - lower) / middle
            
            # ATR وVolatility
            for period in [7, 10, 14, 20, 30]:
                features[f'atr_{period}'] = talib.ATR(df['high'].values, 
                                                     df['low'].values, 
                                                     df['close'].values, 
                                                     timeperiod=period) / df['close']
                features[f'natr_{period}'] = talib.NATR(df['high'].values, 
                                                       df['low'].values, 
                                                       df['close'].values, 
                                                       timeperiod=period)
            
            # ADX وDI
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
            
            # مؤشرات أخرى
            features['cci_14'] = talib.CCI(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=14)
            features['mfi_14'] = talib.MFI(df['high'].values, df['low'].values, 
                                          df['close'].values, df['volume'].values, timeperiod=14)
            features['roc_10'] = talib.ROC(df['close'].values, timeperiod=10)
            features['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, 
                                                df['close'].values, timeperiod=14)
            features['ultimate_osc'] = talib.ULTOSC(df['high'].values, df['low'].values, 
                                                   df['close'].values)
            
            # Pattern Recognition
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
            logger.warning(f"⚠️ بعض مؤشرات TA-Lib فشلت: {e}")
        
        # 4. ميزات الحجم المتقدمة
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_sma_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        
        # Force Index
        features['force_index'] = df['close'].diff() * df['volume']
        features['force_index_ema'] = features['force_index'].ewm(span=13).mean()
        
        # 5. ميزات التقلب المتقدمة
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
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # جلسات التداول بتوقيت أكثر دقة
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
        features['london_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        features['sydney_session'] = ((features['hour'] >= 22) | (features['hour'] < 7)).astype(int)
        
        # تداخل الجلسات
        features['london_ny_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 17)).astype(int)
        features['asian_london_overlap'] = ((features['hour'] >= 8) & (features['hour'] < 9)).astype(int)
        
        # 7. ميزات الارتباط والعلاقات
        # علاقات الأسعار
        for period in [5, 10, 20]:
            features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            features[f'close_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'close_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()
        
        # 8. ميزات الزخم المتقدمة
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features[f'acceleration_{period}'] = features[f'momentum_{period}'].diff()
        
        # 9. مستويات الدعم والمقاومة الديناميكية
        for period in [20, 50, 100]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            features[f'distance_from_high_{period}'] = (rolling_high - df['close']) / df['close']
            features[f'distance_from_low_{period}'] = (df['close'] - rolling_low) / df['close']
            features[f'position_in_range_{period}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)
        
        # 10. ميزات إحصائية متقدمة
        for period in [10, 20, 50]:
            rolling_returns = df['close'].pct_change().rolling(period)
            features[f'skewness_{period}'] = rolling_returns.skew()
            features[f'kurtosis_{period}'] = rolling_returns.kurt()
            features[f'mean_return_{period}'] = rolling_returns.mean()
            features[f'median_return_{period}'] = rolling_returns.median()
        
        # تنظيف البيانات
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة متقدمة")
        
        return features
    
    
    def calculate_pip_value(self, symbol):
        """حساب قيمة النقطة حسب العملة"""
        if 'JPY' in symbol or 'XAG' in symbol:
            return 0.01
        elif 'XAU' in symbol:
            return 0.1
        else:
            return 0.0001
    
    def calculate_dynamic_sl_tp(self, df, position_type, entry_idx, strategy_name):
        """حساب Stop Loss و Take Profit ديناميكي متقدم"""
        
        sl_tp_config = self.sl_tp_settings.get(strategy_name, self.sl_tp_settings['medium_term'])
        
        # الحصول على ATR
        atr_period = 14
        if f'atr_{atr_period}' in df.columns:
            current_atr = df.iloc[entry_idx][f'atr_{atr_period}']
        else:
            # حساب ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            current_atr = true_range.rolling(atr_period).mean().iloc[entry_idx]
        
        current_price = df['close'].iloc[entry_idx]
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        # Stop Loss ديناميكي
        sl_distance = current_atr * sl_tp_config['stop_loss_atr']
        
        # Take Profit متعدد المستويات
        tp_distances = [sl_distance * ratio for ratio in sl_tp_config['take_profit_ratios']]
        
        # Support/Resistance
        support_resistance = self.find_support_resistance_levels(df, entry_idx)
        
        if position_type == 'long':
            # Stop Loss
            stop_loss = current_price - sl_distance
            
            # تعديل SL بناءً على الدعم
            nearest_support = min([s for s in support_resistance['support'] 
                                 if s < current_price and s > stop_loss], 
                                default=stop_loss)
            if nearest_support > stop_loss:
                stop_loss = nearest_support - (2 * pip_value)
            
            # Take Profit levels
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price + tp_distance
                
                # تعديل TP بناءً على المقاومة
                nearest_resistance = min([r for r in support_resistance['resistance'] 
                                        if r > current_price and r < tp], 
                                       default=tp)
                if nearest_resistance < tp:
                    tp = nearest_resistance - (2 * pip_value)
                
                take_profits.append(tp)
            
            # Trailing Stop
            trailing_stop_distance = current_atr * sl_tp_config['trailing_stop_atr']
            
        else:  # Short
            # Stop Loss
            stop_loss = current_price + sl_distance
            
            # تعديل SL بناءً على المقاومة
            nearest_resistance = max([r for r in support_resistance['resistance'] 
                                    if r > current_price and r < stop_loss], 
                                   default=stop_loss)
            if nearest_resistance < stop_loss:
                stop_loss = nearest_resistance + (2 * pip_value)
            
            # Take Profit levels
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price - tp_distance
                
                # تعديل TP بناءً على الدعم
                nearest_support = max([s for s in support_resistance['support'] 
                                     if s < current_price and s > tp], 
                                    default=tp)
                if nearest_support > tp:
                    tp = nearest_support + (2 * pip_value)
                
                take_profits.append(tp)
            
            # Trailing Stop
            trailing_stop_distance = current_atr * sl_tp_config['trailing_stop_atr']
        
        return {
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'trailing_stop_distance': trailing_stop_distance,
            'breakeven_pips': sl_tp_config['breakeven_pips'],
            'risk_amount': abs(current_price - stop_loss),
            'reward_ratios': sl_tp_config['take_profit_ratios'],
            'atr_used': current_atr
        }
    
    def find_support_resistance_levels(self, df, current_idx, lookback=150):
        """إيجاد مستويات الدعم والمقاومة المتقدمة"""
        start_idx = max(0, current_idx - lookback)
        price_data = df[['high', 'low', 'close']].iloc[start_idx:current_idx]
        
        # إيجاد القمم والقيعان
        highs = []
        lows = []
        
        # استخدام خوارزمية متقدمة للكشف عن القمم والقيعان
        window = 5
        for i in range(window, len(price_data) - window):
            # قمة محلية
            if (price_data['high'].iloc[i] == price_data['high'].iloc[i-window:i+window+1].max()):
                highs.append(price_data['high'].iloc[i])
            
            # قاع محلي
            if (price_data['low'].iloc[i] == price_data['low'].iloc[i-window:i+window+1].min()):
                lows.append(price_data['low'].iloc[i])
        
        # إضافة مستويات من Fibonacci
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
            
            # متوسط كل مجموعة مع وزن للتكرار
            result = []
            for cluster in clusters:
                weight = len(cluster)  # كلما زاد التكرار زادت القوة
                avg_level = sum(cluster) / len(cluster)
                result.append({'level': avg_level, 'strength': weight})
            
            # ترتيب حسب القوة
            result.sort(key=lambda x: x['strength'], reverse=True)
            
            # إرجاع أقوى المستويات فقط
            return [item['level'] for item in result[:10]]
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def create_advanced_targets_with_sl_tp(self, df, strategy_name):
        """إنشاء أهداف متقدمة مع معلومات SL/TP كاملة"""
        strategy = self.training_strategies[strategy_name]
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        targets = []
        confidences = []
        sl_tp_info = []
        trade_quality = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            future_highs = df['high'].iloc[i+1:i+lookahead+1].values
            future_lows = df['low'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            # حساب أقصى حركة مع مراعاة الـ wicks
            max_up = (future_highs.max() - current_price) / pip_value
            max_down = (current_price - future_lows.min()) / pip_value
            
            # حساب الحركة الفعلية للإغلاق
            close_up = (future_prices.max() - current_price) / pip_value
            close_down = (current_price - future_prices.min()) / pip_value
            
            # تحديد نوع الصفقة مع معايير محسنة
            if max_up >= min_pips * 2 and close_up >= min_pips * 1.5:
                # Long signal
                targets.append(2)
                
                # حساب الثقة بناءً على عوامل متعددة
                confidence = min(
                    0.5 + (close_up / (min_pips * 4)) * 0.3 +  # قوة الحركة
                    (1 - max_down / max_up) * 0.2,  # نسبة الصعود للهبوط
                    1.0
                )
                confidences.append(confidence)
                
                # حساب SL/TP
                sl_tp = self.calculate_dynamic_sl_tp(df, 'long', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                # تقييم جودة الصفقة
                quality = self.evaluate_trade_quality(df, i, 'long', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            elif max_down >= min_pips * 2 and close_down >= min_pips * 1.5:
                # Short signal
                targets.append(0)
                
                # حساب الثقة
                confidence = min(
                    0.5 + (close_down / (min_pips * 4)) * 0.3 +
                    (1 - max_up / max_down) * 0.2,
                    1.0
                )
                confidences.append(confidence)
                
                # حساب SL/TP
                sl_tp = self.calculate_dynamic_sl_tp(df, 'short', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                # تقييم جودة الصفقة
                quality = self.evaluate_trade_quality(df, i, 'short', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            else:
                # No trade
                targets.append(1)
                confidences.append(0.5)
                sl_tp_info.append(None)
                trade_quality.append(0)
        
        # ملء القيم الأخيرة
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        sl_tp_info.extend([None] * lookahead)
        trade_quality.extend([0] * lookahead)
        
        return (np.array(targets), np.array(confidences), 
                sl_tp_info, np.array(trade_quality))
    
    def evaluate_trade_quality(self, df, idx, position_type, sl_tp, max_up, max_down):
        """تقييم جودة الصفقة بناءً على معايير متعددة"""
        quality_score = 0.5  # البداية من المنتصف
        
        # 1. Risk/Reward Ratio
        if sl_tp and sl_tp['take_profits']:
            avg_rr = np.mean([abs(tp - df['close'].iloc[idx]) / sl_tp['risk_amount'] 
                            for tp in sl_tp['take_profits']])
            if avg_rr >= 3:
                quality_score += 0.2
            elif avg_rr >= 2:
                quality_score += 0.1
        
        # 2. وضوح الاتجاه
        if position_type == 'long':
            trend_clarity = max_up / (max_down + 1)  # تجنب القسمة على صفر
        else:
            trend_clarity = max_down / (max_up + 1)
        
        if trend_clarity >= 3:
            quality_score += 0.15
        elif trend_clarity >= 2:
            quality_score += 0.1
        
        # 3. قوة المؤشرات الفنية
        rsi_col = 'rsi_14'
        if rsi_col in df.columns:
            rsi = df[rsi_col].iloc[idx]
            if position_type == 'long' and rsi < 70 and rsi > 30:
                quality_score += 0.1
            elif position_type == 'short' and rsi > 30 and rsi < 70:
                quality_score += 0.1
        
        # 4. التقلب المناسب
        if sl_tp and 'atr_used' in sl_tp:
            atr_ratio = sl_tp['atr_used'] / df['close'].iloc[idx]
            if 0.001 < atr_ratio < 0.02:  # تقلب معتدل
                quality_score += 0.05
        
        return min(quality_score, 1.0)

    def create_advanced_targets(self, df, strategy):
        """إنشاء أهداف متقدمة متعددة الاستراتيجيات"""
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        # حساب pip value حسب العملة
        if 'JPY' in df.index.name or 'JPY' in str(df.index.name):
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        targets = []
        confidences = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            # حساب أقصى حركة صعودية وهبوطية
            max_up = (future_prices.max() - current_price) / pip_value
            max_down = (current_price - future_prices.min()) / pip_value
            
            # حساب الهدف والثقة
            if max_up >= min_pips * 2:  # صعود قوي
                targets.append(2)
                confidences.append(min(max_up / (min_pips * 3), 1.0))
            elif max_down >= min_pips * 2:  # هبوط قوي
                targets.append(0)
                confidences.append(min(max_down / (min_pips * 3), 1.0))
            else:  # محايد أو حركة ضعيفة
                targets.append(1)
                confidences.append(0.5)
        
        # ملء القيم الأخيرة
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        
        return np.array(targets), np.array(confidences)
    
    def balance_dataset(self, X, y, confidence):
        """موازنة البيانات للحصول على توزيع متساوي"""
        # الاحتفاظ بالعينات عالية الثقة فقط
        high_conf_mask = confidence > 0.7
        X_high = X[high_conf_mask]
        y_high = y[high_conf_mask]
        
        # موازنة الفئات
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # استراتيجية الموازنة
        over = SMOTE(sampling_strategy=0.8, random_state=42)
        under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        
        try:
            X_balanced, y_balanced = pipeline.fit_resample(X_high, y_high)
            logger.info(f"✅ تمت موازنة البيانات: {len(X_balanced)} عينة")
            return X_balanced, y_balanced
        except:
            logger.warning("⚠️ فشلت الموازنة، استخدام البيانات الأصلية")
            return X_high, y_high
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, model_type='lightgbm'):
        """تحسين المعاملات باستخدام Optuna للوصول لأفضل أداء"""
        logger.info(f"🔧 تحسين معاملات {model_type}...")
        
        def objective(trial):
            if model_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                    'num_leaves': trial.suggest_int('num_leaves', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
                    'verbosity': -1
                }
                
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss'
                }
                
                model = xgb.XGBClassifier(**params)
            
            # تدريب وتقييم
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # نريد أيضاً precision عالية
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # الهدف: مزيج من الدقة والprecision
            return accuracy * 0.7 + precision * 0.3
        
        # تشغيل التحسين
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)
        
        logger.info(f"✅ أفضل أداء: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_ensemble_95plus(self, X_train, y_train, X_val, y_val, optimized_params):
        """تدريب نموذج مجمع للوصول لدقة 95%+"""
        logger.info("🎯 تدريب النموذج المجمع المتقدم...")
        
        models = []
        
        # 1. LightGBM محسن
        lgb_params = self.ensemble_models['lightgbm_primary'].copy()
        lgb_params.update(optimized_params.get('lightgbm', {}))
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        models.append(('lightgbm', lgb_model))
        
        # 2. XGBoost محسن
        xgb_params = self.ensemble_models['xgboost_primary'].copy()
        xgb_params.update(optimized_params.get('xgboost', {}))
        xgb_model = xgb.XGBClassifier(**xgb_params)
        models.append(('xgboost', xgb_model))
        
        # 3. CatBoost
        cat_model = CatBoostClassifier(**self.ensemble_models['catboost_primary'])
        models.append(('catboost', cat_model))
        
        # 4. Extra Trees
        et_model = ExtraTreesClassifier(**self.ensemble_models['extra_trees'])
        models.append(('extra_trees', et_model))
        
        # 5. Neural Network
        nn_model = MLPClassifier(**self.ensemble_models['neural_network'])
        models.append(('neural_net', nn_model))
        
        # نموذج مجمع متقدم
        ensemble = VotingClassifier(models, voting='soft', n_jobs=-1)
        
        # تدريب مع تتبع الأداء
        logger.info("⏳ جاري التدريب (قد يستغرق وقتاً)...")
        ensemble.fit(X_train, y_train)
        
        # تقييم مفصل
        train_pred = ensemble.predict(X_train)
        val_pred = ensemble.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # تقييم كل نموذج منفرد
        individual_scores = {}
        for name, model in models:
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            individual_scores[name] = score
            logger.info(f"  • {name}: {score:.4f}")
        
        logger.info(f"✅ دقة التدريب: {train_acc:.4f}")
        logger.info(f"✅ دقة التحقق: {val_acc:.4f}")
        
        return ensemble, {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'individual_scores': individual_scores
        }
    
    def train_symbol_advanced(self, symbol, timeframe):
        """تدريب متقدم لعملة واحدة"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 تدريب متقدم لـ {symbol} {timeframe}")
        logger.info(f"{'='*80}")
        
        # تحميل البيانات
        df = self.load_data_advanced(symbol, timeframe)
        if df is None:
            return None
        
        # إنشاء الميزات المتقدمة
        features = self.create_ultra_advanced_features(df, symbol)
        
        # إعداد البيانات للتدريب
        X = features.values
        feature_names = features.columns.tolist()
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategies': {},
            'best_accuracy': 0,
            'best_strategy': None
        }
        
        # تدريب لكل استراتيجية
        for strategy_name, strategy in self.training_strategies.items():
            logger.info(f"\n📊 استراتيجية: {strategy_name}")
            
            # إنشاء الأهداف
            # إنشاء الأهداف مع SL/TP
            if hasattr(self, 'create_advanced_targets_with_sl_tp'):
                y, confidence, sl_tp_info, quality = self.create_advanced_targets_with_sl_tp(df, strategy_name)
                
                # فلترة الصفقات عالية الجودة فقط
                high_quality_mask = quality > 0.7
                X_balanced, y_balanced = self.balance_dataset(X[high_quality_mask], 
                                                             y[high_quality_mask], 
                                                             confidence[high_quality_mask])
            else:
                # الطريقة القديمة كـ fallback
                y, confidence = self.create_advanced_targets(df, strategy)
                X_balanced, y_balanced = self.balance_dataset(X, y, confidence)

            
            # موازنة البيانات
            X_balanced, y_balanced = self.balance_dataset(X, y, confidence)
            
            # تقسيم البيانات
            split_idx = int(len(X_balanced) * 0.7)
            X_temp = X_balanced[:split_idx]
            X_test = X_balanced[split_idx:]
            y_temp = y_balanced[:split_idx]
            y_test = y_balanced[split_idx:]
            
            split_idx2 = int(len(X_temp) * 0.85)
            X_train = X_temp[:split_idx2]
            X_val = X_temp[split_idx2:]
            y_train = y_temp[:split_idx2]
            y_val = y_temp[split_idx2:]
            
            # معايرة متقدمة
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # تحسين المعاملات
            optimized_params = {}
            if strategy_name in ['short_term', 'medium_term']:  # تحسين للاستراتيجيات المهمة
                lgb_params = self.optimize_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val, 'lightgbm'
                )
                optimized_params['lightgbm'] = lgb_params
            
            # تدريب النموذج المجمع
            model, scores = self.train_ensemble_95plus(
                X_train_scaled, y_train, X_val_scaled, y_val, optimized_params
            )
            
            # تقييم على بيانات الاختبار
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # تقييم إضافي - معدل النجاح للصفقات فقط
            trade_mask = y_pred != 1  # غير محايد
            if trade_mask.sum() > 0:
                trade_accuracy = accuracy_score(y_test[trade_mask], y_pred[trade_mask])
            else:
                trade_accuracy = 0
            
            logger.info(f"📈 دقة الاختبار: {test_accuracy:.4f}")
            logger.info(f"📊 دقة الصفقات: {trade_accuracy:.4f}")
            logger.info(f"🎯 Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
            
            # حفظ النتائج
            strategy_results = {
                'accuracy': test_accuracy,
                'trade_accuracy': trade_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'confidence_threshold': strategy['confidence_threshold'],
                'scores': scores
            }
            
            results['strategies'][strategy_name] = strategy_results
            
            # تحديث أفضل نتيجة
            if test_accuracy > results['best_accuracy']:
                results['best_accuracy'] = test_accuracy
                results['best_strategy'] = strategy_name
            
            # حفظ النموذج إذا كان الأداء ممتاز
            if test_accuracy >= 0.85:  # 85%+ 
                model_dir = Path(f"models/{symbol}_{timeframe}/{strategy_name}")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'strategy': strategy,
                    'results': strategy_results,
                    'training_date': datetime.now()
                }
                
                joblib.dump(model_data, model_dir / 'model_advanced.pkl')
                logger.info(f"💾 تم حفظ النموذج: {model_dir}")
        
        # طباعة الملخص
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ملخص نتائج {symbol} {timeframe}")
        logger.info(f"🏆 أفضل استراتيجية: {results['best_strategy']}")
        logger.info(f"🎯 أفضل دقة: {results['best_accuracy']:.4f}")
        
        return results
    
    def train_all_advanced(self):
        """تدريب جميع العملات المتاحة"""
        logger.info("\n" + "="*100)
        logger.info("🚀 بدء التدريب المتقدم الشامل - هدف 95%+ دقة")
        logger.info("="*100)
        
        # التحقق من قاعدة البيانات
        if not self.check_database():
            return
        
        # الحصول على العملات المتاحة
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
            
            logger.info(f"📊 عدد المجموعات المتاحة: {len(available)}")
            
        except Exception as e:
            logger.error(f"❌ خطأ في قراءة البيانات: {e}")
            return
        
        # نتائج التدريب
        excellent_models = []  # 90%+
        good_models = []       # 85-90%
        acceptable_models = [] # 80-85%
        failed_models = []
        
        # تدريب كل عملة
        for idx, row in available.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            try:
                logger.info(f"\n📊 معالجة {idx+1}/{len(available)}: {symbol} {timeframe}")
                
                results = self.train_symbol_advanced(symbol, timeframe)
                
                if results:
                    best_acc = results['best_accuracy']
                    
                    model_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': best_acc,
                        'strategy': results['best_strategy']
                    }
                    
                    if best_acc >= 0.90:
                        excellent_models.append(model_info)
                        logger.info(f"🌟 ممتاز! دقة {best_acc:.4f}")
                    elif best_acc >= 0.85:
                        good_models.append(model_info)
                        logger.info(f"✅ جيد جداً! دقة {best_acc:.4f}")
                    elif best_acc >= 0.80:
                        acceptable_models.append(model_info)
                        logger.info(f"👍 مقبول! دقة {best_acc:.4f}")
                    else:
                        failed_models.append(f"{symbol} {timeframe}")
                else:
                    failed_models.append(f"{symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"❌ خطأ في تدريب {symbol} {timeframe}: {str(e)}")
                failed_models.append(f"{symbol} {timeframe}: {str(e)}")
        
        # طباعة التقرير النهائي
        self._print_final_report(excellent_models, good_models, acceptable_models, failed_models)
    
    def _print_final_report(self, excellent, good, acceptable, failed):
        """طباعة التقرير النهائي المفصل"""
        logger.info("\n" + "="*100)
        logger.info("📊 التقرير النهائي للتدريب المتقدم")
        logger.info("="*100)
        
        total = len(excellent) + len(good) + len(acceptable) + len(failed)
        
        logger.info(f"\n📈 إحصائيات الأداء:")
        logger.info(f"  🌟 ممتاز (90%+): {len(excellent)} نموذج")
        logger.info(f"  ✅ جيد جداً (85-90%): {len(good)} نموذج")
        logger.info(f"  👍 مقبول (80-85%): {len(acceptable)} نموذج")
        logger.info(f"  ❌ فشل (<80%): {len(failed)} نموذج")
        
        success_rate = (len(excellent) + len(good)) / total * 100 if total > 0 else 0
        logger.info(f"\n🎯 معدل النجاح (85%+): {success_rate:.1f}%")
        
        if excellent:
            logger.info(f"\n🌟 النماذج الممتازة (90%+):")
            for model in sorted(excellent, key=lambda x: x['accuracy'], reverse=True)[:10]:
                logger.info(f"  • {model['symbol']} {model['timeframe']}: "
                          f"{model['accuracy']:.4f} ({model['strategy']})")
        
        if good:
            logger.info(f"\n✅ النماذج الجيدة جداً (85-90%):")
            for model in sorted(good, key=lambda x: x['accuracy'], reverse=True)[:10]:
                logger.info(f"  • {model['symbol']} {model['timeframe']}: "
                          f"{model['accuracy']:.4f} ({model['strategy']})")
        
        # حفظ التقرير
        report = {
            'training_date': datetime.now().isoformat(),
            'total_models': total,
            'excellent_count': len(excellent),
            'good_count': len(good),
            'acceptable_count': len(acceptable),
            'failed_count': len(failed),
            'success_rate': success_rate,
            'excellent_models': excellent,
            'good_models': good,
            'acceptable_models': acceptable,
            'configuration': {
                'min_data_points': self.min_data_points,
                'strategies': list(self.training_strategies.keys()),
                'ensemble_models': list(self.ensemble_models.keys())
            }
        }
        
        report_path = Path("models/advanced_training_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n💾 تم حفظ التقرير: {report_path}")
        
        if success_rate >= 70:
            logger.info("\n🎉 تهانينا! تم تحقيق نتائج ممتازة!")
        else:
            logger.info("\n💡 نصائح لتحسين الأداء:")
            logger.info("  • جمع المزيد من البيانات (3+ سنوات)")
            logger.info("  • تجربة فترات زمنية أطول")
            logger.info("  • ضبط معاملات التدريب")

def main():
    """الدالة الرئيسية"""
    trainer = AdvancedCompleteTrainer()
    
    # للاختبار السريع
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # تدريب عملة واحدة فقط
        trainer.train_symbol_advanced("EURUSD", "H1")
    else:
        # تدريب شامل
        trainer.train_all_advanced()

if __name__ == "__main__":
    main()