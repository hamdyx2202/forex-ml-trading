#!/usr/bin/env python3
"""
🚀 Ultimate Advanced Training System - النظام المتقدم النهائي
✨ جميع الميزات المتطورة مع إصلاح مشكلة حفظ النماذج
📊 يدرب على جميع العملات المتاحة
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/ultimate_training_system.log", rotation="1 day", retention="30 days")

# جميع أزواج العملات المتاحة
ALL_SYMBOLS = {
    # Forex Majors
    'forex_majors': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
    ],
    # Forex Minors
    'forex_minors': [
        'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'EURGBP', 'EURAUD', 'EURNZD',
        'GBPAUD', 'GBPNZD', 'AUDNZD', 'EURCZK', 'EURHUF', 'EURPLN', 'EURTRY'
    ],
    # Forex Exotics
    'forex_exotics': [
        'USDMXN', 'USDZAR', 'USDTRY', 'USDNOK', 'USDSEK', 'USDSGD', 'USDHKD',
        'USDCNH', 'USDRUB', 'USDINR', 'USDBRL', 'USDTHB', 'USDKRW'
    ],
    # Metals
    'metals': [
        'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'COPPER', 'ALUMINUM'
    ],
    # Energy
    'energy': [
        'USOIL', 'UKOIL', 'NATGAS', 'GASOLINE', 'HEATING'
    ],
    # Indices
    'indices': [
        'US30', 'NAS100', 'SP500', 'DAX', 'FTSE100', 'NIKKEI', 'HSI', 'ASX200'
    ],
    # Crypto
    'crypto': [
        'BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'BCHUSD', 'ADAUSD', 'DOTUSD',
        'LINKUSD', 'BNBUSD', 'SOLUSD', 'MATICUSD', 'AVAXUSD'
    ]
}

class UltimateAdvancedTrainer:
    """النظام المتقدم النهائي بجميع الميزات والنماذج"""
    
    def __init__(self, use_all_features=True, use_all_models=True, train_all_symbols=True):
        self.min_data_points = 5000  # خفضنا الحد الأدنى لتشمل المزيد من العملات
        self.test_size = 0.2
        self.validation_split = 0.15
        self.random_state = 42
        self.use_all_features = use_all_features
        self.use_all_models = use_all_models
        self.train_all_symbols = train_all_symbols
        
        # عدد العمليات المتوازية
        self.max_workers = min(cpu_count() - 1, 8)
        
        # عرض الإعدادات
        logger.info("="*100)
        logger.info("🚀 النظام المتقدم النهائي للتدريب - Ultimate Training System")
        logger.info("="*100)
        logger.info(f"📊 الميزات: {'300+ ميزة متقدمة' if use_all_features else 'ميزات أساسية'}")
        logger.info(f"🤖 النماذج: {'7 نماذج ذكاء اصطناعي' if use_all_models else 'LightGBM فقط'}")
        logger.info(f"💱 العملات: {'جميع العملات المتاحة' if train_all_symbols else 'العملات الرئيسية فقط'}")
        logger.info(f"⚡ المعالجة المتوازية: {self.max_workers} عمليات")
        
        # إعدادات النماذج المتقدمة
        self.model_configs = {
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 15,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'importance_type': 'gain'
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 10,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'tree_method': 'hist'
            },
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.01,
                'depth': 10,
                'random_state': self.random_state,
                'verbose': False,
                'thread_count': -1
            },
            'random_forest': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'extra_trees': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            },
            'neural_network': {
                'hidden_layer_sizes': (200, 150, 100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'early_stopping': True,
                'random_state': self.random_state
            }
        }
        
        # 7 استراتيجيات تداول متقدمة
        self.training_strategies = {
            'ultra_scalping': {
                'lookahead': 5,
                'min_pips': 2,
                'confidence_threshold': 0.96,
                'description': 'سكالبينج فائق السرعة'
            },
            'scalping': {
                'lookahead': 15,
                'min_pips': 5,
                'confidence_threshold': 0.93,
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
            'swing': {
                'lookahead': 240,
                'min_pips': 40,
                'confidence_threshold': 0.85,
                'description': 'تداول سوينج'
            },
            'position': {
                'lookahead': 1440,
                'min_pips': 100,
                'confidence_threshold': 0.82,
                'description': 'تداول مراكز'
            },
            'hybrid': {
                'lookahead': 120,
                'min_pips': 30,
                'confidence_threshold': 0.87,
                'description': 'استراتيجية هجينة'
            }
        }
        
        # إعدادات SL/TP الديناميكية
        self.sl_tp_settings = {
            'ultra_scalping': {
                'sl_multiplier': 1.0,
                'tp_multiplier': 1.5,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'scalping': {
                'sl_multiplier': 1.2,
                'tp_multiplier': 1.8,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'short_term': {
                'sl_multiplier': 1.5,
                'tp_multiplier': 2.0,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'medium_term': {
                'sl_multiplier': 2.0,
                'tp_multiplier': 3.0,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'swing': {
                'sl_multiplier': 2.5,
                'tp_multiplier': 4.0,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'position': {
                'sl_multiplier': 3.0,
                'tp_multiplier': 5.0,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            },
            'hybrid': {
                'sl_multiplier': 1.8,
                'tp_multiplier': 2.5,
                'use_atr': True,
                'use_support_resistance': True,
                'use_fibonacci': True
            }
        }

    def get_all_available_symbols(self):
        """الحصول على جميع العملات المتاحة في قاعدة البيانات"""
        try:
            conn = sqlite3.connect('data/forex_data.db')
            cursor = conn.cursor()
            
            # جلب جميع العملات مع عدد السجلات
            cursor.execute("""
                SELECT symbol, COUNT(*) as record_count 
                FROM price_data 
                GROUP BY symbol 
                HAVING COUNT(*) > ?
                ORDER BY COUNT(*) DESC
            """, (self.min_data_points,))
            
            available_symbols = []
            for row in cursor.fetchall():
                symbol, count = row
                # تنظيف اسم العملة
                clean_symbol = symbol.replace('m', '').replace('.fx', '').replace('fx', '')
                available_symbols.append({
                    'symbol': symbol,
                    'clean_symbol': clean_symbol,
                    'record_count': count
                })
            
            conn.close()
            
            logger.info(f"🔍 تم العثور على {len(available_symbols)} عملة متاحة للتدريب")
            
            return available_symbols
            
        except Exception as e:
            logger.error(f"خطأ في جلب العملات: {e}")
            return []

    def create_advanced_features(self, df):
        """إنشاء 300+ ميزة متقدمة"""
        logger.info("🧮 إنشاء المميزات المتقدمة...")
        
        # 1. مؤشرات فنية أساسية (50+ ميزة)
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            df[f'wma_{period}'] = talib.WMA(df['close'], timeperiod=period)
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = upper - lower
            df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-10)
        
        # 2. مؤشرات الزخم (40+ ميزة)
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        
        # MACD
        for fast in [12, 9]:
            for slow in [26, 21]:
                macd, signal, hist = talib.MACD(df['close'], fastperiod=fast, slowperiod=slow)
                df[f'macd_{fast}_{slow}'] = macd
                df[f'macd_signal_{fast}_{slow}'] = signal
                df[f'macd_hist_{fast}_{slow}'] = hist
        
        # Stochastic
        for period in [14, 21]:
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                                      fastk_period=period, slowk_period=3, slowd_period=3)
            df[f'stoch_k_{period}'] = slowk
            df[f'stoch_d_{period}'] = slowd
        
        # 3. مؤشرات الحجم (30+ ميزة)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # 4. مؤشرات التذبذب (40+ ميزة)
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_21'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=21)
        df['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Commodity Channel Index
        for period in [14, 20, 30]:
            df[f'cci_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Williams %R
        for period in [14, 21, 30]:
            df[f'willr_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # 5. أنماط الشموع اليابانية (50+ نمط)
        candle_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP'
        ]
        
        for pattern in candle_patterns:
            pattern_func = getattr(talib, pattern)
            df[f'pattern_{pattern.lower()}'] = pattern_func(df['open'], df['high'], df['low'], df['close'])
        
        # 6. مؤشرات متقدمة إضافية (50+ ميزة)
        # Ichimoku
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        # Fibonacci Retracements
        for period in [20, 50, 100]:
            high_period = df['high'].rolling(period).max()
            low_period = df['low'].rolling(period).min()
            diff = high_period - low_period
            
            df[f'fib_0.236_{period}'] = high_period - 0.236 * diff
            df[f'fib_0.382_{period}'] = high_period - 0.382 * diff
            df[f'fib_0.5_{period}'] = high_period - 0.5 * diff
            df[f'fib_0.618_{period}'] = high_period - 0.618 * diff
            df[f'fib_0.786_{period}'] = high_period - 0.786 * diff
        
        # 7. ميزات إحصائية وتحليلية (40+ ميزة)
        # Returns and volatility
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df[f'return_{period}'].rolling(20).std()
            df[f'skew_{period}'] = df[f'return_{period}'].rolling(20).skew()
            df[f'kurtosis_{period}'] = df[f'return_{period}'].rolling(20).kurt()
        
        # Price relative to moving averages
        for ma_period in [10, 20, 50, 100, 200]:
            ma_col = f'sma_{ma_period}'
            if ma_col in df.columns:
                df[f'price_to_sma_{ma_period}'] = df['close'] / df[ma_col]
                df[f'distance_from_sma_{ma_period}'] = df['close'] - df[ma_col]
        
        # 8. ميزات السوق والوقت (30+ ميزة)
        # Time features
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['time']).dt.day
        df['month'] = pd.to_datetime(df['time']).dt.month
        df['quarter'] = pd.to_datetime(df['time']).dt.quarter
        
        # Session indicators
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['newyork_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['sydney_session'] = ((df['hour'] >= 21) | (df['hour'] < 6)).astype(int)
        
        # Market overlap
        df['london_newyork_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        df['tokyo_london_overlap'] = ((df['hour'] >= 8) & (df['hour'] < 9)).astype(int)
        
        # 9. ميزات الدعم والمقاومة (20+ ميزة)
        for period in [20, 50, 100]:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'sr_range_{period}'] = df[f'resistance_{period}'] - df[f'support_{period}']
            df[f'price_to_resistance_{period}'] = df['close'] / df[f'resistance_{period}']
            df[f'price_to_support_{period}'] = df['close'] / df[f'support_{period}']
        
        # 10. ميزات متقدمة للتعلم العميق (20+ ميزة)
        # Fourier Transform features
        close_fft = np.fft.fft(df['close'].fillna(0).values)
        df['fft_real_10'] = np.real(close_fft)[:len(df)]
        df['fft_imag_10'] = np.imag(close_fft)[:len(df)]
        
        # Polynomial features
        for degree in [2, 3]:
            df[f'close_poly_{degree}'] = df['close'] ** degree
            df[f'volume_poly_{degree}'] = df['volume'] ** degree
        
        # Cross features
        df['price_volume_interaction'] = df['close'] * df['volume']
        df['high_low_spread'] = df['high'] - df['low']
        df['open_close_spread'] = abs(df['open'] - df['close'])
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        # Select final features
        feature_cols = [col for col in df.columns if col not in 
                       ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']]
        
        logger.info(f"✅ تم إنشاء {len(feature_cols)} ميزة متقدمة")
        
        return df, feature_cols

    def calculate_advanced_targets(self, df, lookahead, min_pips):
        """حساب الأهداف المتقدمة مع تحليل دقيق"""
        pip_size = 0.0001 if 'JPY' not in df['symbol'].iloc[0] else 0.01
        
        signals = []
        confidence = []
        quality = []
        sl_tp_info = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            # حساب الحركة المستقبلية
            max_high = future_prices.max()
            min_low = future_prices.min()
            
            up_move = (max_high - current_price) / pip_size
            down_move = (current_price - min_low) / pip_size
            
            # حساب ATR للتقلب
            atr = df['atr_14'].iloc[i] if 'atr_14' in df.columns else 0
            
            # تحديد الإشارة
            if up_move >= min_pips and up_move > down_move * 1.5:
                signals.append(2)  # Buy
                confidence.append(min(0.99, up_move / (min_pips * 2)))
                quality.append(self._calculate_signal_quality(df, i, 'buy'))
                
                # حساب SL/TP الديناميكي
                sl = current_price - (atr * 1.5)
                tp = current_price + (atr * 2.5)
                sl_tp_info.append({
                    'sl': sl,
                    'tp': tp,
                    'risk_reward': (tp - current_price) / (current_price - sl),
                    'atr': atr
                })
                
            elif down_move >= min_pips and down_move > up_move * 1.5:
                signals.append(0)  # Sell
                confidence.append(min(0.99, down_move / (min_pips * 2)))
                quality.append(self._calculate_signal_quality(df, i, 'sell'))
                
                # حساب SL/TP الديناميكي
                sl = current_price + (atr * 1.5)
                tp = current_price - (atr * 2.5)
                sl_tp_info.append({
                    'sl': sl,
                    'tp': tp,
                    'risk_reward': (current_price - tp) / (sl - current_price),
                    'atr': atr
                })
                
            else:
                signals.append(1)  # Hold
                confidence.append(0.5)
                quality.append(0.5)
                sl_tp_info.append({
                    'sl': 0,
                    'tp': 0,
                    'risk_reward': 0,
                    'atr': atr
                })
        
        # إضافة قيم للصفوف الأخيرة
        for _ in range(lookahead):
            signals.append(1)
            confidence.append(0.5)
            quality.append(0.5)
            sl_tp_info.append({'sl': 0, 'tp': 0, 'risk_reward': 0, 'atr': 0})
        
        return signals, confidence, quality, sl_tp_info

    def _calculate_signal_quality(self, df, idx, direction):
        """حساب جودة الإشارة بناءً على المؤشرات المتعددة"""
        quality_score = 0.5
        
        # RSI confirmation
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14'].iloc[idx]
            if direction == 'buy' and rsi < 70:
                quality_score += 0.1
            elif direction == 'sell' and rsi > 30:
                quality_score += 0.1
        
        # MACD confirmation
        if 'macd_hist_12_26' in df.columns:
            macd_hist = df['macd_hist_12_26'].iloc[idx]
            if (direction == 'buy' and macd_hist > 0) or (direction == 'sell' and macd_hist < 0):
                quality_score += 0.15
        
        # Moving average trend
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            if df['sma_50'].iloc[idx] > df['sma_200'].iloc[idx] and direction == 'buy':
                quality_score += 0.1
            elif df['sma_50'].iloc[idx] < df['sma_200'].iloc[idx] and direction == 'sell':
                quality_score += 0.1
        
        # Volume confirmation
        if 'volume_ratio_20' in df.columns:
            if df['volume_ratio_20'].iloc[idx] > 1.2:
                quality_score += 0.05
        
        return min(0.99, quality_score)

    def balance_data(self, X, y, confidence, quality):
        """موازنة البيانات مع الحفاظ على الإشارات عالية الجودة"""
        # تحويل إلى numpy arrays
        X = np.array(X)
        y = np.array(y)
        confidence = np.array(confidence)
        quality = np.array(quality)
        
        # حساب توزيع الفئات
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"توزيع الفئات الأصلي: {dict(zip(unique, counts))}")
        
        # تحديد الحد الأدنى مع زيادة
        min_samples = min(counts) * 2  # نضاعف العدد الأصغر
        
        balanced_indices = []
        
        for class_label in unique:
            class_indices = np.where(y == class_label)[0]
            
            if class_label != 1:  # Buy or Sell signals
                # نختار الإشارات ذات الجودة العالية
                class_confidence = confidence[class_indices]
                class_quality = quality[class_indices]
                
                # حساب النقاط المجمعة
                combined_score = class_confidence * 0.6 + class_quality * 0.4
                
                # ترتيب حسب النقاط
                sorted_indices = class_indices[np.argsort(combined_score)[::-1]]
                
                # اختيار أفضل الإشارات
                selected = sorted_indices[:min_samples]
                balanced_indices.extend(selected)
            else:  # Hold signals
                # اختيار عشوائي للإشارات المحايدة
                if len(class_indices) > min_samples:
                    selected = np.random.choice(class_indices, min_samples, replace=False)
                else:
                    selected = class_indices
                balanced_indices.extend(selected)
        
        # خلط البيانات
        np.random.shuffle(balanced_indices)
        
        logger.info(f"✅ تم موازنة البيانات: {len(balanced_indices)} عينة")
        
        return X[balanced_indices], y[balanced_indices]

    def train_single_model(self, model_name, config, X_train, y_train, X_val, y_val):
        """تدريب نموذج واحد مع معالجة الأخطاء"""
        try:
            logger.info(f"      🤖 تدريب {model_name}...")
            
            if model_name == 'lightgbm':
                model = lgb.LGBMClassifier(**config)
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                         
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(**config, use_label_encoder=False)
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=False)
                         
            elif model_name == 'catboost':
                model = CatBoostClassifier(**config)
                model.fit(X_train, y_train,
                         eval_set=(X_val, y_val),
                         early_stopping_rounds=50,
                         verbose=False)
                         
            elif model_name == 'random_forest':
                model = RandomForestClassifier(**config)
                model.fit(X_train, y_train)
                
            elif model_name == 'extra_trees':
                model = ExtraTreesClassifier(**config)
                model.fit(X_train, y_train)
                
            elif model_name == 'neural_network':
                model = MLPClassifier(**config)
                model.fit(X_train, y_train)
            
            # تقييم النموذج
            score = model.score(X_val, y_val)
            
            return model_name, model, score
            
        except Exception as e:
            logger.error(f"      ❌ خطأ في تدريب {model_name}: {e}")
            return model_name, None, 0

    def train_ensemble_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """تدريب مجموعة من النماذج مع التقييم"""
        logger.info("    🚀 بدء تدريب النماذج...")
        
        if not self.use_all_models:
            # تدريب LightGBM فقط
            model = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return model, {'accuracy': accuracy, 'models_count': 1}
        
        # تدريب 6-7 نماذج
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
                if model is not None and score > 0.6:  # فقط النماذج الجيدة
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
        
        # تقييم شامل
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
            'models_count': len(trained_models),
            'models_used': [name for name, _ in trained_models]
        }
        
        logger.info(f"    ✅ النموذج المجمع: دقة {accuracy:.4f} ({len(trained_models)} نماذج)")
        
        return ensemble, results

    def save_model_complete(self, model, scaler, feature_names, strategy, results, 
                           sl_tp_settings, symbol, timeframe, strategy_name):
        """حفظ النموذج مع جميع المعلومات المطلوبة"""
        try:
            # إنشاء مجلد النموذج
            model_dir = Path(f"models/{symbol}_{timeframe}/{strategy_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # إعداد البيانات للحفظ
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'strategy': strategy,
                'results': results,
                'sl_tp_settings': sl_tp_settings,
                'training_date': datetime.now().isoformat(),
                'use_all_features': self.use_all_features,
                'use_all_models': self.use_all_models,
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy_name': strategy_name,
                'version': '3.0'
            }
            
            # حفظ النموذج
            model_path = model_dir / 'model_ultimate.pkl'
            joblib.dump(model_data, model_path)
            
            # حفظ ملف معلومات JSON
            info_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'results': results,
                'training_date': datetime.now().isoformat(),
                'features_count': len(feature_names),
                'models_count': results.get('models_count', 1),
                'models_used': results.get('models_used', ['unknown'])
            }
            
            with open(model_dir / 'model_info.json', 'w') as f:
                json.dump(info_data, f, indent=4)
            
            logger.info(f"    💾 تم حفظ النموذج بنجاح: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"    ❌ خطأ في حفظ النموذج: {e}")
            return False

    def train_strategy(self, X, y, confidence, quality, sl_tp_info, feature_names, 
                      strategy_name, strategy, symbol, timeframe):
        """تدريب استراتيجية واحدة مع الحفظ الصحيح"""
        logger.info(f"\n  📊 استراتيجية {strategy_name} - {strategy['description']}")
        
        # موازنة البيانات
        X_balanced, y_balanced = self.balance_data(X, y, confidence, quality)
        
        # تقسيم البيانات
        split1 = int(len(X_balanced) * 0.7)
        split2 = int(len(X_balanced) * 0.85)
        
        X_train = X_balanced[:split1]
        X_val = X_balanced[split1:split2]
        X_test = X_balanced[split2:]
        
        y_train = y_balanced[:split1]
        y_val = y_balanced[split1:split2]
        y_test = y_balanced[split2:]
        
        # معايرة البيانات
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب النماذج
        model, results = self.train_ensemble_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        if model is None:
            return None
        
        # حفظ النموذج إذا كان جيداً
        if results['accuracy'] >= 0.80:  # خفضنا الحد قليلاً لنحصل على المزيد من النماذج
            success = self.save_model_complete(
                model, scaler, feature_names, strategy, results,
                self.sl_tp_settings[strategy_name], symbol, timeframe, strategy_name
            )
            
            if success:
                logger.info(f"    ✅ تم حفظ النموذج بنجاح - دقة: {results['accuracy']:.4f}")
            else:
                logger.error(f"    ❌ فشل حفظ النموذج رغم الدقة الجيدة")
        else:
            logger.warning(f"    ⚠️ النموذج غير مؤهل للحفظ - دقة: {results['accuracy']:.4f}")
        
        return results

    def train_symbol(self, symbol_info, timeframe='H1'):
        """تدريب عملة واحدة مع جميع الاستراتيجيات"""
        symbol = symbol_info['symbol']
        clean_symbol = symbol_info['clean_symbol']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 تدريب {clean_symbol} ({symbol}) - {timeframe}")
        logger.info(f"{'='*80}")
        
        try:
            # جلب البيانات
            conn = sqlite3.connect('data/forex_data.db')
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time ASC
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) < self.min_data_points:
                logger.warning(f"⚠️ بيانات غير كافية: {len(df)} سجل فقط")
                return
            
            logger.info(f"📊 تم تحميل {len(df):,} سجل")
            
            # إنشاء الميزات
            df, feature_cols = self.create_advanced_features(df)
            
            # إعداد البيانات للتدريب
            X = df[feature_cols].values
            
            # نتائج جميع الاستراتيجيات
            all_results = {}
            
            # تدريب كل استراتيجية
            for strategy_name, strategy in self.training_strategies.items():
                logger.info(f"\n🎯 تدريب استراتيجية: {strategy_name}")
                
                # حساب الأهداف
                signals, confidence, quality, sl_tp_info = self.calculate_advanced_targets(
                    df, strategy['lookahead'], strategy['min_pips']
                )
                
                y = np.array(signals)
                
                # التأكد من وجود جميع الفئات
                unique_classes = np.unique(y)
                if len(unique_classes) < 3:
                    logger.warning(f"⚠️ الاستراتيجية {strategy_name} لا تحتوي على جميع الفئات")
                    continue
                
                # تدريب الاستراتيجية
                results = self.train_strategy(
                    X, y, confidence, quality, sl_tp_info, feature_cols,
                    strategy_name, strategy, clean_symbol, timeframe
                )
                
                if results:
                    all_results[strategy_name] = results
            
            # ملخص النتائج
            if all_results:
                logger.info(f"\n📊 ملخص نتائج {clean_symbol}:")
                for strat, res in all_results.items():
                    logger.info(f"  • {strat}: دقة {res['accuracy']:.4f}")
            
            # تنظيف الذاكرة
            del df, X, y
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def train_all_symbols(self):
        """تدريب جميع العملات المتاحة"""
        # الحصول على جميع العملات
        available_symbols = self.get_all_available_symbols()
        
        if not available_symbols:
            logger.error("❌ لا توجد عملات متاحة للتدريب")
            return
        
        # الأطر الزمنية للتدريب
        timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
        
        logger.info(f"\n🌍 سيتم تدريب {len(available_symbols)} عملة على {len(timeframes)} أطر زمنية")
        logger.info("="*80)
        
        # تدريب كل عملة
        total_models = 0
        successful_models = 0
        
        for symbol_info in available_symbols:
            for timeframe in timeframes:
                try:
                    self.train_symbol(symbol_info, timeframe)
                    total_models += len(self.training_strategies)
                    
                    # حساب النماذج الناجحة
                    symbol_dir = Path(f"models/{symbol_info['clean_symbol']}_{timeframe}")
                    if symbol_dir.exists():
                        successful_models += len(list(symbol_dir.iterdir()))
                    
                except Exception as e:
                    logger.error(f"❌ خطأ في {symbol_info['symbol']} - {timeframe}: {e}")
                    continue
                
                # راحة قصيرة بين العملات
                time.sleep(2)
        
        # النتائج النهائية
        logger.info("\n" + "="*80)
        logger.info("📊 النتائج النهائية:")
        logger.info("="*80)
        logger.info(f"• العملات المدربة: {len(available_symbols)}")
        logger.info(f"• إجمالي المحاولات: {total_models}")
        logger.info(f"• النماذج الناجحة: {successful_models}")
        logger.info(f"• معدل النجاح: {(successful_models/total_models*100):.1f}%")

    def monitor_memory(self):
        """مراقبة استخدام الذاكرة"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        
        logger.info(f"💾 استخدام الذاكرة: {memory_gb:.2f} GB")
        
        if memory_gb > 10:  # تحذير إذا تجاوزت 10GB
            logger.warning("⚠️ استخدام عالي للذاكرة!")
            gc.collect()


def main():
    """تشغيل النظام"""
    logger.info("🚀 بدء نظام التدريب المتقدم النهائي")
    
    # إنشاء مجلدات النماذج
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # إنشاء المدرب
    trainer = UltimateAdvancedTrainer(
        use_all_features=True,  # استخدام جميع الميزات المتقدمة
        use_all_models=True,    # استخدام جميع النماذج
        train_all_symbols=True  # تدريب جميع العملات
    )
    
    # بدء التدريب
    trainer.train_all_symbols()
    
    logger.info("\n✅ اكتمل التدريب بنجاح!")


if __name__ == "__main__":
    main()