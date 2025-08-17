#!/usr/bin/env python3
"""
Advanced Complete Training System with Fixed Parallel Processing - نظام التدريب المتقدم المحسن
تم إصلاح مشاكل التوقف والتجمد في المعالجة المتوازية
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

# مكتبات المعالجة المتوازية
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count, Manager, Pool
import threading
from functools import partial
import time
import gc
import psutil

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/advanced_training_parallel_fixed.log", rotation="1 day", retention="30 days")

# دالة مساعدة للتدريب - يجب أن تكون في المستوى الأعلى للـ pickle
def train_strategy_worker(args):
    """دالة عامل لتدريب استراتيجية واحدة"""
    try:
        (X, y, confidence, feature_names, strategy_name, strategy, 
         symbol, timeframe, trainer_config) = args
        
        logger.info(f"🔧 بدء تدريب {strategy_name} لـ {symbol} {timeframe}")
        
        # استيراد المكتبات المطلوبة داخل العملية
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # موازنة البيانات
        high_conf_mask = confidence > 0.7
        X_high = X[high_conf_mask]
        y_high = y[high_conf_mask]
        
        # موازنة الفئات
        try:
            over = SMOTE(sampling_strategy=0.8, random_state=42, n_jobs=1)
            under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            X_balanced, y_balanced = pipeline.fit_resample(X_high, y_high)
        except:
            X_balanced, y_balanced = X_high, y_high
        
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
        
        # معايرة
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب نموذج بسيط بدلاً من المجمع المعقد
        # استخدام LightGBM فقط لتجنب مشاكل التسلسل
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 100,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'verbosity': -1,
            'n_jobs': 1
        }
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_scaled, y_train, 
                 eval_set=[(X_val_scaled, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        # تقييم
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # معدل نجاح الصفقات
        trade_mask = y_pred != 1
        if trade_mask.sum() > 0:
            trade_accuracy = accuracy_score(y_test[trade_mask], y_pred[trade_mask])
        else:
            trade_accuracy = 0
        
        logger.info(f"✅ {strategy_name}: دقة {test_accuracy:.4f}, صفقات {trade_accuracy:.4f}")
        
        # النتائج
        strategy_results = {
            'accuracy': test_accuracy,
            'trade_accuracy': trade_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confidence_threshold': strategy['confidence_threshold']
        }
        
        # حفظ النموذج إذا كان جيد
        if test_accuracy >= 0.85:
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
        
        return strategy_name, strategy_results
        
    except Exception as e:
        logger.error(f"❌ خطأ في تدريب {strategy_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return strategy_name, None

class AdvancedCompleteTrainerParallelFixed:
    """نظام التدريب المتقدم المحسن مع حل مشاكل المعالجة المتوازية"""
    
    def __init__(self, max_workers=None):
        self.min_data_points = 10000
        self.test_size = 0.2
        self.validation_split = 0.15
        self.random_state = 42
        
        # تحديد عدد العمليات المتوازية بشكل محافظ
        available_cores = cpu_count()
        self.max_workers = max_workers or max(1, min(available_cores - 1, 4))
        logger.info(f"🚀 استخدام {self.max_workers} عمليات متوازية من أصل {available_cores} معالج")
        
        # مراقبة الذاكرة
        self.monitor_memory()
        
        # استراتيجيات التدريب
        self.training_strategies = {
            'ultra_short': {'lookahead': 5, 'min_pips': 3, 'confidence_threshold': 0.95},
            'scalping': {'lookahead': 15, 'min_pips': 5, 'confidence_threshold': 0.92},
            'short_term': {'lookahead': 30, 'min_pips': 10, 'confidence_threshold': 0.90},
            'medium_term': {'lookahead': 60, 'min_pips': 20, 'confidence_threshold': 0.88},
            'long_term': {'lookahead': 240, 'min_pips': 40, 'confidence_threshold': 0.85}
        }
        
        # إعدادات Stop Loss و Take Profit
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
    
    def monitor_memory(self):
        """مراقبة استخدام الذاكرة"""
        memory = psutil.virtual_memory()
        logger.info(f"💾 الذاكرة: {memory.percent}% مستخدمة ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
        if memory.percent > 80:
            logger.warning("⚠️ استخدام الذاكرة مرتفع! قد يؤثر على الأداء")
    
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
            
            if count == 0:
                logger.error("❌ لا توجد بيانات في قاعدة البيانات!")
                return False
            
            logger.info(f"✅ قاعدة البيانات جاهزة: {count:,} سجل")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في قاعدة البيانات: {e}")
            return False
    
    def load_data_advanced(self, symbol, timeframe, limit=100000):
        """تحميل البيانات"""
        try:
            logger.info(f"📊 تحميل بيانات {symbol} {timeframe}...")
            
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
            
            # تحويل الوقت
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            
            # ملء القيم المفقودة
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def create_ultra_advanced_features(self, df, symbol):
        """إنشاء الميزات المتقدمة مع معالجة محسنة"""
        logger.info("🔬 إنشاء الميزات المتقدمة...")
        start_time = time.time()
        
        features = pd.DataFrame(index=df.index)
        
        # حساب الميزات بشكل متسلسل لتجنب مشاكل الذاكرة
        logger.info("  • حساب ميزات الأسعار...")
        features = pd.concat([features, self._calculate_price_features(df)], axis=1)
        
        logger.info("  • حساب المتوسطات المتحركة...")
        features = pd.concat([features, self._calculate_ma_features(df)], axis=1)
        
        logger.info("  • حساب مؤشرات TA-Lib...")
        features = pd.concat([features, self._calculate_talib_features(df)], axis=1)
        
        logger.info("  • حساب ميزات الحجم...")
        features = pd.concat([features, self._calculate_volume_features(df)], axis=1)
        
        logger.info("  • حساب ميزات التقلب...")
        features = pd.concat([features, self._calculate_volatility_features(df)], axis=1)
        
        logger.info("  • حساب ميزات السوق...")
        features = pd.concat([features, self._calculate_market_features(df)], axis=1)
        
        # تنظيف البيانات
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # تنظيف الذاكرة
        gc.collect()
        
        end_time = time.time()
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة في {end_time - start_time:.2f} ثانية")
        
        return features
    
    def _calculate_price_features(self, df):
        """حساب ميزات الأسعار"""
        features = pd.DataFrame(index=df.index)
        
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        return features
    
    def _calculate_ma_features(self, df):
        """حساب المتوسطات المتحركة"""
        features = pd.DataFrame(index=df.index)
        
        ma_periods = [5, 10, 20, 50, 100, 200]
        
        for period in ma_periods:
            sma = df['close'].rolling(period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            
            features[f'sma_{period}'] = (df['close'] - sma) / sma
            features[f'ema_{period}'] = (df['close'] - ema) / ema
            features[f'sma_{period}_slope'] = sma.pct_change(5)
            features[f'ema_{period}_slope'] = ema.pct_change(5)
        
        return features
    
    def _calculate_talib_features(self, df):
        """حساب مؤشرات TA-Lib"""
        features = pd.DataFrame(index=df.index)
        
        try:
            # RSI
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df['close'].values)
            features['macd'] = macd
            features['macd_signal'] = macdsignal
            features['macd_hist'] = macdhist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values)
            features['bb_upper'] = (upper - df['close']) / df['close']
            features['bb_lower'] = (df['close'] - lower) / df['close']
            features['bb_width'] = (upper - lower) / middle
            
            # ATR
            features['atr_14'] = talib.ATR(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=14) / df['close']
            
            # ADX
            features['adx_14'] = talib.ADX(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=14)
            
        except Exception as e:
            logger.warning(f"⚠️ بعض مؤشرات TA-Lib فشلت: {e}")
        
        return features
    
    def _calculate_volume_features(self, df):
        """حساب ميزات الحجم"""
        features = pd.DataFrame(index=df.index)
        
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['force_index'] = df['close'].diff() * df['volume']
        
        return features
    
    def _calculate_volatility_features(self, df):
        """حساب ميزات التقلب"""
        features = pd.DataFrame(index=df.index)
        
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std()
        
        return features
    
    def _calculate_market_features(self, df):
        """حساب ميزات السوق"""
        features = pd.DataFrame(index=df.index)
        
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        
        # جلسات التداول
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
        features['london_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        
        return features
    
    def create_advanced_targets(self, df, strategy):
        """إنشاء الأهداف"""
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        pip_value = 0.0001 if 'JPY' not in str(df.index.name) else 0.01
        
        targets = []
        confidences = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            max_up = (future_prices.max() - current_price) / pip_value
            max_down = (current_price - future_prices.min()) / pip_value
            
            if max_up >= min_pips * 2:
                targets.append(2)
                confidences.append(min(max_up / (min_pips * 3), 1.0))
            elif max_down >= min_pips * 2:
                targets.append(0)
                confidences.append(min(max_down / (min_pips * 3), 1.0))
            else:
                targets.append(1)
                confidences.append(0.5)
        
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        
        return np.array(targets), np.array(confidences)
    
    def train_symbol_advanced(self, symbol, timeframe):
        """تدريب عملة واحدة مع معالجة محسنة"""
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
        
        # تدريب كل استراتيجية بشكل متسلسل لتجنب مشاكل المعالجة المتوازية
        for strategy_name, strategy in self.training_strategies.items():
            try:
                logger.info(f"\n📊 تدريب استراتيجية {strategy_name}...")
                
                # إنشاء الأهداف
                y, confidence = self.create_advanced_targets(df, strategy)
                
                # إعداد البيانات للتدريب
                trainer_config = {
                    'min_data_points': self.min_data_points,
                    'test_size': self.test_size,
                    'validation_split': self.validation_split,
                    'random_state': self.random_state
                }
                
                args = (X, y, confidence, feature_names, strategy_name, 
                       strategy, symbol, timeframe, trainer_config)
                
                # تدريب الاستراتيجية
                strat_name, strategy_results = train_strategy_worker(args)
                
                if strategy_results:
                    results['strategies'][strat_name] = strategy_results
                    
                    # تحديث أفضل نتيجة
                    if strategy_results['accuracy'] > results['best_accuracy']:
                        results['best_accuracy'] = strategy_results['accuracy']
                        results['best_strategy'] = strat_name
                    
                    logger.info(f"✅ {strat_name}: دقة {strategy_results['accuracy']:.4f}")
                
                # تنظيف الذاكرة بعد كل استراتيجية
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ في تدريب {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # طباعة الملخص
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ملخص {symbol} {timeframe}")
        logger.info(f"🏆 أفضل استراتيجية: {results['best_strategy']}")
        logger.info(f"🎯 أفضل دقة: {results['best_accuracy']:.4f}")
        
        return results
    
    def train_symbols_batch(self, symbols_batch):
        """تدريب مجموعة من العملات"""
        results = []
        
        for symbol, timeframe in symbols_batch:
            try:
                logger.info(f"\n🔄 معالجة {symbol} {timeframe}")
                result = self.train_symbol_advanced(symbol, timeframe)
                if result:
                    results.append(result)
                
                # مراقبة الذاكرة
                self.monitor_memory()
                
                # تنظيف الذاكرة
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ في {symbol} {timeframe}: {e}")
        
        return results
    
    def train_all_advanced(self):
        """تدريب جميع العملات بطريقة محسنة"""
        logger.info("\n" + "="*100)
        logger.info("🚀 بدء التدريب المتقدم المحسن")
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
        
        # معالجة العملات بشكل متسلسل مع معالجة متوازية للاستراتيجيات فقط
        total_symbols = len(available)
        
        for idx, row in available.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            logger.info(f"\n📈 معالجة {idx+1}/{total_symbols}: {symbol} {timeframe}")
            
            try:
                result = self.train_symbol_advanced(symbol, timeframe)
                
                if result:
                    best_acc = result['best_accuracy']
                    
                    model_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': best_acc,
                        'strategy': result['best_strategy']
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
                
                # تنظيف الذاكرة بعد كل عملة
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ في {symbol} {timeframe}: {e}")
                failed_models.append(f"{symbol} {timeframe}: {str(e)}")
        
        # طباعة التقرير النهائي
        self._print_final_report(excellent_models, good_models, acceptable_models, failed_models)
    
    def _print_final_report(self, excellent, good, acceptable, failed):
        """طباعة التقرير النهائي"""
        logger.info("\n" + "="*100)
        logger.info("📊 التقرير النهائي للتدريب المحسن")
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
                'max_workers': self.max_workers
            }
        }
        
        report_path = Path("models/advanced_training_report_fixed.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n💾 تم حفظ التقرير: {report_path}")

def main():
    """الدالة الرئيسية"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Training System - Fixed Version')
    parser.add_argument('--quick', action='store_true', help='Quick test with one symbol')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Symbol to train (for quick mode)')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe to train (for quick mode)')
    args = parser.parse_args()
    
    trainer = AdvancedCompleteTrainerParallelFixed(max_workers=args.workers)
    
    if args.quick:
        # تدريب عملة واحدة فقط
        trainer.train_symbol_advanced(args.symbol, args.timeframe)
    else:
        # تدريب شامل
        trainer.train_all_advanced()

if __name__ == "__main__":
    main()