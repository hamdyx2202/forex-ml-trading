#!/usr/bin/env python3
"""
Advanced Complete Training System with All Models - نظام التدريب الكامل بجميع النماذج
يستخدم 5 نماذج ذكاء اصطناعي مختلفة مع معالجة آمنة
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
logger.add("logs/advanced_training_all_models.log", rotation="1 day", retention="30 days")

class AdvancedCompleteTrainerAllModels:
    """نظام التدريب المتقدم باستخدام جميع نماذج الذكاء الاصطناعي"""
    
    def __init__(self, use_all_models=True):
        self.min_data_points = 10000
        self.test_size = 0.2
        self.validation_split = 0.15
        self.random_state = 42
        self.use_all_models = use_all_models
        
        # عدد العمليات المتوازية
        self.max_workers = min(cpu_count() - 1, 4)
        logger.info(f"🚀 استخدام {self.max_workers} عمليات متوازية")
        
        # إظهار النماذج المستخدمة
        if self.use_all_models:
            logger.info("🤖 استخدام 5 نماذج ذكاء اصطناعي:")
            logger.info("  1️⃣ LightGBM")
            logger.info("  2️⃣ XGBoost") 
            logger.info("  3️⃣ CatBoost")
            logger.info("  4️⃣ Extra Trees")
            logger.info("  5️⃣ Neural Network")
        else:
            logger.info("🤖 استخدام LightGBM فقط (الوضع السريع)")
        
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
        
        # إعدادات النماذج
        self.model_configs = {
            'lightgbm': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbosity': -1,
                'n_jobs': 1
            },
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 10,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 0.1,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'n_jobs': 1
            },
            'catboost': {
                'loss_function': 'MultiClass',
                'classes_count': 3,
                'iterations': 1000,
                'depth': 8,
                'learning_rate': 0.01,
                'l2_leaf_reg': 5,
                'verbose': False,
                'thread_count': 1
            },
            'extra_trees': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'n_jobs': 1
            },
            'neural_network': {
                'hidden_layer_sizes': (150, 100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }
    
    def monitor_memory(self):
        """مراقبة الذاكرة"""
        memory = psutil.virtual_memory()
        logger.info(f"💾 الذاكرة: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    
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
            
            logger.info(f"✅ قاعدة البيانات جاهزة: {count:,} سجل")
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
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ: {e}")
            return None
    
    def create_features(self, df):
        """إنشاء ميزات متقدمة"""
        logger.info("🔬 إنشاء الميزات...")
        features = pd.DataFrame(index=df.index)
        
        # 1. ميزات الأسعار
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # 2. المتوسطات المتحركة
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            features[f'sma_{period}'] = (df['close'] - sma) / sma
            features[f'ema_{period}'] = (df['close'] - ema) / ema
            features[f'sma_slope_{period}'] = sma.pct_change(5)
        
        # 3. مؤشرات فنية
        try:
            # RSI
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            
            # MACD
            macd, signal, hist = talib.MACD(df['close'].values)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values)
            features['bb_upper'] = (upper - df['close']) / df['close']
            features['bb_lower'] = (df['close'] - lower) / df['close']
            
            # ATR
            features['atr_14'] = talib.ATR(df['high'].values, df['low'].values, 
                                          df['close'].values, timeperiod=14) / df['close']
            
            # ADX
            features['adx_14'] = talib.ADX(df['high'].values, df['low'].values,
                                          df['close'].values, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
        except Exception as e:
            logger.warning(f"⚠️ بعض المؤشرات فشلت: {e}")
        
        # 4. ميزات الحجم
        features['volume_sma'] = df['volume'] / df['volume'].rolling(20).mean()
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # 5. ميزات الوقت
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        
        # 6. التقلب
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std()
        
        # تنظيف
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة")
        return features
    
    def create_targets(self, df, strategy):
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
                targets.append(2)  # Buy
                confidences.append(min(max_up / (min_pips * 3), 1.0))
            elif max_down >= min_pips * 2:
                targets.append(0)  # Sell
                confidences.append(min(max_down / (min_pips * 3), 1.0))
            else:
                targets.append(1)  # Hold
                confidences.append(0.5)
        
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        
        return np.array(targets), np.array(confidences)
    
    def balance_data(self, X, y, confidence):
        """موازنة البيانات"""
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # فلترة العينات عالية الثقة
        high_conf = confidence > 0.7
        X_high = X[high_conf]
        y_high = y[high_conf]
        
        try:
            # موازنة
            over = SMOTE(sampling_strategy=0.8, random_state=42)
            under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
            pipeline = Pipeline([('o', over), ('u', under)])
            
            X_balanced, y_balanced = pipeline.fit_resample(X_high, y_high)
            logger.info(f"✅ تمت الموازنة: {len(X_balanced)} عينة")
            return X_balanced, y_balanced
        except:
            return X_high, y_high
    
    def train_single_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """تدريب نموذج واحد"""
        logger.info(f"  • تدريب {model_name}...")
        
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
            
            # تقييم
            val_score = model.score(X_val, y_val)
            logger.info(f"    ✓ {model_name}: دقة {val_score:.4f}")
            
            return model_name, model, val_score
            
        except Exception as e:
            logger.error(f"    ✗ فشل {model_name}: {e}")
            return model_name, None, 0
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """تدريب جميع النماذج وإنشاء نموذج مجمع"""
        logger.info("🎯 تدريب النماذج...")
        
        if not self.use_all_models:
            # استخدام LightGBM فقط
            model = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return model, {'accuracy': accuracy}
        
        # تدريب جميع النماذج بالتوازي
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
        
        if len(trained_models) < 2:
            logger.warning("⚠️ عدد النماذج الناجحة قليل، استخدام أفضل نموذج")
            if trained_models:
                best_model = trained_models[0][1]
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                return best_model, {'accuracy': accuracy}
            else:
                return None, {'accuracy': 0}
        
        # إنشاء نموذج مجمع
        logger.info("🔗 إنشاء النموذج المجمع...")
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
        
        logger.info(f"✅ النموذج المجمع: دقة {accuracy:.4f} ({len(trained_models)} نماذج)")
        
        return ensemble, results
    
    def train_strategy(self, X, y, confidence, feature_names, strategy_name, strategy, symbol, timeframe):
        """تدريب استراتيجية واحدة"""
        logger.info(f"\n📊 تدريب استراتيجية {strategy_name}...")
        
        # موازنة البيانات
        X_balanced, y_balanced = self.balance_data(X, y, confidence)
        
        # تقسيم البيانات
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
        
        # تدريب النماذج
        model, results = self.train_ensemble_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        if model is None:
            return None
        
        # حفظ النموذج إذا كان جيد
        if results['accuracy'] >= 0.85:
            model_dir = Path(f"models/{symbol}_{timeframe}/{strategy_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'strategy': strategy,
                'results': results,
                'training_date': datetime.now(),
                'use_all_models': self.use_all_models
            }
            
            joblib.dump(model_data, model_dir / 'model_advanced.pkl')
            logger.info(f"💾 تم حفظ النموذج")
        
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
        features = self.create_features(df)
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
                # إنشاء الأهداف
                y, confidence = self.create_targets(df, strategy)
                
                # تدريب
                strategy_results = self.train_strategy(
                    X, y, confidence, feature_names,
                    strategy_name, strategy, symbol, timeframe
                )
                
                if strategy_results:
                    results['strategies'][strategy_name] = strategy_results
                    
                    if strategy_results['accuracy'] > results['best_accuracy']:
                        results['best_accuracy'] = strategy_results['accuracy']
                        results['best_strategy'] = strategy_name
                    
                    logger.info(f"✅ {strategy_name}: دقة {strategy_results['accuracy']:.4f}")
                
                # تنظيف الذاكرة
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ في {strategy_name}: {e}")
        
        # ملخص
        logger.info(f"\n📊 ملخص {symbol} {timeframe}:")
        logger.info(f"🏆 أفضل استراتيجية: {results['best_strategy']}")
        logger.info(f"🎯 أفضل دقة: {results['best_accuracy']:.4f}")
        
        return results
    
    def train_all(self):
        """تدريب جميع العملات"""
        logger.info("\n" + "="*100)
        logger.info("🚀 بدء التدريب الشامل")
        if self.use_all_models:
            logger.info("📊 استخدام جميع نماذج الذكاء الاصطناعي (5 نماذج)")
        else:
            logger.info("📊 الوضع السريع - LightGBM فقط")
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
        
        # تدريب كل عملة
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
                        'strategy': result['best_strategy']
                    }
                    
                    if acc >= 0.90:
                        excellent.append(model_info)
                    elif acc >= 0.85:
                        good.append(model_info)
                    elif acc >= 0.80:
                        acceptable.append(model_info)
                    else:
                        failed.append(f"{symbol} {timeframe}")
                
                # مراقبة الذاكرة
                self.monitor_memory()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ خطأ: {e}")
                failed.append(f"{symbol} {timeframe}")
        
        # التقرير النهائي
        self._print_report(excellent, good, acceptable, failed)
    
    def _print_report(self, excellent, good, acceptable, failed):
        """طباعة التقرير النهائي"""
        logger.info("\n" + "="*100)
        logger.info("📊 التقرير النهائي")
        logger.info("="*100)
        
        total = len(excellent) + len(good) + len(acceptable) + len(failed)
        
        logger.info(f"\n📈 الإحصائيات:")
        logger.info(f"  🌟 ممتاز (90%+): {len(excellent)}")
        logger.info(f"  ✅ جيد (85-90%): {len(good)}")
        logger.info(f"  👍 مقبول (80-85%): {len(acceptable)}")
        logger.info(f"  ❌ فشل (<80%): {len(failed)}")
        
        if total > 0:
            success_rate = (len(excellent) + len(good)) / total * 100
            logger.info(f"\n🎯 معدل النجاح: {success_rate:.1f}%")
        
        # أفضل النماذج
        if excellent:
            logger.info(f"\n🌟 أفضل النماذج:")
            for m in sorted(excellent, key=lambda x: x['accuracy'], reverse=True)[:5]:
                logger.info(f"  • {m['symbol']} {m['timeframe']}: {m['accuracy']:.4f}")
        
        # حفظ التقرير
        report = {
            'date': datetime.now().isoformat(),
            'use_all_models': self.use_all_models,
            'total': total,
            'excellent': excellent,
            'good': good,
            'acceptable': acceptable,
            'failed': failed
        }
        
        report_path = Path("models/training_report_all_models.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n💾 تم حفظ التقرير: {report_path}")

def main():
    """الدالة الرئيسية"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Training with All Models')
    parser.add_argument('--quick', action='store_true', help='Quick mode (LightGBM only)')
    parser.add_argument('--symbol', type=str, help='Train specific symbol only')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    args = parser.parse_args()
    
    # تحديد وضع التدريب
    use_all_models = not args.quick
    
    trainer = AdvancedCompleteTrainerAllModels(use_all_models=use_all_models)
    
    if args.symbol:
        # تدريب عملة واحدة
        trainer.train_symbol(args.symbol, args.timeframe)
    else:
        # تدريب شامل
        trainer.train_all()

if __name__ == "__main__":
    main()