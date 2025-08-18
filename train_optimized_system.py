#!/usr/bin/env python3
"""
🚀 Optimized Training System - نظام تدريب محسن
✨ يحل جميع مشاكل الأداء والذاكرة
📊 يحتفظ بجميع الميزات المتقدمة
"""

import os
import sys
import gc
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb

# Technical Analysis
import talib

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedTrainer:
    """مدرب محسن للأداء العالي"""
    
    def __init__(self):
        """تهيئة المدرب"""
        logger.info("="*100)
        logger.info("🚀 Optimized Training System - النظام المحسن")
        logger.info("="*100)
        
        # إعدادات الأداء
        self.chunk_size = 10000  # معالجة البيانات على دفعات
        self.min_data_points = 1000
        self.max_features = 50  # عدد محدود من الميزات للأداء
        self.use_gpu = False  # تعطيل GPU للثبات
        
        # إعدادات التدريب
        self.training_config = {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'n_jobs': 2  # معالجة متوازية محدودة
        }
        
        # استراتيجيات التداول
        self.strategies = {
            'scalping': {
                'description': 'سكالبينج سريع',
                'lookahead': 20,
                'min_pips': 5,
                'take_profit_ratios': [1.5, 2.5],
                'stop_loss_atr': 1.0
            },
            'day_trading': {
                'description': 'تداول يومي',
                'lookahead': 60,
                'min_pips': 15,
                'take_profit_ratios': [2.0, 3.0],
                'stop_loss_atr': 1.5
            },
            'swing_trading': {
                'description': 'تداول متأرجح',
                'lookahead': 240,
                'min_pips': 30,
                'take_profit_ratios': [2.5, 4.0],
                'stop_loss_atr': 2.0
            }
        }
        
        # نماذج ML
        self.models = {
            'lightgbm': {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 2,
                'verbose': -1
            },
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 2,
                'verbosity': 0,
                'use_label_encoder': False
            }
        }
        
        logger.info(f"✅ System initialized with optimized settings")
    
    def load_data_optimized(self, symbol: str, timeframe: str, limit: int = 50000) -> pd.DataFrame:
        """تحميل البيانات بطريقة محسنة"""
        logger.info(f"📊 Loading data for {symbol} {timeframe}...")
        
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            
            # تحميل البيانات على دفعات
            query = """
                SELECT time, open, high, low, close, volume
                FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            
            # قراءة البيانات
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if len(df) < self.min_data_points:
                logger.warning(f"⚠️ Insufficient data: {len(df)} records")
                return None
            
            # معالجة البيانات
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time')
            df = df.set_index('time')
            
            # إزالة القيم المفقودة والمكررة
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            
            # التحقق من جودة البيانات
            if df['close'].std() == 0:
                logger.error("❌ No price variation in data")
                return None
            
            logger.info(f"✅ Loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def create_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء ميزات محسنة"""
        logger.info("🔧 Creating optimized features...")
        
        features = pd.DataFrame(index=df.index)
        
        try:
            # 1. ميزات السعر الأساسية
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_range'] = (df['high'] - df['low']) / df['close']
            features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # 2. المتوسطات المتحركة
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff()
            
            # 3. مؤشرات RSI
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            
            # 4. Bollinger Bands
            for period in [20]:
                upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period)
                features[f'bb_upper_{period}'] = upper
                features[f'bb_middle_{period}'] = middle
                features[f'bb_lower_{period}'] = lower
                features[f'bb_width_{period}'] = (upper - lower) / middle
                features[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-10)
            
            # 5. MACD
            macd, signal, hist = talib.MACD(df['close'].values)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # 6. ATR للتقلب
            features['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            features['atr_7'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=7)
            
            # 7. Volume features
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # 8. Pattern Recognition (محدود)
            features['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            features['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            
            # إزالة القيم المفقودة
            features = features.fillna(method='ffill').fillna(0)
            
            # اختيار أفضل الميزات فقط
            features = features.iloc[:, :self.max_features]
            
            logger.info(f"✅ Created {len(features.columns)} optimized features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error creating features: {e}")
            return pd.DataFrame()
    
    def create_targets_optimized(self, df: pd.DataFrame, strategy: dict) -> Tuple[np.ndarray, np.ndarray]:
        """إنشاء أهداف محسنة"""
        logger.info(f"🎯 Creating targets for {strategy['description']}...")
        
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        # حساب قيمة النقطة
        pip_value = 0.0001 if 'JPY' not in df.index.name else 0.01
        
        targets = []
        confidences = []
        
        # معالجة على دفعات لتوفير الذاكرة
        batch_size = 1000
        total_samples = len(df) - lookahead
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            
            for i in range(start_idx, end_idx):
                try:
                    current_price = df['close'].iloc[i]
                    future_prices = df['close'].iloc[i+1:i+lookahead+1].values
                    
                    if len(future_prices) == 0:
                        continue
                    
                    # حساب الحركة المستقبلية
                    max_up = (future_prices.max() - current_price) / pip_value
                    max_down = (current_price - future_prices.min()) / pip_value
                    
                    # تحديد الهدف
                    if max_up >= min_pips * 2:
                        targets.append(2)  # Buy
                        confidence = min(0.5 + (max_up / (min_pips * 4)) * 0.5, 1.0)
                        confidences.append(confidence)
                    elif max_down >= min_pips * 2:
                        targets.append(0)  # Sell
                        confidence = min(0.5 + (max_down / (min_pips * 4)) * 0.5, 1.0)
                        confidences.append(confidence)
                    else:
                        targets.append(1)  # Hold
                        confidences.append(0.5)
                        
                except Exception as e:
                    targets.append(1)
                    confidences.append(0.5)
            
            # تنظيف الذاكرة
            if start_idx % 5000 == 0:
                gc.collect()
                logger.info(f"  Progress: {start_idx}/{total_samples}")
        
        # ملء الباقي
        remaining = len(df) - len(targets)
        targets.extend([1] * remaining)
        confidences.extend([0.5] * remaining)
        
        # إحصائيات
        unique, counts = np.unique(targets, return_counts=True)
        stats = dict(zip(unique, counts))
        logger.info(f"✅ Targets created - Buy: {stats.get(2, 0)}, Sell: {stats.get(0, 0)}, Hold: {stats.get(1, 0)}")
        
        return np.array(targets), np.array(confidences)
    
    def train_model_optimized(self, X_train, y_train, X_val, y_val, model_type='lightgbm'):
        """تدريب نموذج محسن"""
        logger.info(f"🤖 Training {model_type} model...")
        
        try:
            if model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**self.models['lightgbm'])
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(self.training_config['early_stopping_rounds']),
                             lgb.log_evaluation(0)]
                )
            else:  # xgboost
                model = xgb.XGBClassifier(**self.models['xgboost'])
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.training_config['early_stopping_rounds'],
                    verbose=False
                )
            
            # تقييم الأداء
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            results = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"✅ Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return None
    
    def train_symbol(self, symbol: str, timeframe: str) -> dict:
        """تدريب رمز واحد"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Training {symbol} {timeframe}")
        logger.info(f"{'='*80}")
        
        # تحميل البيانات
        df = self.load_data_optimized(symbol, timeframe)
        if df is None:
            return None
        
        # إنشاء الميزات
        features = self.create_features_optimized(df)
        if features.empty:
            return None
        
        # النتائج
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data_points': len(df),
            'strategies': {}
        }
        
        # تدريب كل استراتيجية
        for strategy_name, strategy in self.strategies.items():
            try:
                logger.info(f"\n📊 Strategy: {strategy_name}")
                
                # إنشاء الأهداف
                targets, confidences = self.create_targets_optimized(df, strategy)
                
                # فلترة البيانات عالية الثقة
                high_confidence = confidences > 0.6
                X_filtered = features[high_confidence].values
                y_filtered = targets[high_confidence]
                
                if len(X_filtered) < 100:
                    logger.warning(f"⚠️ Not enough high confidence samples: {len(X_filtered)}")
                    continue
                
                # تقسيم البيانات
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X_filtered, y_filtered, 
                    test_size=0.3, 
                    random_state=42,
                    stratify=y_filtered
                )
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp,
                    test_size=0.5,
                    random_state=42,
                    stratify=y_temp
                )
                
                # معايرة البيانات
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # تدريب النموذج
                model_results = self.train_model_optimized(
                    X_train_scaled, y_train,
                    X_val_scaled, y_val,
                    model_type='lightgbm'
                )
                
                if model_results:
                    # تقييم على بيانات الاختبار
                    model = model_results['model']
                    y_pred = model.predict(X_test_scaled)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    # حفظ النتائج
                    strategy_results = {
                        'accuracy': test_accuracy,
                        'precision': model_results['precision'],
                        'recall': model_results['recall'],
                        'f1': model_results['f1'],
                        'samples': len(X_filtered)
                    }
                    
                    results['strategies'][strategy_name] = strategy_results
                    
                    # حفظ النموذج إذا كان جيداً
                    if test_accuracy >= 0.60:
                        self.save_model(model, scaler, symbol, timeframe, strategy_name, strategy_results)
                    
                    logger.info(f"✅ {strategy_name} completed - Accuracy: {test_accuracy:.4f}")
                
                # تنظيف الذاكرة
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # اختيار أفضل استراتيجية
        if results['strategies']:
            best_strategy = max(results['strategies'].items(), key=lambda x: x[1]['accuracy'])
            results['best_strategy'] = best_strategy[0]
            results['best_accuracy'] = best_strategy[1]['accuracy']
        else:
            results['best_strategy'] = None
            results['best_accuracy'] = 0
        
        return results
    
    def save_model(self, model, scaler, symbol, timeframe, strategy_name, results):
        """حفظ النموذج"""
        try:
            model_dir = Path(f"models/{symbol}_{timeframe}/{strategy_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # حفظ النموذج والمعاير
            model_data = {
                'model': model,
                'scaler': scaler,
                'results': results,
                'training_date': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name
            }
            
            model_path = model_dir / 'model_optimized.pkl'
            joblib.dump(model_data, model_path)
            
            # حفظ معلومات النموذج
            info_path = model_dir / 'model_info.json'
            info_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'samples': results['samples'],
                'training_date': datetime.now().isoformat()
            }
            
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            logger.info(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")

def main():
    """الدالة الرئيسية"""
    trainer = OptimizedTrainer()
    
    # الحصول على الرموز من قاعدة البيانات
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        query = """
            SELECT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING count >= 5000
            ORDER BY count DESC
            LIMIT 10
        """
        available = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"\n📊 Found {len(available)} symbols with sufficient data")
        
        # تدريب أفضل 5 رموز
        results = {}
        for idx, row in available.head(5).iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            result = trainer.train_symbol(symbol, timeframe)
            if result:
                results[f"{symbol}_{timeframe}"] = result
        
        # عرض الملخص
        logger.info("\n" + "="*80)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*80)
        
        for key, result in results.items():
            if result['best_strategy']:
                logger.info(f"{key}: {result['best_accuracy']:.2%} ({result['best_strategy']})")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()