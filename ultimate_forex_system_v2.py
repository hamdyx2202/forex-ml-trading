#!/usr/bin/env python3
"""
🚀 Ultimate Forex ML System V2 - Production Ready
✨ نظام متكامل محسن للإنتاج
📊 يعمل مع جميع الرموز المتاحة في قاعدة البيانات
"""

import os
import sys
import gc
import json
import sqlite3
import asyncio
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

# Technical Analysis
try:
    import talib
except ImportError:
    print("Warning: TA-Lib not installed. Using basic indicators only.")
    talib = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('forex_system_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexSystemV2:
    """النظام المحسن V2"""
    
    def __init__(self):
        """تهيئة النظام"""
        logger.info("="*100)
        logger.info("🚀 Ultimate Forex ML System V2")
        logger.info("="*100)
        
        self.db_path = "data/forex_ml.db"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)
        
        # المكونات
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.available_symbols = {}
        
        # الإعدادات
        self.config = {
            'min_data_points': 1000,
            'max_data_points': 30000,
            'batch_size': 2000,
            'min_accuracy': 0.55,
            'min_confidence': 0.65,
            'strategies': {
                'M5': {'lookahead': 20, 'min_pips': 5, 'sl_atr': 1.0, 'tp_ratios': [1.5, 2.5, 3.5]},
                'M15': {'lookahead': 20, 'min_pips': 10, 'sl_atr': 1.2, 'tp_ratios': [1.5, 2.5, 3.5]},
                'M30': {'lookahead': 20, 'min_pips': 15, 'sl_atr': 1.5, 'tp_ratios': [2.0, 3.0, 4.0]},
                'H1': {'lookahead': 24, 'min_pips': 20, 'sl_atr': 1.5, 'tp_ratios': [2.0, 3.0, 4.0]},
                'H4': {'lookahead': 24, 'min_pips': 40, 'sl_atr': 2.0, 'tp_ratios': [2.5, 4.0, 6.0]}
            }
        }
    
    def check_database(self) -> bool:
        """فحص قاعدة البيانات"""
        if not os.path.exists(self.db_path):
            logger.error(f"❌ Database not found: {self.db_path}")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            logger.info(f"✅ Database OK - {count:,} records")
            
            # الحصول على الرموز المتاحة
            query = """
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn, params=(self.config['min_data_points'],))
            
            # تنظيم الرموز حسب الإطار الزمني
            for _, row in df.iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                count = row['count']
                
                if symbol not in self.available_symbols:
                    self.available_symbols[symbol] = {}
                self.available_symbols[symbol][timeframe] = count
            
            conn.close()
            
            logger.info(f"📊 Found {len(self.available_symbols)} symbols with sufficient data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """تحميل البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT time, open, high, low, close, volume
                FROM price_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(
                symbol, timeframe, self.config['max_data_points']
            ))
            conn.close()
            
            if len(df) < self.config['min_data_points']:
                logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)}")
                return None
            
            # معالجة البيانات
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time').set_index('time')
            df = df.dropna()
            
            # التحقق من جودة البيانات
            if df['close'].std() == 0:
                logger.warning(f"No price variation in {symbol} {timeframe}")
                return None
            
            logger.info(f"✅ Loaded {len(df)} records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {symbol} {timeframe}: {e}")
            return None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء الميزات"""
        features = pd.DataFrame(index=df.index)
        
        try:
            # الأسعار
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_range'] = (df['high'] - df['low']) / df['close']
            features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # المتوسطات
            for period in [5, 10, 20, 50]:
                sma = df['close'].rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'price_sma_{period}'] = df['close'] / sma
                features[f'sma_{period}_slope'] = sma.diff()
            
            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            features['bb_upper'] = sma20 + (2 * std20)
            features['bb_lower'] = sma20 - (2 * std20)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma20
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
            
            # RSI محسن
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features['atr'] = true_range.rolling(14).mean()
            features['atr_ratio'] = features['atr'] / df['close']
            
            # Volume
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # الوقت
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            
            # Patterns بسيطة
            features['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                                     (df['high'].shift(1) > df['high'].shift(2))).astype(int)
            features['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                                   (df['low'].shift(1) < df['low'].shift(2))).astype(int)
            
            # تنظيف
            features = features.fillna(method='ffill').fillna(0)
            
            # إزالة القيم اللانهائية
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def create_targets(self, df: pd.DataFrame, features: pd.DataFrame, 
                      strategy: dict) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """إنشاء الأهداف"""
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        # حساب قيمة النقطة
        symbol_name = str(df.index.name) if hasattr(df.index, 'name') else ''
        if 'JPY' in symbol_name:
            pip_value = 0.01
        elif 'XAU' in symbol_name:
            pip_value = 0.1
        elif 'BTC' in symbol_name or 'ETH' in symbol_name:
            pip_value = 1.0
        else:
            pip_value = 0.0001
        
        targets = []
        confidences = []
        sl_tp_info = []
        
        total = len(df) - lookahead
        logger.info(f"  Creating targets for {total} samples...")
        
        for i in range(total):
            # عرض التقدم
            if i % 5000 == 0 and i > 0:
                logger.info(f"    Progress: {(i/total)*100:.1f}%")
            
            current_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i+1:i+lookahead+1].max()
            future_low = df['low'].iloc[i+1:i+lookahead+1].min()
            
            # حركة السعر
            max_up = (future_high - current_price) / pip_value
            max_down = (current_price - future_low) / pip_value
            
            # ATR للـ SL/TP
            current_atr = features['atr'].iloc[i] if 'atr' in features.columns else pip_value * 20
            
            # تحديد الاتجاه
            if max_up >= min_pips * 2 and max_up > max_down * 1.5:
                targets.append(2)  # BUY
                confidence = min(0.5 + (max_up / (min_pips * 5)) * 0.5, 0.95)
                confidences.append(confidence)
                
                # SL/TP
                sl = current_price - (current_atr * strategy['sl_atr'])
                tps = [current_price + (current_atr * ratio) for ratio in strategy['tp_ratios']]
                
                sl_tp_info.append({
                    'stop_loss': sl,
                    'take_profits': tps,
                    'atr': current_atr
                })
                
            elif max_down >= min_pips * 2 and max_down > max_up * 1.5:
                targets.append(0)  # SELL
                confidence = min(0.5 + (max_down / (min_pips * 5)) * 0.5, 0.95)
                confidences.append(confidence)
                
                # SL/TP
                sl = current_price + (current_atr * strategy['sl_atr'])
                tps = [current_price - (current_atr * ratio) for ratio in strategy['tp_ratios']]
                
                sl_tp_info.append({
                    'stop_loss': sl,
                    'take_profits': tps,
                    'atr': current_atr
                })
                
            else:
                targets.append(1)  # HOLD
                confidences.append(0.5)
                sl_tp_info.append(None)
        
        # ملء الباقي
        remaining = len(df) - len(targets)
        targets.extend([1] * remaining)
        confidences.extend([0.5] * remaining)
        sl_tp_info.extend([None] * remaining)
        
        # إحصائيات
        unique, counts = np.unique(targets, return_counts=True)
        stats = dict(zip(unique, counts))
        logger.info(f"  Targets: Buy={stats.get(2,0)}, Sell={stats.get(0,0)}, Hold={stats.get(1,0)}")
        
        return np.array(targets), np.array(confidences), sl_tp_info
    
    def train_model(self, symbol: str, timeframe: str) -> Optional[dict]:
        """تدريب نموذج"""
        logger.info(f"\n🎯 Training {symbol} {timeframe}")
        
        # تحميل البيانات
        df = self.load_data(symbol, timeframe)
        if df is None:
            return None
        
        # إنشاء الميزات
        features = self.create_features(df)
        if features.empty:
            return None
        
        # إضافة ATR للـ dataframe
        df['atr'] = features['atr']
        
        # الاستراتيجية المناسبة
        if timeframe not in self.config['strategies']:
            logger.warning(f"No strategy for timeframe {timeframe}")
            return None
            
        strategy = self.config['strategies'][timeframe]
        
        # إنشاء الأهداف
        targets, confidences, sl_tp_info = self.create_targets(df, features, strategy)
        
        # فلترة عالية الثقة
        high_conf = confidences > 0.6
        X = features[high_conf].values
        y = targets[high_conf]
        
        if len(X) < 500:
            logger.warning(f"Not enough high confidence samples: {len(X)}")
            return None
        
        # تقسيم البيانات
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            logger.error(f"Error splitting data: {e}")
            return None
        
        # معايرة
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            verbose=-1,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # تقييم
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Model accuracy: {accuracy:.2%}")
        
        # حفظ النموذج
        if accuracy >= self.config['min_accuracy']:
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_names[model_key] = list(features.columns)
            
            # حفظ على القرص
            model_data = {
                'model': model,
                'scaler': scaler,
                'features': list(features.columns),
                'accuracy': accuracy,
                'symbol': symbol,
                'timeframe': timeframe,
                'strategy': strategy,
                'training_date': datetime.now().isoformat()
            }
            
            model_path = self.models_dir / f"{model_key}.pkl"
            joblib.dump(model_data, model_path)
            logger.info(f"💾 Model saved: {model_path}")
        
        return {'accuracy': accuracy, 'samples': len(X)}
    
    async def train_all_models(self):
        """تدريب جميع النماذج"""
        logger.info("\n📊 Starting training for all available symbols...")
        
        results = {}
        successful = 0
        
        # تدريب كل رمز
        for symbol, timeframes in self.available_symbols.items():
            for timeframe, count in timeframes.items():
                if timeframe in self.config['strategies']:
                    try:
                        result = self.train_model(symbol, timeframe)
                        if result and result['accuracy'] >= self.config['min_accuracy']:
                            successful += 1
                            results[f"{symbol}_{timeframe}"] = result
                    except Exception as e:
                        logger.error(f"Error training {symbol} {timeframe}: {e}")
                    
                    # تنظيف الذاكرة
                    gc.collect()
        
        # ملخص
        logger.info("\n" + "="*80)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total symbols available: {len(self.available_symbols)}")
        logger.info(f"Models trained: {len(results)}")
        logger.info(f"Successful models: {successful}")
        
        if results:
            best = max(results.items(), key=lambda x: x[1]['accuracy'])
            logger.info(f"Best model: {best[0]} - {best[1]['accuracy']:.2%}")
    
    def generate_signal(self, symbol: str, timeframe: str) -> Optional[dict]:
        """توليد إشارة تداول"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            return None
        
        # تحميل آخر البيانات
        df = self.load_data(symbol, timeframe)
        if df is None or len(df) < 100:
            return None
        
        # إنشاء الميزات
        features = self.create_features(df)
        if features.empty:
            return None
        
        # آخر صف
        X = features.iloc[-1:].values
        
        # معايرة وتنبؤ
        try:
            X_scaled = self.scalers[model_key].transform(X)
            prediction = self.models[model_key].predict(X_scaled)[0]
            probabilities = self.models[model_key].predict_proba(X_scaled)[0]
            confidence = probabilities.max()
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None
        
        # توليد الإشارة
        if confidence >= self.config['min_confidence'] and prediction != 1:
            current_price = df['close'].iloc[-1]
            current_atr = features['atr'].iloc[-1]
            strategy = self.config['strategies'][timeframe]
            
            if prediction == 2:  # BUY
                action = 'BUY'
                sl = current_price - (current_atr * strategy['sl_atr'])
                tps = [current_price + (current_atr * ratio) for ratio in strategy['tp_ratios']]
            else:  # SELL
                action = 'SELL'
                sl = current_price + (current_atr * strategy['sl_atr'])
                tps = [current_price - (current_atr * ratio) for ratio in strategy['tp_ratios']]
            
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'action': action,
                'entry_price': float(current_price),
                'stop_loss': float(sl),
                'take_profit_1': float(tps[0]),
                'take_profit_2': float(tps[1]),
                'take_profit_3': float(tps[2]),
                'confidence': float(confidence),
                'atr': float(current_atr),
                'timestamp': datetime.now().isoformat(),
                'magic_number': int(datetime.now().timestamp() % 1000000)
            }
            
            logger.info(f"{'🟢' if action == 'BUY' else '🔴'} {action} Signal: {symbol} @ {current_price:.5f} (Confidence: {confidence:.2%})")
            return signal
        
        return None
    
    async def signal_generation_loop(self):
        """حلقة توليد الإشارات"""
        logger.info("\n🔄 Starting signal generation loop...")
        
        while True:
            try:
                all_signals = []
                
                # فحص جميع النماذج
                for model_key in self.models.keys():
                    symbol, timeframe = model_key.split('_')
                    signal = self.generate_signal(symbol, timeframe)
                    
                    if signal:
                        all_signals.append(signal)
                
                # حفظ الإشارات
                if all_signals:
                    signals_file = self.signals_dir / 'active_signals.json'
                    with open(signals_file, 'w') as f:
                        json.dump(all_signals, f, indent=2)
                    
                    logger.info(f"📤 Saved {len(all_signals)} signals")
                
                # انتظار دقيقة
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(60)
    
    def load_existing_models(self):
        """تحميل النماذج الموجودة"""
        logger.info("📁 Loading existing models...")
        
        model_files = list(self.models_dir.glob("*.pkl"))
        loaded = 0
        
        for model_file in model_files:
            try:
                data = joblib.load(model_file)
                model_key = model_file.stem
                
                self.models[model_key] = data['model']
                self.scalers[model_key] = data['scaler']
                self.feature_names[model_key] = data['features']
                
                loaded += 1
                logger.info(f"✅ Loaded: {model_key} (Accuracy: {data.get('accuracy', 0):.2%})")
                
            except Exception as e:
                logger.error(f"Error loading {model_file}: {e}")
        
        logger.info(f"📊 Loaded {loaded} models successfully")

async def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Ultimate Forex ML System V2')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'trade', 'full', 'check'],
                       help='System mode')
    
    args = parser.parse_args()
    
    # إنشاء النظام
    system = ForexSystemV2()
    
    # فحص قاعدة البيانات
    if not system.check_database():
        logger.error("❌ Database check failed!")
        return
    
    try:
        if args.mode == 'check':
            # عرض الرموز المتاحة فقط
            logger.info("\n📊 Available symbols:")
            for symbol, timeframes in sorted(system.available_symbols.items()):
                logger.info(f"{symbol}: {list(timeframes.keys())}")
                
        elif args.mode == 'train':
            # التدريب فقط
            await system.train_all_models()
            
        elif args.mode == 'trade':
            # التداول فقط
            system.load_existing_models()
            if not system.models:
                logger.error("❌ No models found! Train first.")
                return
            await system.signal_generation_loop()
            
        elif args.mode == 'full':
            # النظام الكامل
            await system.train_all_models()
            
            # تحميل النماذج الناجحة
            system.models = {}  # مسح النماذج في الذاكرة
            system.load_existing_models()
            
            if system.models:
                await system.signal_generation_loop()
            else:
                logger.warning("⚠️ No successful models to generate signals")
                
    except KeyboardInterrupt:
        logger.info("\n⏹️ System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())