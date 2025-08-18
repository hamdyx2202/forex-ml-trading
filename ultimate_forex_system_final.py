#!/usr/bin/env python3
"""
🚀 Ultimate Forex ML System - Final Version
✨ نظام متكامل نهائي مع جميع الميزات
📊 يدعم التدريب والتعلم المستمر والفرضيات والتداول
🎯 متوافق مع أحدث Expert Advisor
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb

# Technical Analysis
import talib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ultimate_forex_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexSignal:
    """إشارة تداول للاكسبيرت"""
    def __init__(self, symbol: str, action: str, entry_price: float,
                 stop_loss: float, take_profits: List[float], 
                 confidence: float, strategy: str):
        self.symbol = symbol
        self.action = action  # BUY, SELL, CLOSE
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profits = take_profits
        self.confidence = confidence
        self.strategy = strategy
        self.timestamp = datetime.now()
        self.magic_number = self._generate_magic_number()
    
    def _generate_magic_number(self) -> int:
        """توليد رقم سحري فريد للصفقة"""
        return int(datetime.now().timestamp() % 1000000)
    
    def to_dict(self) -> dict:
        """تحويل لـ dictionary للإرسال للاكسبيرت"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profits[0] if len(self.take_profits) > 0 else 0,
            'take_profit_2': self.take_profits[1] if len(self.take_profits) > 1 else 0,
            'take_profit_3': self.take_profits[2] if len(self.take_profits) > 2 else 0,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'magic_number': self.magic_number,
            'timestamp': self.timestamp.isoformat()
        }

class UltimateTradingSystem:
    """النظام المتكامل النهائي"""
    
    def __init__(self):
        """تهيئة النظام"""
        logger.info("="*100)
        logger.info("🚀 Ultimate Forex ML System - Final Version")
        logger.info("="*100)
        
        self.db_path = "data/forex_ml.db"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # إعدادات النظام
        self.config = self._load_config()
        
        # المكونات
        self.models = {}
        self.scalers = {}
        self.active_signals = []
        self.hypotheses = []
        self.performance_metrics = {}
        
        # إعدادات التداول
        self.trading_config = {
            'min_confidence': 0.65,
            'max_concurrent_trades': 5,
            'risk_per_trade': 0.02,
            'partial_close_percentages': [40, 30, 30],
            'trailing_stop_enabled': True,
            'breakeven_pips': 20
        }
        
        # استراتيجيات التداول
        self.strategies = {
            'scalping': {
                'timeframe': 'M5',
                'lookahead': 20,
                'min_pips': 5,
                'take_profit_ratios': [1.5, 2.5, 3.5],
                'stop_loss_atr': 1.0
            },
            'day_trading': {
                'timeframe': 'H1',
                'lookahead': 24,
                'min_pips': 20,
                'take_profit_ratios': [2.0, 3.0, 4.0],
                'stop_loss_atr': 1.5
            },
            'swing_trading': {
                'timeframe': 'H4',
                'lookahead': 120,
                'min_pips': 50,
                'take_profit_ratios': [2.5, 4.0, 6.0],
                'stop_loss_atr': 2.0
            }
        }
    
    def _load_config(self) -> dict:
        """تحميل الإعدادات"""
        default_config = {
            'database': {
                'path': 'data/forex_ml.db',
                'chunk_size': 5000,
                'max_records': 50000
            },
            'training': {
                'batch_size': 5000,
                'use_all_features': False,  # للأداء
                'max_features': 50,
                'min_accuracy': 0.60
            },
            'symbols': [
                'EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm',
                'BTCUSDm', 'ETHUSDm'
            ],
            'api': {
                'host': 'localhost',
                'port': 5555,
                'timeout': 30
            }
        }
        
        # محاولة تحميل إعدادات مخصصة
        config_path = Path('config/forex_config.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Could not load custom config: {e}")
        
        return default_config
    
    def check_system(self) -> bool:
        """فحص النظام"""
        logger.info("🔍 Checking system...")
        
        # فحص قاعدة البيانات
        if not os.path.exists(self.db_path):
            logger.error(f"❌ Database not found: {self.db_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"✅ Database OK - {count:,} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    def load_data_fast(self, symbol: str, timeframe: str, limit: int = 20000) -> pd.DataFrame:
        """تحميل البيانات بسرعة"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT time, open, high, low, close, volume
                FROM price_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if len(df) < 1000:
                return None
            
            # معالجة البيانات
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time').set_index('time')
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def create_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء ميزات سريعة ومحسنة"""
        features = pd.DataFrame(index=df.index)
        
        try:
            # السعر
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_range'] = (df['high'] - df['low']) / df['close']
            
            # المتوسطات
            for period in [5, 10, 20]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            
            # RSI
            features['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            features['rsi_7'] = talib.RSI(df['close'].values, timeperiod=7)
            
            # MACD
            macd, signal, hist = talib.MACD(df['close'].values)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            features['bb_upper'] = upper
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / middle
            features['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
            
            # ATR
            features['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 14)
            features['atr_ratio'] = features['atr_14'] / df['close']
            
            # Volume
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # Patterns
            features['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, 
                                           df['low'].values, df['close'].values)
            features['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values,
                                               df['low'].values, df['close'].values)
            
            # تنظيف
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def create_targets_fast(self, df: pd.DataFrame, strategy: dict) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """إنشاء أهداف سريعة مع SL/TP"""
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        pip_value = 0.0001 if 'JPY' not in df.index.name else 0.01
        
        targets = []
        confidences = []
        sl_tp_info = []
        
        # معالجة على دفعات صغيرة
        batch_size = 2000
        total = len(df) - lookahead
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            
            # عرض التقدم
            if start % 10000 == 0:
                progress = (start / total) * 100
                logger.info(f"  Progress: {progress:.1f}%")
            
            for i in range(start, end):
                current_price = df['close'].iloc[i]
                future_high = df['high'].iloc[i+1:i+lookahead+1].max()
                future_low = df['low'].iloc[i+1:i+lookahead+1].min()
                
                max_up = (future_high - current_price) / pip_value
                max_down = (current_price - future_low) / pip_value
                
                # حساب ATR للـ SL/TP
                current_atr = df['atr_14'].iloc[i] if 'atr_14' in df.columns else pip_value * 20
                
                if max_up >= min_pips * 2:
                    targets.append(2)  # BUY
                    confidence = min(0.5 + (max_up / (min_pips * 4)) * 0.5, 0.95)
                    confidences.append(confidence)
                    
                    # حساب SL/TP
                    sl = current_price - (current_atr * strategy['stop_loss_atr'])
                    tps = [current_price + (current_atr * ratio) for ratio in strategy['take_profit_ratios']]
                    
                    sl_tp_info.append({
                        'stop_loss': sl,
                        'take_profits': tps,
                        'risk_amount': current_price - sl
                    })
                    
                elif max_down >= min_pips * 2:
                    targets.append(0)  # SELL
                    confidence = min(0.5 + (max_down / (min_pips * 4)) * 0.5, 0.95)
                    confidences.append(confidence)
                    
                    # حساب SL/TP
                    sl = current_price + (current_atr * strategy['stop_loss_atr'])
                    tps = [current_price - (current_atr * ratio) for ratio in strategy['take_profit_ratios']]
                    
                    sl_tp_info.append({
                        'stop_loss': sl,
                        'take_profits': tps,
                        'risk_amount': sl - current_price
                    })
                    
                else:
                    targets.append(1)  # HOLD
                    confidences.append(0.5)
                    sl_tp_info.append(None)
            
            # تنظيف الذاكرة
            if end % 10000 == 0:
                gc.collect()
        
        # ملء الباقي
        remaining = len(df) - len(targets)
        targets.extend([1] * remaining)
        confidences.extend([0.5] * remaining)
        sl_tp_info.extend([None] * remaining)
        
        return np.array(targets), np.array(confidences), sl_tp_info
    
    def train_model(self, symbol: str, timeframe: str, strategy_name: str) -> dict:
        """تدريب نموذج واحد"""
        logger.info(f"\n🎯 Training {symbol} {timeframe} - {strategy_name}")
        
        # تحميل البيانات
        df = self.load_data_fast(symbol, timeframe)
        if df is None:
            logger.error(f"No data for {symbol} {timeframe}")
            return None
        
        # إنشاء الميزات
        features = self.create_features_fast(df)
        if features.empty:
            return None
        
        # إضافة ATR للـ dataframe
        df['atr_14'] = features['atr_14']
        
        # إنشاء الأهداف
        strategy = self.strategies[strategy_name]
        targets, confidences, sl_tp_info = self.create_targets_fast(df, strategy)
        
        # فلترة البيانات عالية الثقة
        high_conf = confidences > 0.6
        X = features[high_conf].values
        y = targets[high_conf]
        conf = confidences[high_conf]
        sl_tp = [sl_tp_info[i] for i in range(len(sl_tp_info)) if high_conf[i]]
        
        if len(X) < 500:
            logger.warning(f"Not enough samples: {len(X)}")
            return None
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # معايرة
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            verbose=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # تقييم
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"✅ Model trained - Accuracy: {accuracy:.2%}, F1: {f1:.2%}")
        
        # حفظ النموذج
        if accuracy >= self.config['training']['min_accuracy']:
            model_key = f"{symbol}_{timeframe}_{strategy_name}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # حفظ على القرص
            model_path = self.models_dir / f"{model_key}.pkl"
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'features': list(features.columns),
                'accuracy': accuracy,
                'strategy': strategy_name
            }, model_path)
            
            logger.info(f"💾 Model saved: {model_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': len(X)
        }
    
    def generate_signal(self, symbol: str, timeframe: str) -> Optional[ForexSignal]:
        """توليد إشارة تداول"""
        # البحث عن نموذج مناسب
        model_key = None
        for key in self.models.keys():
            if symbol in key and timeframe in key:
                model_key = key
                break
        
        if not model_key:
            return None
        
        # تحميل البيانات الحديثة
        df = self.load_data_fast(symbol, timeframe, limit=1000)
        if df is None:
            return None
        
        # إنشاء الميزات
        features = self.create_features_fast(df)
        if features.empty:
            return None
        
        # آخر صف للتنبؤ
        X = features.iloc[-1:].values
        
        # معايرة وتنبؤ
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        X_scaled = scaler.transform(X)
        
        # التنبؤ
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = probabilities.max()
        
        # توليد الإشارة
        if confidence >= self.trading_config['min_confidence']:
            current_price = df['close'].iloc[-1]
            current_atr = features['atr_14'].iloc[-1]
            
            # تحديد الاستراتيجية
            strategy_name = model_key.split('_')[-1]
            strategy = self.strategies[strategy_name]
            
            if prediction == 2:  # BUY
                sl = current_price - (current_atr * strategy['stop_loss_atr'])
                tps = [current_price + (current_atr * ratio) for ratio in strategy['take_profit_ratios']]
                
                signal = ForexSignal(
                    symbol=symbol,
                    action='BUY',
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profits=tps,
                    confidence=confidence,
                    strategy=strategy_name
                )
                
                logger.info(f"🟢 BUY Signal: {symbol} @ {current_price:.5f}")
                return signal
                
            elif prediction == 0:  # SELL
                sl = current_price + (current_atr * strategy['stop_loss_atr'])
                tps = [current_price - (current_atr * ratio) for ratio in strategy['take_profit_ratios']]
                
                signal = ForexSignal(
                    symbol=symbol,
                    action='SELL',
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profits=tps,
                    confidence=confidence,
                    strategy=strategy_name
                )
                
                logger.info(f"🔴 SELL Signal: {symbol} @ {current_price:.5f}")
                return signal
        
        return None
    
    async def train_all_models(self):
        """تدريب جميع النماذج"""
        logger.info("\n📊 Starting model training...")
        
        results = {}
        
        # تدريب لكل رمز واستراتيجية
        for symbol in self.config['symbols']:
            for strategy_name, strategy in self.strategies.items():
                timeframe = strategy['timeframe']
                
                try:
                    result = self.train_model(symbol, timeframe, strategy_name)
                    if result:
                        key = f"{symbol}_{timeframe}_{strategy_name}"
                        results[key] = result
                        
                except Exception as e:
                    logger.error(f"Error training {symbol} {strategy_name}: {e}")
                
                # تنظيف الذاكرة
                gc.collect()
        
        # ملخص
        logger.info("\n📊 Training Summary:")
        successful = sum(1 for r in results.values() if r['accuracy'] >= 0.6)
        logger.info(f"Successful models: {successful}/{len(results)}")
        
        # أفضل النماذج
        if results:
            best = max(results.items(), key=lambda x: x[1]['accuracy'])
            logger.info(f"Best model: {best[0]} - {best[1]['accuracy']:.2%}")
    
    async def start_signal_generation(self):
        """بدء توليد الإشارات"""
        logger.info("\n🔄 Starting signal generation...")
        
        while True:
            try:
                # فحص كل رمز
                for symbol in self.config['symbols']:
                    # توليد إشارات لكل إطار زمني
                    for strategy in self.strategies.values():
                        signal = self.generate_signal(symbol, strategy['timeframe'])
                        
                        if signal:
                            self.active_signals.append(signal)
                            
                            # إرسال للاكسبيرت (يحتاج تطوير API)
                            await self.send_to_expert(signal)
                
                # انتظار دقيقة
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in signal generation: {e}")
                await asyncio.sleep(60)
    
    async def send_to_expert(self, signal: ForexSignal):
        """إرسال الإشارة للاكسبيرت"""
        # هنا يمكن إضافة كود الإرسال عبر API أو ملف
        signal_data = signal.to_dict()
        
        # حفظ في ملف للاكسبيرت
        signals_file = Path('signals/active_signals.json')
        signals_file.parent.mkdir(exist_ok=True)
        
        # قراءة الإشارات الحالية
        if signals_file.exists():
            with open(signals_file, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
        
        # إضافة الإشارة الجديدة
        signals.append(signal_data)
        
        # حفظ
        with open(signals_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        logger.info(f"📤 Signal sent to expert: {signal.symbol} {signal.action}")
    
    def load_existing_models(self):
        """تحميل النماذج الموجودة"""
        logger.info("📁 Loading existing models...")
        
        model_files = list(self.models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            try:
                data = joblib.load(model_file)
                model_key = model_file.stem
                
                self.models[model_key] = data['model']
                self.scalers[model_key] = data['scaler']
                
                logger.info(f"✅ Loaded: {model_key} (Accuracy: {data.get('accuracy', 0):.2%})")
                
            except Exception as e:
                logger.error(f"Error loading {model_file}: {e}")
        
        logger.info(f"📊 Loaded {len(self.models)} models")

async def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Ultimate Forex ML System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'trade', 'full'],
                       help='System mode')
    
    args = parser.parse_args()
    
    # إنشاء النظام
    system = UltimateTradingSystem()
    
    # فحص النظام
    if not system.check_system():
        logger.error("System check failed!")
        return
    
    try:
        if args.mode == 'train':
            # التدريب فقط
            await system.train_all_models()
            
        elif args.mode == 'trade':
            # التداول فقط
            system.load_existing_models()
            await system.start_signal_generation()
            
        elif args.mode == 'full':
            # النظام الكامل
            await system.train_all_models()
            await system.start_signal_generation()
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())