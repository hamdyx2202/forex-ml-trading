#!/usr/bin/env python3
"""
🚀 نظام التداول الحي مع التعلم المستمر
📊 يتداول ويتعلم من كل صفقة
🧠 يحلل الأسباب ويحسن الأداء تلقائياً
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import time
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5

# ML Libraries
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Import trained system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_with_real_data import RealDataTrainingSystem

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('live_trading_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingWithContinuousLearning:
    """نظام التداول الحي مع التعلم المستمر"""
    
    def __init__(self):
        logger.info("="*100)
        logger.info("🚀 Live Trading with Continuous Learning System")
        logger.info("📊 Trades, Analyzes, Learns, and Improves Automatically")
        logger.info("="*100)
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return
            
        # Trading parameters
        self.lot_size = 0.01
        self.max_positions = 3
        self.min_confidence = 0.65
        
        # Learning parameters
        self.learning_batch_size = 100  # تعلم بعد كل 100 صفقة
        self.min_trades_for_update = 50
        self.performance_window = 500  # آخر 500 صفقة للتقييم
        
        # Database paths
        self.db_path = './data/forex_ml.db'
        self.trades_db_path = './trading_performance.db'
        
        # Initialize databases
        self.init_databases()
        
        # Load existing models
        self.models = self.load_all_models()
        
        # Trading system
        self.training_system = RealDataTrainingSystem()
        
        # Performance tracking
        self.trade_history = []
        self.learning_history = []
        
    def init_databases(self):
        """تهيئة قواعد البيانات"""
        # قاعدة بيانات الصفقات والأداء
        conn = sqlite3.connect(self.trades_db_path)
        cursor = conn.cursor()
        
        # جدول الصفقات
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timeframe TEXT,
            signal TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_price REAL,
            exit_price REAL,
            sl_price REAL,
            tp_price REAL,
            result TEXT,
            pnl_pips REAL,
            pnl_amount REAL,
            confidence REAL,
            model_prediction TEXT,
            features_snapshot TEXT,
            exit_reason TEXT,
            market_conditions TEXT,
            learning_notes TEXT
        )''')
        
        # جدول التعلم المستمر
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            symbol TEXT,
            timeframe TEXT,
            trades_analyzed INTEGER,
            win_rate_before REAL,
            win_rate_after REAL,
            model_accuracy_before REAL,
            model_accuracy_after REAL,
            patterns_learned TEXT,
            improvements_made TEXT
        )''')
        
        # جدول الأنماط المكتشفة
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS discovered_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_name TEXT,
            pattern_type TEXT,
            success_rate REAL,
            occurrences INTEGER,
            avg_profit_pips REAL,
            conditions TEXT,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
        
    def load_all_models(self):
        """تحميل جميع النماذج المدربة"""
        models = {}
        model_dir = 'models'
        
        if not os.path.exists(model_dir):
            logger.warning("No models directory found")
            return models
            
        for file in os.listdir(model_dir):
            if file.endswith('_ensemble.pkl'):
                try:
                    model_path = os.path.join(model_dir, file)
                    model_data = joblib.load(model_path)
                    model_key = file.replace('_ensemble.pkl', '')
                    models[model_key] = model_data
                    logger.info(f"✅ Loaded model: {model_key}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
                    
        logger.info(f"📊 Total models loaded: {len(models)}")
        return models
        
    def get_current_signals(self):
        """الحصول على إشارات التداول الحالية"""
        signals = []
        
        # الرموز النشطة
        symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
        timeframes = {'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}
        
        for symbol in symbols:
            for tf_name, tf_value in timeframes.items():
                # التحقق من وجود نموذج
                model_key = f"{symbol}_{tf_name}_scalp"
                if model_key not in self.models:
                    continue
                    
                # الحصول على البيانات
                rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 500)
                if rates is None or len(rates) < 200:
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # حساب الميزات
                features = self.training_system.calculate_features(df)
                latest_features = features.iloc[-1:].copy()
                
                # التنبؤ
                try:
                    model_data = self.models[model_key]
                    scaler = model_data['scaler']
                    models_dict = model_data['models']
                    
                    # تطبيق المعايرة
                    features_scaled = scaler.transform(latest_features)
                    
                    # التنبؤ من كل نموذج
                    predictions = []
                    confidences = []
                    
                    for model_name, model in models_dict.items():
                        pred = model.predict(features_scaled)[0]
                        prob = model.predict_proba(features_scaled)[0]
                        predictions.append(pred)
                        confidences.append(np.max(prob))
                        
                    # القرار النهائي
                    final_prediction = max(set(predictions), key=predictions.count)
                    avg_confidence = np.mean(confidences)
                    
                    if avg_confidence >= self.min_confidence and final_prediction != 2:
                        signal = {
                            'symbol': symbol,
                            'timeframe': tf_name,
                            'action': 'BUY' if final_prediction == 0 else 'SELL',
                            'confidence': avg_confidence,
                            'price': df['close'].iloc[-1],
                            'time': datetime.now(),
                            'features': latest_features.to_dict(),
                            'predictions': {
                                'all': predictions,
                                'final': final_prediction,
                                'confidences': confidences
                            }
                        }
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error predicting {symbol} {tf_name}: {str(e)}")
                    
        return signals
        
    def execute_trade(self, signal):
        """تنفيذ الصفقة"""
        symbol = signal['symbol']
        
        # التحقق من عدم وجود صفقة مفتوحة
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            return None
            
        # إعداد الصفقة
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info or not symbol_info.visible:
            mt5.symbol_select(symbol, True)
            
        point = symbol_info.point
        
        # حساب SL و TP
        if signal['action'] == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            sl = signal['price'] - (50 * point)  # 50 نقطة
            tp = signal['price'] + (100 * point)  # 100 نقطة
        else:
            order_type = mt5.ORDER_TYPE_SELL
            sl = signal['price'] + (50 * point)
            tp = signal['price'] - (100 * point)
            
        # إرسال الأمر
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": signal['price'],
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"ML_{signal['confidence']:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # حفظ الصفقة في قاعدة البيانات
            self.save_trade_entry(signal, result)
            logger.info(f"✅ Trade executed: {symbol} {signal['action']} @ {signal['price']}")
            return result
        else:
            logger.error(f"❌ Trade failed: {result.comment}")
            return None
            
    def save_trade_entry(self, signal, trade_result):
        """حفظ بيانات دخول الصفقة"""
        conn = sqlite3.connect(self.trades_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (
            symbol, timeframe, signal, entry_time, entry_price,
            sl_price, tp_price, confidence, model_prediction,
            features_snapshot, market_conditions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            signal['timeframe'],
            signal['action'],
            datetime.now(),
            signal['price'],
            trade_result.request.sl,
            trade_result.request.tp,
            signal['confidence'],
            json.dumps(signal['predictions']),
            json.dumps(signal['features']),
            self.analyze_market_conditions()
        ))
        
        conn.commit()
        conn.close()
        
    def analyze_market_conditions(self):
        """تحليل ظروف السوق الحالية"""
        conditions = {
            'volatility': 'normal',
            'trend': 'neutral',
            'session': self.get_current_session(),
            'major_news': False,
            'timestamp': datetime.now().isoformat()
        }
        return json.dumps(conditions)
        
    def get_current_session(self):
        """تحديد الجلسة التداولية الحالية"""
        hour = datetime.now().hour
        
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 16:
            return 'London'
        elif 13 <= hour < 21:
            return 'NewYork'
        else:
            return 'Sydney'
            
    def monitor_open_positions(self):
        """مراقبة الصفقات المفتوحة"""
        positions = mt5.positions_get()
        
        for position in positions:
            # جلب معلومات الصفقة من قاعدة البيانات
            conn = sqlite3.connect(self.trades_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM trades 
            WHERE symbol = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
            ''', (position.symbol,))
            
            trade_data = cursor.fetchone()
            
            if trade_data:
                # تحليل أداء الصفقة الحالي
                current_pnl = position.profit
                current_pips = self.calculate_pips(
                    position.symbol,
                    position.price_open,
                    position.price_current,
                    position.type
                )
                
                # التحقق من الحاجة لتعديل SL/TP
                self.analyze_and_adjust_trade(position, trade_data, current_pips)
                
            conn.close()
            
    def analyze_and_adjust_trade(self, position, trade_data, current_pips):
        """تحليل وتعديل الصفقة بناءً على الأداء"""
        # إذا كانت الصفقة رابحة بأكثر من 30 نقطة، نقل SL للتعادل
        if current_pips > 30:
            new_sl = position.price_open
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Moved SL to breakeven for {position.symbol}")
                
    def analyze_closed_trades(self):
        """تحليل الصفقات المغلقة والتعلم منها"""
        # جلب آخر الصفقات المغلقة
        conn = sqlite3.connect(self.trades_db_path)
        
        query = '''
        SELECT * FROM trades 
        WHERE exit_time IS NOT NULL 
        AND learning_notes IS NULL
        ORDER BY exit_time DESC 
        LIMIT 50
        '''
        
        trades_df = pd.read_sql_query(query, conn)
        
        if len(trades_df) == 0:
            conn.close()
            return
            
        learning_insights = []
        
        for _, trade in trades_df.iterrows():
            # تحليل كل صفقة
            insights = self.analyze_single_trade(trade)
            learning_insights.append(insights)
            
            # تحديث ملاحظات التعلم
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE trades 
            SET learning_notes = ? 
            WHERE id = ?
            ''', (json.dumps(insights), trade['id']))
            
        conn.commit()
        conn.close()
        
        # تطبيق التعلم على النماذج
        if len(learning_insights) >= self.min_trades_for_update:
            self.update_models_with_insights(learning_insights)
            
    def analyze_single_trade(self, trade):
        """تحليل صفقة واحدة واستخراج الدروس"""
        insights = {
            'trade_id': trade['id'],
            'symbol': trade['symbol'],
            'result': trade['result'],
            'pnl_pips': trade['pnl_pips']
        }
        
        # تحليل الميزات عند الدخول
        features = json.loads(trade['features_snapshot'])
        
        # تحديد الأسباب المحتملة للنجاح/الفشل
        reasons = []
        
        # 1. تحليل قوة الاتجاه
        if 'adx_14' in features:
            if features['adx_14'] < 20 and trade['result'] == 'LOSS':
                reasons.append('Weak trend - ADX < 20')
            elif features['adx_14'] > 40 and trade['result'] == 'WIN':
                reasons.append('Strong trend confirmed - ADX > 40')
                
        # 2. تحليل التشبع
        if 'rsi_14' in features:
            if features['rsi_14'] > 70 and trade['signal'] == 'BUY' and trade['result'] == 'LOSS':
                reasons.append('Overbought entry - RSI > 70')
            elif features['rsi_14'] < 30 and trade['signal'] == 'SELL' and trade['result'] == 'LOSS':
                reasons.append('Oversold entry - RSI < 30')
                
        # 3. تحليل الجلسة التداولية
        market_conditions = json.loads(trade['market_conditions'])
        if market_conditions['session'] == 'Tokyo' and trade['symbol'] in ['EURUSDm', 'GBPUSDm']:
            if trade['result'] == 'LOSS':
                reasons.append('Low liquidity session for this pair')
                
        # 4. تحليل سبب الخروج
        if trade['exit_reason'] == 'SL_HIT':
            reasons.append('Stop loss too tight')
        elif trade['exit_reason'] == 'TP_HIT':
            reasons.append('Good risk/reward achieved')
            
        insights['reasons'] = reasons
        insights['timestamp'] = datetime.now().isoformat()
        
        return insights
        
    def update_models_with_insights(self, insights):
        """تحديث النماذج بناءً على التعلم"""
        logger.info("🧠 Updating models with new insights...")
        
        # تجميع البيانات حسب الرمز والإطار الزمني
        updates_by_symbol = {}
        
        for insight in insights:
            key = f"{insight['symbol']}_{insight.get('timeframe', 'M15')}"
            if key not in updates_by_symbol:
                updates_by_symbol[key] = []
            updates_by_symbol[key].append(insight)
            
        # تحديث كل نموذج
        for model_key, model_insights in updates_by_symbol.items():
            self.retrain_model(model_key, model_insights)
            
    def retrain_model(self, model_key, insights):
        """إعادة تدريب نموذج واحد"""
        try:
            # جلب البيانات الجديدة
            symbol, timeframe = model_key.split('_')[:2]
            
            # استخدام نظام التدريب الأساسي
            result = self.training_system.train_symbol_timeframe(symbol, timeframe)
            
            if result:
                logger.info(f"✅ Successfully retrained {model_key}")
                
                # حفظ سجل التعلم
                self.save_learning_history(model_key, insights, result)
                
        except Exception as e:
            logger.error(f"Error retraining {model_key}: {str(e)}")
            
    def save_learning_history(self, model_key, insights, training_result):
        """حفظ سجل التعلم"""
        conn = sqlite3.connect(self.trades_db_path)
        cursor = conn.cursor()
        
        # حساب معدل الفوز
        wins = sum(1 for i in insights if i.get('result') == 'WIN')
        total = len(insights)
        win_rate = wins / total if total > 0 else 0
        
        # استخراج الأنماط المكتشفة
        patterns = {}
        for insight in insights:
            for reason in insight.get('reasons', []):
                patterns[reason] = patterns.get(reason, 0) + 1
                
        cursor.execute('''
        INSERT INTO learning_history (
            timestamp, symbol, timeframe, trades_analyzed,
            win_rate_before, win_rate_after, patterns_learned,
            improvements_made
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            model_key.split('_')[0],
            model_key.split('_')[1],
            len(insights),
            win_rate,
            training_result.get('ensemble_accuracy', 0),
            json.dumps(patterns),
            json.dumps({'model_updated': True})
        ))
        
        conn.commit()
        conn.close()
        
    def calculate_pips(self, symbol, open_price, current_price, trade_type):
        """حساب النقاط"""
        if 'JPY' in symbol:
            multiplier = 100
        else:
            multiplier = 10000
            
        if trade_type == mt5.ORDER_TYPE_BUY:
            pips = (current_price - open_price) * multiplier
        else:
            pips = (open_price - current_price) * multiplier
            
        return round(pips, 1)
        
    def run_trading_cycle(self):
        """دورة التداول الرئيسية"""
        logger.info("🔄 Starting trading cycle...")
        
        while True:
            try:
                # 1. الحصول على الإشارات
                signals = self.get_current_signals()
                
                # 2. تنفيذ الصفقات الجديدة
                for signal in signals:
                    self.execute_trade(signal)
                    
                # 3. مراقبة الصفقات المفتوحة
                self.monitor_open_positions()
                
                # 4. تحليل الصفقات المغلقة
                self.analyze_closed_trades()
                
                # 5. عرض الإحصائيات
                self.display_performance_stats()
                
                # الانتظار دقيقة واحدة
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                time.sleep(60)
                
    def display_performance_stats(self):
        """عرض إحصائيات الأداء"""
        conn = sqlite3.connect(self.trades_db_path)
        
        # إحصائيات اليوم
        today_stats = pd.read_sql_query('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(pnl_pips) as total_pips,
            AVG(confidence) as avg_confidence
        FROM trades
        WHERE DATE(entry_time) = DATE('now')
        ''', conn)
        
        if today_stats['total_trades'].iloc[0] > 0:
            win_rate = today_stats['wins'].iloc[0] / today_stats['total_trades'].iloc[0] * 100
            
            logger.info(f"""
            📊 Today's Performance:
            - Trades: {today_stats['total_trades'].iloc[0]}
            - Win Rate: {win_rate:.1f}%
            - Total Pips: {today_stats['total_pips'].iloc[0]:.1f}
            - Avg Confidence: {today_stats['avg_confidence'].iloc[0]:.2f}
            """)
            
        conn.close()

def main():
    """تشغيل النظام"""
    system = LiveTradingWithContinuousLearning()
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Live Trading with Continuous Learning Started")
    logger.info("📊 The system will:")
    logger.info("   1. Trade automatically based on ML signals")
    logger.info("   2. Monitor and adjust open positions")
    logger.info("   3. Analyze closed trades for patterns")
    logger.info("   4. Learn and improve models continuously")
    logger.info("="*60 + "\n")
    
    # بدء دورة التداول
    system.run_trading_cycle()
    
    # إغلاق MT5
    mt5.shutdown()

if __name__ == "__main__":
    main()