#!/usr/bin/env python3
"""
🚀 السيرفر الموحد للتنبؤات والتداول
📊 يستقبل 200 شمعة من MT5 ويرسل إشارات مع SL/TP
🧠 يستخدم النظام الموحد للتدريب والتعلم المستمر
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import joblib
import os
import threading
import time
import sqlite3

# Import our unified system
from unified_trading_learning_system import UnifiedTradingLearningSystem

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('unified_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global system instance
unified_system = None

class UnifiedPredictionServer:
    """السيرفر الموحد للتنبؤات"""
    
    def __init__(self):
        logger.info("="*80)
        logger.info("🚀 Unified Prediction Server")
        logger.info("📊 Receives 200 candles, sends signals with SL/TP")
        logger.info("🧠 Uses unified training and continuous learning")
        logger.info("="*80)
        
        # Initialize unified system
        self.system = UnifiedTradingLearningSystem()
        
        # Risk management parameters
        self.risk_params = {
            'default_sl_pips': 50,
            'default_tp_pips': 100,
            'max_sl_pips': 100,
            'min_sl_pips': 20,
            'risk_reward_ratio': 2.0,
            'use_dynamic_sl': True,
            'use_pattern_based_targets': True
        }
        
        # Active signals tracking
        self.active_signals = {}
        self.signal_history = []
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'successful_predictions': 0,
            'total_pips': 0,
            'win_rate': 0
        }
        
        # Start continuous learning thread
        self.start_continuous_learning()
        
    def start_continuous_learning(self):
        """بدء خيط التعلم المستمر"""
        def learning_loop():
            while True:
                try:
                    # تحليل الصفقات والتعلم
                    self.system.analyze_and_learn_from_trades()
                    
                    # تحديث النماذج دورياً
                    self.system.periodic_model_update()
                    
                    # الانتظار 30 دقيقة
                    time.sleep(1800)
                    
                except Exception as e:
                    logger.error(f"Error in learning loop: {str(e)}")
                    time.sleep(300)
                    
        thread = threading.Thread(target=learning_loop, daemon=True)
        thread.start()
        logger.info("✅ Continuous learning thread started")
        
    def process_prediction_request(self, data):
        """معالجة طلب التنبؤ"""
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            candles = data['candles']
            
            logger.info(f"\n📊 Processing prediction for {symbol} {timeframe}")
            logger.info(f"   Received {len(candles)} candles")
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # التأكد من وجود نموذج مدرب
            model_key = f"{symbol}_{timeframe}"
            
            if model_key not in self.system.models:
                logger.info(f"   Training new model for {model_key}...")
                self.system.train_unified_model(symbol, timeframe)
                
            # حساب الميزات
            features = self.system.calculate_adaptive_features(df, symbol, timeframe)
            
            # آخر صف للتنبؤ
            latest_features = features.iloc[-1:].copy()
            
            # التنبؤ مع مطابقة الأنماط
            prediction, confidence = self.system.predict_with_pattern_matching(
                symbol, timeframe, latest_features
            )
            
            # إنشاء الإشارة
            signal = self.create_trading_signal(
                symbol, timeframe, df, prediction, confidence, latest_features
            )
            
            # حفظ الإشارة
            self.save_signal(signal)
            
            logger.info(f"   ✅ Signal: {signal['action']} with {signal['confidence']:.1%} confidence")
            logger.info(f"   📍 SL: {signal['sl_price']:.5f} | TP1: {signal['tp1_price']:.5f} | TP2: {signal['tp2_price']:.5f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def create_trading_signal(self, symbol, timeframe, df, prediction, confidence, features):
        """إنشاء إشارة التداول مع SL/TP"""
        current_price = df['close'].iloc[-1]
        
        # القرار الأساسي
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
            
        # حساب SL/TP الديناميكي
        if self.risk_params['use_dynamic_sl'] and action != 'NONE':
            sl_pips, tp1_pips, tp2_pips = self.calculate_dynamic_levels(
                df, features, action, symbol
            )
        else:
            sl_pips = self.risk_params['default_sl_pips']
            tp1_pips = self.risk_params['default_tp_pips']
            tp2_pips = tp1_pips * 1.5
            
        # تحويل النقاط إلى أسعار
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if action == 'BUY':
            sl_price = current_price - (sl_pips * pip_value)
            tp1_price = current_price + (tp1_pips * pip_value)
            tp2_price = current_price + (tp2_pips * pip_value)
        elif action == 'SELL':
            sl_price = current_price + (sl_pips * pip_value)
            tp1_price = current_price - (tp1_pips * pip_value)
            tp2_price = current_price - (tp2_pips * pip_value)
        else:
            sl_price = tp1_price = tp2_price = current_price
            
        # البحث عن أنماط مربحة
        pattern_info = self.get_matching_patterns(features)
        
        signal = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': float(current_price),
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_pips),
            'tp1_pips': float(tp1_pips),
            'tp2_pips': float(tp2_pips),
            'risk_reward_ratio': float(tp1_pips / sl_pips) if sl_pips > 0 else 0,
            'pattern_match': pattern_info,
            'timestamp': datetime.now().isoformat(),
            'features_snapshot': features.to_dict()
        }
        
        return signal
        
    def calculate_dynamic_levels(self, df, features, action, symbol):
        """حساب مستويات SL/TP الديناميكية"""
        # ATR للتقلب
        atr = features.get('atr_14', df['high'].rolling(14).max() - df['low'].rolling(14).min()).iloc[-1]
        
        # تحويل ATR إلى نقاط
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        atr_pips = atr / pip_value
        
        # SL بناءً على ATR
        sl_pips = max(
            min(atr_pips * 1.5, self.risk_params['max_sl_pips']),
            self.risk_params['min_sl_pips']
        )
        
        # TP بناءً على نسبة المخاطرة/المكافأة
        tp1_pips = sl_pips * self.risk_params['risk_reward_ratio']
        tp2_pips = sl_pips * (self.risk_params['risk_reward_ratio'] + 1)
        
        # تعديل بناءً على قوة الإشارة
        if features.get('adx_14', 0).iloc[-1] > 30:  # اتجاه قوي
            tp1_pips *= 1.2
            tp2_pips *= 1.3
            
        # تعديل بناءً على الجلسة التداولية
        hour = datetime.now().hour
        if 8 <= hour <= 16:  # جلسة لندن
            sl_pips *= 0.9  # SL أضيق
            
        return sl_pips, tp1_pips, tp2_pips
        
    def get_matching_patterns(self, features):
        """البحث عن الأنماط المطابقة"""
        try:
            conn = sqlite3.connect(self.system.unified_db)
            
            # البحث عن أنماط مربحة
            query = """
            SELECT * FROM profitable_patterns
            WHERE avg_profit_pips > 30
            AND occurrences >= 5
            ORDER BY avg_profit_pips DESC
            LIMIT 5
            """
            
            patterns_df = pd.read_sql_query(query, conn)
            conn.close()
            
            matching_patterns = []
            
            for _, pattern in patterns_df.iterrows():
                conditions = json.loads(pattern['feature_conditions'])
                
                # مطابقة بسيطة
                match_score = 0
                
                if 'rsi_range' in conditions:
                    rsi_value = features.get('rsi_14', 50).iloc[-1]
                    if conditions['rsi_range'] in str(self.system.get_range(rsi_value)):
                        match_score += 1
                        
                if match_score > 0:
                    matching_patterns.append({
                        'description': pattern['pattern_description'],
                        'avg_pips': pattern['avg_profit_pips'],
                        'occurrences': pattern['occurrences']
                    })
                    
            return matching_patterns
            
        except Exception as e:
            logger.error(f"Error getting patterns: {str(e)}")
            return []
            
    def save_signal(self, signal):
        """حفظ الإشارة"""
        self.active_signals[signal['symbol']] = signal
        self.signal_history.append(signal)
        
        # تحديث الإحصائيات
        self.performance_stats['total_signals'] += 1
        
        # حفظ في قاعدة البيانات
        try:
            conn = sqlite3.connect(self.system.trading_db)
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
                signal['current_price'],
                signal['sl_price'],
                signal['tp1_price'],
                signal['confidence'],
                json.dumps({'action': signal['action'], 'patterns': signal['pattern_match']}),
                json.dumps(signal['features_snapshot']),
                json.dumps({'timestamp': signal['timestamp']})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signal: {str(e)}")
            
    def process_trade_result(self, data):
        """معالجة نتيجة الصفقة"""
        try:
            symbol = data['symbol']
            result = data['result']  # WIN/LOSS
            entry_price = data['entry_price']
            exit_price = data['exit_price']
            exit_reason = data.get('exit_reason', 'UNKNOWN')
            
            logger.info(f"\n📈 Trade result for {symbol}: {result}")
            
            # حساب النقاط
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            if data.get('action') == 'BUY':
                pips = (exit_price - entry_price) / pip_value
            else:
                pips = (entry_price - exit_price) / pip_value
                
            # تحديث قاعدة البيانات
            conn = sqlite3.connect(self.system.trading_db)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE trades 
            SET exit_time = ?, exit_price = ?, result = ?, 
                pnl_pips = ?, exit_reason = ?
            WHERE symbol = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
            ''', (
                datetime.now(), exit_price, result, pips, exit_reason, symbol
            ))
            
            conn.commit()
            conn.close()
            
            # تحديث الإحصائيات
            self.performance_stats['total_pips'] += pips
            if result == 'WIN':
                self.performance_stats['successful_predictions'] += 1
                
            # حساب معدل الفوز
            if self.performance_stats['total_signals'] > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['successful_predictions'] / 
                    self.performance_stats['total_signals']
                )
                
            logger.info(f"   Pips: {pips:.1f}")
            logger.info(f"   Win Rate: {self.performance_stats['win_rate']:.1%}")
            
            # تحليل فوري للتعلم
            if result == 'LOSS':
                self.analyze_loss_reasons(symbol, data)
                
            return {'status': 'success', 'pips': pips}
            
        except Exception as e:
            logger.error(f"Error processing trade result: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def analyze_loss_reasons(self, symbol, trade_data):
        """تحليل أسباب الخسارة"""
        logger.info(f"   🔍 Analyzing loss reasons for {symbol}")
        
        # هنا يمكن إضافة تحليل مفصل
        reasons = []
        
        if trade_data.get('exit_reason') == 'SL_HIT':
            reasons.append("Stop loss hit - may need wider SL")
            
        # حفظ التحليل للتعلم المستقبلي
        # ...
        
    def get_server_status(self):
        """الحصول على حالة السيرفر"""
        return {
            'status': 'running',
            'models_loaded': len(self.system.models),
            'active_signals': len(self.active_signals),
            'performance': self.performance_stats,
            'last_update': datetime.now().isoformat()
        }

# Global server instance
server = UnifiedPredictionServer()

@app.route('/predict', methods=['POST'])
def predict():
    """نقطة النهاية للتنبؤ"""
    try:
        data = request.json
        signal = server.process_prediction_request(data)
        
        if signal:
            return jsonify(signal)
        else:
            return jsonify({'error': 'Failed to generate signal'}), 500
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/trade_result', methods=['POST'])
def trade_result():
    """نقطة النهاية لنتائج الصفقات"""
    try:
        data = request.json
        result = server.process_trade_result(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Trade result error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """نقطة النهاية لحالة السيرفر"""
    return jsonify(server.get_server_status())

@app.route('/retrain', methods=['POST'])
def retrain():
    """نقطة النهاية لإعادة التدريب"""
    try:
        data = request.json
        symbol = data.get('symbol', 'EURUSDm')
        timeframe = data.get('timeframe', 'M15')
        
        logger.info(f"Manual retrain requested for {symbol} {timeframe}")
        
        # إعادة التدريب
        server.system.train_unified_model(symbol, timeframe, force_retrain=True)
        
        return jsonify({
            'status': 'success',
            'message': f'Retrained {symbol} {timeframe}'
        })
        
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def main():
    """تشغيل السيرفر"""
    logger.info("\n" + "="*80)
    logger.info("🚀 Starting Unified Prediction Server")
    logger.info("📊 Endpoints:")
    logger.info("   POST /predict - Get trading signal with SL/TP")
    logger.info("   POST /trade_result - Report trade results")
    logger.info("   GET  /status - Server status")
    logger.info("   POST /retrain - Force model retrain")
    logger.info("="*80 + "\n")
    
    # تشغيل السيرفر
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()