#!/usr/bin/env python3
"""
MT5 Prediction Server - خادم التنبؤات المتقدم
يستقبل البيانات من MT5 ويرسل التنبؤات مع SL/TP
"""

from flask import Flask, request, jsonify
import json
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
from pathlib import Path
import joblib
import logging
from logging.handlers import RotatingFileHandler

# استيراد النظام المتقدم
from train_advanced_complete import AdvancedCompleteTrainer
from continuous_learner_advanced_v2 import AdvancedContinuousLearnerV2

app = Flask(__name__)

# إعداد السجلات
def setup_logging():
    if not app.debug:
        file_handler = RotatingFileHandler('logs/mt5_server.log', maxBytes=10240000, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('MT5 Prediction Server startup')

# تهيئة النظام
trainer = AdvancedCompleteTrainer()
continuous_learner = AdvancedContinuousLearnerV2()

# ذاكرة التخزين المؤقت للنماذج
models_cache = {}

@app.route('/api/predict_advanced', methods=['POST'])
def predict_advanced():
    """نقطة نهاية التنبؤ المتقدمة"""
    try:
        data = request.get_json()
        
        # استخراج البيانات
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        candles = data.get('candles', [])
        requested_strategies = data.get('strategies', ['scalping', 'short_term'])
        
        app.logger.info(f"📊 طلب تنبؤ: {symbol} {timeframe}")
        
        # التحقق من البيانات
        if not symbol or not timeframe or len(candles) < 100:
            return jsonify({
                'error': 'Invalid data',
                'message': 'Symbol, timeframe and at least 100 candles required'
            }), 400
        
        # تحويل الشموع إلى DataFrame
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        
        # التأكد من وجود الأعمدة المطلوبة
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({
                    'error': 'Missing columns',
                    'message': f'Column {col} is required'
                }), 400
        
        # إنشاء الميزات المتقدمة
        features_df = trainer.create_ultra_advanced_features(df, symbol)
        
        if features_df.empty:
            return jsonify({
                'error': 'Feature creation failed',
                'message': 'Could not create features from data'
            }), 500
        
        # الحصول على آخر صف من الميزات
        latest_features = features_df.values[-1]
        
        # التنبؤ باستخدام النماذج المختلفة
        predictions = {}
        
        for strategy in requested_strategies:
            if strategy not in trainer.training_strategies:
                continue
            
            # محاولة الحصول على النموذج من الذاكرة المؤقتة أو تحميله
            model_key = f"{symbol}_{timeframe}_{strategy}"
            
            if model_key not in models_cache:
                model_path = Path(f"models/{symbol}_{timeframe}/{strategy}/latest_models.pkl")
                
                if not model_path.exists():
                    # محاولة استخدام نموذج عام
                    general_model_path = Path(f"models/EURUSD_H1/{strategy}/latest_models.pkl")
                    if general_model_path.exists():
                        model_path = general_model_path
                        app.logger.warning(f"استخدام نموذج عام لـ {symbol} {timeframe} {strategy}")
                    else:
                        app.logger.warning(f"لا يوجد نموذج لـ {strategy}")
                        continue
                
                try:
                    # تحميل النموذج
                    model_data = joblib.load(model_path)
                    models_cache[model_key] = model_data
                except Exception as e:
                    app.logger.error(f"خطأ في تحميل النموذج: {e}")
                    continue
            
            model_data = models_cache[model_key]
            
            try:
                # التنبؤ
                prediction = predict_with_model(
                    model_data, 
                    latest_features, 
                    df, 
                    symbol, 
                    strategy
                )
                
                if prediction:
                    predictions[strategy] = prediction
                    
            except Exception as e:
                app.logger.error(f"خطأ في التنبؤ لـ {strategy}: {e}")
        
        # إضافة معلومات السوق
        current_price = float(df['close'].iloc[-1])
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions
        }
        
        # حفظ التنبؤ في قاعدة البيانات للتعلم
        save_prediction_to_db(symbol, timeframe, predictions)
        
        app.logger.info(f"✅ تم إرسال {len(predictions)} تنبؤ لـ {symbol} {timeframe}")
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"خطأ عام: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

def predict_with_model(model_data, features, df, symbol, strategy):
    """التنبؤ باستخدام نموذج محدد"""
    
    # استخراج المكونات
    models = model_data.get('signal_models', {})
    sl_models = model_data.get('sl_models', {})
    tp_models = model_data.get('tp_models', {})
    scaler = model_data.get('scaler')
    
    if not models:
        return None
    
    # معايرة البيانات
    if scaler:
        features_scaled = scaler.transform(features.reshape(1, -1))
    else:
        features_scaled = features.reshape(1, -1)
    
    # التنبؤ بالإشارة
    signal_predictions = []
    signal_probabilities = []
    
    for model_name, model in models.items():
        try:
            pred = model.predict(features_scaled)[0]
            signal_predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                signal_probabilities.append(proba)
        except:
            continue
    
    if not signal_predictions:
        return None
    
    # حساب الإشارة النهائية
    final_signal = int(np.median(signal_predictions))
    
    # حساب الثقة
    if signal_probabilities:
        avg_proba = np.mean(signal_probabilities, axis=0)
        confidence = float(np.max(avg_proba))
    else:
        signal_counts = np.bincount(signal_predictions)
        confidence = float(signal_counts[final_signal] / len(signal_predictions))
    
    # إذا كانت الإشارة محايدة، لا نرسل تنبؤ
    if final_signal == 1:
        return None
    
    current_price = float(df['close'].iloc[-1])
    
    # التنبؤ بـ Stop Loss
    sl_predictions = []
    for model_name, model in sl_models.items():
        try:
            sl_pred = model.predict(features_scaled)[0]
            sl_predictions.append(sl_pred)
        except:
            continue
    
    if sl_predictions:
        final_sl = float(np.median(sl_predictions))
    else:
        # حساب SL افتراضي بناءً على ATR
        atr = calculate_atr(df)
        sl_distance = atr * trainer.sl_tp_settings[strategy]['stop_loss_atr']
        final_sl = current_price - sl_distance if final_signal == 2 else current_price + sl_distance
    
    # التنبؤ بـ Take Profit
    tp_levels = {}
    for tp_level in ['tp1', 'tp2', 'tp3']:
        tp_predictions = []
        
        if tp_level in tp_models:
            for model_name, model in tp_models[tp_level].items():
                try:
                    tp_pred = model.predict(features_scaled)[0]
                    tp_predictions.append(tp_pred)
                except:
                    continue
        
        if tp_predictions:
            tp_levels[tp_level] = float(np.median(tp_predictions))
        else:
            # حساب TP افتراضي
            ratios = trainer.sl_tp_settings[strategy]['take_profit_ratios']
            ratio_idx = int(tp_level[-1]) - 1
            
            if ratio_idx < len(ratios):
                tp_distance = abs(current_price - final_sl) * ratios[ratio_idx]
                tp_levels[tp_level] = current_price + tp_distance if final_signal == 2 else current_price - tp_distance
    
    return {
        'signal': final_signal,
        'confidence': confidence,
        'stop_loss': final_sl,
        'take_profit_1': tp_levels.get('tp1', current_price),
        'take_profit_2': tp_levels.get('tp2', current_price),
        'take_profit_3': tp_levels.get('tp3', current_price),
        'strategy_settings': trainer.training_strategies[strategy],
        'sl_tp_settings': trainer.sl_tp_settings[strategy]
    }

def calculate_atr(df, period=14):
    """حساب ATR"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    
    return atr

def save_prediction_to_db(symbol, timeframe, predictions):
    """حفظ التنبؤات في قاعدة البيانات للتعلم"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.cursor()
        
        # إنشاء جدول التنبؤات إذا لم يكن موجوداً
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                timeframe TEXT,
                strategy TEXT,
                signal INTEGER,
                confidence REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL
            )
        """)
        
        # حفظ كل تنبؤ
        for strategy, pred in predictions.items():
            cursor.execute("""
                INSERT INTO predictions 
                (symbol, timeframe, strategy, signal, confidence, 
                 stop_loss, take_profit_1, take_profit_2, take_profit_3)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, strategy,
                pred['signal'], pred['confidence'],
                pred['stop_loss'], pred['take_profit_1'],
                pred['take_profit_2'], pred['take_profit_3']
            ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        app.logger.error(f"خطأ في حفظ التنبؤ: {e}")

@app.route('/api/update_trade_result', methods=['POST'])
def update_trade_result():
    """تحديث نتيجة الصفقة للتعلم"""
    try:
        data = request.get_json()
        
        # حفظ النتيجة في قاعدة البيانات
        save_trade_result(data)
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def save_trade_result(data):
    """حفظ نتيجة الصفقة"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.cursor()
        
        # إنشاء جدول النتائج إذا لم يكن موجوداً
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticket INTEGER,
                symbol TEXT,
                timeframe TEXT,
                strategy TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                profit REAL,
                profit_pips REAL,
                duration_minutes INTEGER,
                result TEXT
            )
        """)
        
        cursor.execute("""
            INSERT INTO trade_results 
            (ticket, symbol, timeframe, strategy, entry_price, exit_price, 
             stop_loss, take_profit, profit, profit_pips, duration_minutes, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('ticket'),
            data.get('symbol'),
            data.get('timeframe'),
            data.get('strategy'),
            data.get('entry_price'),
            data.get('exit_price'),
            data.get('stop_loss'),
            data.get('take_profit'),
            data.get('profit'),
            data.get('profit_pips'),
            data.get('duration_minutes'),
            data.get('result')
        ))
        
        conn.commit()
        conn.close()
        
        app.logger.info(f"✅ تم حفظ نتيجة الصفقة #{data.get('ticket')}")
        
    except Exception as e:
        app.logger.error(f"خطأ في حفظ نتيجة الصفقة: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models_cache),
        'version': '2.0'
    })

@app.route('/api/get_performance', methods=['GET'])
def get_performance():
    """الحصول على أداء النظام"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        
        # إحصائيات عامة
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(profit) as total_profit,
                AVG(profit_pips) as avg_pips,
                MAX(profit) as best_trade,
                MIN(profit) as worst_trade
            FROM trade_results
            WHERE timestamp > datetime('now', '-30 days')
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        stats = cursor.fetchone()
        
        # إحصائيات حسب الاستراتيجية
        query2 = """
            SELECT 
                strategy,
                COUNT(*) as trades,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                SUM(profit) as profit
            FROM trade_results
            WHERE timestamp > datetime('now', '-30 days')
            GROUP BY strategy
        """
        
        strategy_stats = pd.read_sql_query(query2, conn)
        
        conn.close()
        
        return jsonify({
            'overall': {
                'total_trades': stats[0] or 0,
                'winning_trades': stats[1] or 0,
                'win_rate': (stats[1] / stats[0] * 100) if stats[0] else 0,
                'total_profit': stats[2] or 0,
                'avg_pips': stats[3] or 0,
                'best_trade': stats[4] or 0,
                'worst_trade': stats[5] or 0
            },
            'by_strategy': strategy_stats.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_logging()
    
    # إنشاء مجلد السجلات
    Path("logs").mkdir(exist_ok=True)
    
    # تشغيل الخادم
    app.run(host='0.0.0.0', port=5000, debug=False)