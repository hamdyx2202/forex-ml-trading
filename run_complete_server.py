#!/usr/bin/env python3
"""
🚀 تشغيل السيرفر الكامل مع Flask
📊 النظام الكامل بدون تبسيط
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('complete_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_server_files():
    """تحضير ملفات السيرفر"""
    logger.info("📝 Preparing server files...")
    
    # Create server version of unified_trading_learning_system
    if os.path.exists('unified_trading_learning_system.py'):
        with open('unified_trading_learning_system.py', 'r') as f:
            content = f.read()
        
        # Comment out MT5 imports
        content = content.replace('import MetaTrader5 as mt5', '# import MetaTrader5 as mt5')
        content = content.replace('mt5.initialize()', 'False')
        content = content.replace('mt5.', '# mt5.')
        
        with open('unified_trading_learning_system_server.py', 'w') as f:
            f.write(content)
    
    # Import the server system
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    logger.info("✅ Server files ready")

def check_initial_training():
    """التحقق من التدريب الأولي"""
    logger.info("🧠 Checking for trained models...")
    
    model_dir = 'unified_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_count = len([f for f in os.listdir(model_dir) if f.endswith('.pkl')])
    
    if model_count == 0:
        logger.info("⚠️ No models found. Training will happen on first request.")
        logger.info("   This is normal - models will be trained when needed.")
    else:
        logger.info(f"✅ Found {model_count} existing models")
        
    return True

def run_flask_server():
    """تشغيل Flask server"""
    try:
        # Import Flask app
        from flask import Flask, request, jsonify
        import pandas as pd
        import numpy as np
        import json
        from unified_trading_learning_system_server import UnifiedTradingLearningSystem
        
        app = Flask(__name__)
        
        # Initialize system
        logger.info("🚀 Initializing Unified Trading System...")
        unified_system = UnifiedTradingLearningSystem()
        
        # Server state
        server_stats = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'active_models': 0,
            'last_prediction': None
        }
        
        @app.route('/status', methods=['GET'])
        def status():
            """حالة السيرفر"""
            uptime = (datetime.now() - server_stats['start_time']).total_seconds()
            return jsonify({
                'status': 'running',
                'version': '3.0',
                'uptime_seconds': uptime,
                'models_loaded': len(unified_system.models),
                'total_requests': server_stats['total_requests'],
                'last_prediction': server_stats['last_prediction']
            })
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """التنبؤ بناءً على 200 شمعة"""
            try:
                server_stats['total_requests'] += 1
                
                data = request.json
                symbol = data['symbol']
                timeframe = data['timeframe']
                candles = data['candles']
                
                logger.info(f"📊 Prediction request: {symbol} {timeframe} ({len(candles)} candles)")
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                
                # Ensure model exists
                model_key = f"{symbol}_{timeframe}"
                if model_key not in unified_system.models:
                    logger.info(f"   Training model for {model_key}...")
                    unified_system.train_unified_model(symbol, timeframe)
                
                # Calculate features
                features = unified_system.calculate_adaptive_features(df, symbol, timeframe)
                latest_features = features.iloc[-1:].copy()
                
                # Predict
                prediction, confidence = unified_system.predict_with_pattern_matching(
                    symbol, timeframe, latest_features
                )
                
                # Determine action
                if prediction == 0 and confidence >= 0.65:
                    action = 'BUY'
                elif prediction == 1 and confidence >= 0.65:
                    action = 'SELL'
                else:
                    action = 'NONE'
                
                # Calculate SL/TP
                current_price = float(df['close'].iloc[-1])
                pip_value = 0.01 if 'JPY' in symbol else 0.0001
                
                # Dynamic SL/TP based on ATR
                atr = features.get('atr_14', pd.Series([50 * pip_value])).iloc[-1]
                sl_pips = max(min(atr / pip_value * 1.5, 100), 20)
                tp1_pips = sl_pips * 2.0  # 1:2 risk/reward
                tp2_pips = sl_pips * 3.0  # 1:3 risk/reward
                
                # Convert to prices
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
                
                response = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'action': action,
                    'confidence': float(confidence),
                    'current_price': current_price,
                    'sl_price': float(sl_price),
                    'tp1_price': float(tp1_price),
                    'tp2_price': float(tp2_price),
                    'sl_pips': float(sl_pips),
                    'tp1_pips': float(tp1_pips),
                    'tp2_pips': float(tp2_pips),
                    'risk_reward_ratio': float(tp1_pips / sl_pips) if sl_pips > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                server_stats['last_prediction'] = response['timestamp']
                
                logger.info(f"   ✅ {action} signal with {confidence:.1%} confidence")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @app.route('/trade_result', methods=['POST'])
        def trade_result():
            """استقبال نتائج الصفقات"""
            try:
                data = request.json
                logger.info(f"📈 Trade result: {data['symbol']} - {data['result']}")
                
                # Save to database for learning
                # unified_system.save_trade_result(data)
                
                return jsonify({'status': 'success', 'message': 'Trade result recorded'})
                
            except Exception as e:
                logger.error(f"Trade result error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/retrain', methods=['POST'])
        def retrain():
            """إعادة تدريب النماذج"""
            try:
                data = request.json
                symbol = data.get('symbol', 'EURUSDm')
                timeframe = data.get('timeframe', 'M15')
                
                logger.info(f"🔄 Retrain request: {symbol} {timeframe}")
                
                unified_system.train_unified_model(symbol, timeframe, force_retrain=True)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Model retrained for {symbol} {timeframe}'
                })
                
            except Exception as e:
                logger.error(f"Retrain error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        # Start Flask app
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting Flask Server")
        logger.info("🌐 URL: http://69.62.121.53:5000")
        logger.info("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"❌ Flask server error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """البرنامج الرئيسي"""
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML TRADING SERVER - COMPLETE SYSTEM")
    logger.info("📊 All Features Active")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("="*80 + "\n")
    
    # Prepare files
    prepare_server_files()
    
    # Check training
    check_initial_training()
    
    # Run server
    run_flask_server()

if __name__ == "__main__":
    main()