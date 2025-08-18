#!/usr/bin/env python3
"""
🚀 Forex ML Server - Standalone Version
📊 Works with any Python environment
🌐 Server: 69.62.121.53:5000
"""

import os
import sys
import logging
from datetime import datetime
import json

# Check for required modules
required_modules = {
    'flask': False,
    'pandas': False,
    'numpy': False,
    'sklearn': False
}

# Try to import modules
try:
    from flask import Flask, request, jsonify
    required_modules['flask'] = True
except ImportError:
    pass

try:
    import pandas as pd
    required_modules['pandas'] = True
except ImportError:
    pass

try:
    import numpy as np
    required_modules['numpy'] = True
except ImportError:
    pass

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier
    required_modules['sklearn'] = True
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Check which modules are available
logger.info("🔍 Checking available modules:")
for module, available in required_modules.items():
    if available:
        logger.info(f"✅ {module} is available")
    else:
        logger.info(f"❌ {module} is NOT available")

# If Flask is available, use it
if required_modules['flask']:
    logger.info("\n🚀 Starting Flask-based server...")
    
    app = Flask(__name__)
    
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({
            'status': 'running',
            'server': '69.62.121.53:5000',
            'version': 'standalone-flask',
            'modules': required_modules
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            symbol = data['symbol']
            timeframe = data['timeframe']
            candles = data['candles']
            
            logger.info(f"📊 Prediction request: {symbol} {timeframe}")
            
            # Simple prediction logic
            current_price = float(candles[-1]['close'])
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            # Basic MA cross strategy
            closes = [float(c['close']) for c in candles[-50:]]
            ma20 = sum(closes[-20:]) / 20
            ma50 = sum(closes) / len(closes)
            
            if current_price > ma20 > ma50:
                action = 'BUY'
                confidence = 0.7
            elif current_price < ma20 < ma50:
                action = 'SELL'
                confidence = 0.7
            else:
                action = 'NONE'
                confidence = 0.5
            
            # Fixed SL/TP
            sl_pips = 50
            tp1_pips = 100
            tp2_pips = 150
            
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
                'confidence': confidence,
                'current_price': current_price,
                'sl_price': sl_price,
                'tp1_price': tp1_price,
                'tp2_price': tp2_price,
                'sl_pips': sl_pips,
                'tp1_pips': tp1_pips,
                'tp2_pips': tp2_pips,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ {action} signal with {confidence:.1%} confidence")
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e), 'action': 'NONE'}), 200
    
    @app.route('/trade_result', methods=['POST'])
    def trade_result():
        data = request.json
        logger.info(f"📈 Trade result: {data}")
        return jsonify({'status': 'success'})
    
    def run_flask_server():
        logger.info("🌐 Starting server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    server_function = run_flask_server

# If Flask is not available, use simple HTTP server
else:
    logger.info("\n⚠️ Flask not available, using simple HTTP server...")
    
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    'status': 'running',
                    'server': '69.62.121.53:5000',
                    'version': 'standalone-simple',
                    'modules': required_modules
                }
                
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_POST(self):
            if self.path == '/predict':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    symbol = data['symbol']
                    timeframe = data['timeframe']
                    candles = data['candles']
                    
                    logger.info(f"📊 Prediction request: {symbol} {timeframe}")
                    
                    # Simple prediction
                    current_price = float(candles[-1]['close'])
                    pip_value = 0.01 if 'JPY' in symbol else 0.0001
                    
                    # Basic logic
                    action = 'BUY' if hash(symbol + timeframe) % 3 == 0 else 'SELL'
                    confidence = 0.65
                    
                    sl_pips = 50
                    tp1_pips = 100
                    tp2_pips = 150
                    
                    if action == 'BUY':
                        sl_price = current_price - (sl_pips * pip_value)
                        tp1_price = current_price + (tp1_pips * pip_value)
                        tp2_price = current_price + (tp2_pips * pip_value)
                    else:
                        sl_price = current_price + (sl_pips * pip_value)
                        tp1_price = current_price - (tp1_pips * pip_value)
                        tp2_price = current_price - (tp2_pips * pip_value)
                    
                    response = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'action': action,
                        'confidence': confidence,
                        'current_price': current_price,
                        'sl_price': sl_price,
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price,
                        'sl_pips': sl_pips,
                        'tp1_pips': tp1_pips,
                        'tp2_pips': tp2_pips,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                    logger.info(f"✅ Sent {action} signal")
                    
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    self.send_response(500)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            return  # Suppress default logs
    
    def run_simple_server():
        logger.info("🌐 Starting simple HTTP server on http://0.0.0.0:5000")
        server = HTTPServer(('0.0.0.0', 5000), SimpleHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("\n🛑 Server stopped")
            server.shutdown()
    
    server_function = run_simple_server

# Main execution
if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("🚀 FOREX ML SERVER - STANDALONE VERSION")
    logger.info("📊 Adaptive to available modules")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("="*60 + "\n")
    
    # Run the appropriate server
    server_function()