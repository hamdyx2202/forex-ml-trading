#!/usr/bin/env python3
"""
🚀 Simple HTTP Server for Forex ML
📊 Works without Flask
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Simple prediction system
class SimplePredictionSystem:
    def __init__(self):
        self.models = {}
        logger.info("🚀 Simple Prediction System initialized")
        
    def predict(self, symbol, timeframe, candles):
        """Make simple prediction based on indicators"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Calculate simple indicators
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            
            # Simple RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Latest values
            current_price = float(df['close'].iloc[-1])
            sma20 = df['sma20'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Simple logic
            confidence = 0.7
            
            if current_price > sma20 and sma20 > sma50 and current_rsi < 70:
                action = 'BUY'
                confidence = 0.75
            elif current_price < sma20 and sma20 < sma50 and current_rsi > 30:
                action = 'SELL'
                confidence = 0.75
            else:
                action = 'NONE'
                confidence = 0.5
                
            # Calculate SL/TP
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
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
                
            return {
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
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}

# Global system
system = SimplePredictionSystem()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'running',
                'server': '69.62.121.53:5000',
                'version': 'simple-1.0'
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
                
                # Make prediction
                result = system.predict(symbol, timeframe, candles)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
                logger.info(f"✅ Sent {result.get('action', 'NONE')} signal")
                
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                
        elif self.path == '/trade_result':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                logger.info(f"📈 Trade result: {data}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Custom logging
        logger.info(f"{self.address_string()} - {format % args}")

def main():
    logger.info("\n" + "="*60)
    logger.info("🚀 SIMPLE FOREX ML SERVER")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("📊 Ready for MT5 connections")
    logger.info("="*60 + "\n")
    
    server = HTTPServer(('0.0.0.0', 5000), RequestHandler)
    logger.info("🔌 Server listening on port 5000...")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()