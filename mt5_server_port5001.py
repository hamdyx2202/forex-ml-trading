#!/usr/bin/env python3
"""
MT5 Prediction Server - نسخة بسيطة جداً بدون مكتبات خارجية
Minimal version using only standard library - Port 5001
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import random

class PredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'models_loaded': 0,
                'version': '2.0-minimal',
                'message': 'Server running on port 5001'
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/api/get_performance':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'overall': {
                    'total_trades': 10,
                    'winning_trades': 7,
                    'win_rate': 70.0,
                    'total_profit': 250.50,
                    'avg_pips': 25,
                    'best_trade': 100,
                    'worst_trade': -50
                },
                'by_strategy': [
                    {'strategy': 'scalping', 'trades': 5, 'wins': 4, 'profit': 150},
                    {'strategy': 'short_term', 'trades': 5, 'wins': 3, 'profit': 100.50}
                ],
                'message': 'Test performance data'
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except:
            data = {}
        
        if self.path == '/api/predict_advanced':
            # Log the request
            symbol = data.get('symbol', 'UNKNOWN')
            timeframe = data.get('timeframe', 'UNKNOWN')
            candles = data.get('candles', [])
            
            print(f"\n📊 Prediction request received:")
            print(f"   Symbol: {symbol}")
            print(f"   Timeframe: {timeframe}")
            print(f"   Candles: {len(candles)}")
            
            # Generate test prediction
            current_price = float(candles[-1]['close']) if candles else 1.0850
            
            # Random signal for testing
            signal = random.choice([0, 2])  # 0=Sell, 2=Buy
            confidence = random.uniform(0.75, 0.95)
            
            # Calculate SL/TP
            atr = 0.0020  # Fixed ATR for testing
            if signal == 2:  # Buy
                sl = current_price - (atr * 1.5)
                tp1 = current_price + (atr * 1.0)
                tp2 = current_price + (atr * 2.0)
                tp3 = current_price + (atr * 3.0)
            else:  # Sell
                sl = current_price + (atr * 1.5)
                tp1 = current_price - (atr * 1.0)
                tp2 = current_price - (atr * 2.0)
                tp3 = current_price - (atr * 3.0)
            
            response = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    'scalping': {
                        'signal': signal,
                        'confidence': confidence,
                        'stop_loss': round(sl, 5),
                        'take_profit_1': round(tp1, 5),
                        'take_profit_2': round(tp2, 5),
                        'take_profit_3': round(tp3, 5)
                    }
                },
                'message': 'Test prediction from minimal server on port 5001'
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
            print(f"✅ Sent prediction: {'BUY' if signal == 2 else 'SELL'} ({confidence:.1%})")
            
        elif self.path == '/api/update_trade_result':
            # Log trade result
            ticket = data.get('ticket', 0)
            result = data.get('result', 'unknown')
            profit = data.get('profit', 0)
            
            print(f"\n💰 Trade result received:")
            print(f"   Ticket: #{ticket}")
            print(f"   Result: {result}")
            print(f"   Profit: ${profit}")
            
            response = {'status': 'success'}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log to reduce noise"""
        if '/api/' in args[0]:
            print(f"[{self.log_date_time_string()}] {args[0]}")

def run_server(port=5001):
    """Run the minimal server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, PredictionHandler)
    
    print("=" * 60)
    print("🚀 MT5 Prediction Server (Minimal Version)")
    print("=" * 60)
    print(f"📊 Server URL: http://localhost:{port}")
    print("⚠️  This is a minimal test server without external dependencies")
    print("\n✅ Available endpoints:")
    print("   - GET  /api/health")
    print("   - POST /api/predict_advanced")
    print("   - POST /api/update_trade_result")
    print("   - GET  /api/get_performance")
    print("\n⚠️  NOTE: Running on port 5001 (not 5000)")
    print("\n🔍 Server is ready for connections...")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

if __name__ == '__main__':
    run_server()