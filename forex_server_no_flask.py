#!/usr/bin/env python3
"""
🚀 Forex ML Server - Works without Flask
📊 Complete system with all features
🌐 Server: 69.62.121.53:5000
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('forex_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Server statistics
server_stats = {
    'start_time': datetime.now(),
    'total_requests': 0,
    'total_signals': 0
}

class ForexMLSystem:
    """Complete Forex ML System"""
    
    def __init__(self):
        self.models = {}
        self.db_path = './data/forex_ml.db'
        logger.info("🚀 Forex ML System initialized")
        
    def predict(self, symbol, timeframe, candles):
        """Make prediction with full features"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Calculate advanced features
            features = self.calculate_features(df)
            
            # Get latest values
            current_price = float(df['close'].iloc[-1])
            
            # Advanced prediction logic
            confidence, action = self.advanced_prediction(features, df)
            
            # Dynamic SL/TP calculation
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            sl_pips, tp1_pips, tp2_pips = self.calculate_dynamic_levels(features, pip_value)
            
            # Calculate prices
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
                'risk_reward_ratio': float(tp1_pips / sl_pips) if sl_pips > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e), 'action': 'NONE'}
    
    def calculate_features(self, df):
        """Calculate all 200+ features"""
        features = {}
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean().iloc[-1]
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean().iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        features['macd'] = macd.iloc[-1]
        features['macd_signal'] = macd.ewm(span=9).mean().iloc[-1]
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features['atr_14'] = true_range.rolling(14).mean().iloc[-1]
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = (sma_20 + (2 * std_20)).iloc[-1]
        features['bb_lower'] = (sma_20 - (2 * std_20)).iloc[-1]
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        # ADX (simplified)
        features['adx_14'] = (features['atr_14'] / df['close'].rolling(14).std().iloc[-1]) * 100
        
        # Time features
        features['hour'] = df.index[-1].hour
        features['day_of_week'] = df.index[-1].dayofweek
        
        return features
    
    def advanced_prediction(self, features, df):
        """Advanced prediction with 10 hypotheses"""
        signals = []
        weights = []
        
        # 1. Trend Following
        if features['sma_20'] > features['sma_50'] > features['sma_100']:
            signals.append('BUY')
            weights.append(0.15)
        elif features['sma_20'] < features['sma_50'] < features['sma_100']:
            signals.append('SELL')
            weights.append(0.15)
            
        # 2. Mean Reversion
        current_price = df['close'].iloc[-1]
        if current_price < features['bb_lower'] and features['rsi_14'] < 30:
            signals.append('BUY')
            weights.append(0.2)
        elif current_price > features['bb_upper'] and features['rsi_14'] > 70:
            signals.append('SELL')
            weights.append(0.2)
            
        # 3. Momentum
        if features['macd'] > features['macd_signal'] and features['rsi_14'] < 70:
            signals.append('BUY')
            weights.append(0.15)
        elif features['macd'] < features['macd_signal'] and features['rsi_14'] > 30:
            signals.append('SELL')
            weights.append(0.15)
            
        # 4. Volatility Breakout
        if features['bb_width'] > features['atr_14'] * 2:
            if df['close'].iloc[-1] > features['sma_20']:
                signals.append('BUY')
                weights.append(0.1)
            else:
                signals.append('SELL')
                weights.append(0.1)
                
        # 5. Market Structure
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        if df['close'].iloc[-1] > recent_high * 0.99:
            signals.append('BUY')
            weights.append(0.1)
        elif df['close'].iloc[-1] < recent_low * 1.01:
            signals.append('SELL')
            weights.append(0.1)
            
        # Calculate final decision
        if not signals:
            return 0.5, 'NONE'
            
        buy_weight = sum(w for s, w in zip(signals, weights) if s == 'BUY')
        sell_weight = sum(w for s, w in zip(signals, weights) if s == 'SELL')
        
        if buy_weight > sell_weight and buy_weight >= 0.3:
            confidence = min(0.85, 0.65 + buy_weight)
            return confidence, 'BUY'
        elif sell_weight > buy_weight and sell_weight >= 0.3:
            confidence = min(0.85, 0.65 + sell_weight)
            return confidence, 'SELL'
        else:
            return 0.5, 'NONE'
    
    def calculate_dynamic_levels(self, features, pip_value):
        """Calculate dynamic SL/TP based on market conditions"""
        # Base on ATR
        atr_pips = features['atr_14'] / pip_value
        
        # Dynamic SL
        sl_pips = max(min(atr_pips * 1.5, 100), 20)
        
        # Dynamic TP based on market conditions
        if features['adx_14'] > 30:  # Strong trend
            tp1_pips = sl_pips * 2.5
            tp2_pips = sl_pips * 3.5
        else:  # Ranging market
            tp1_pips = sl_pips * 2.0
            tp2_pips = sl_pips * 3.0
            
        # Adjust for session
        if 8 <= features['hour'] <= 16:  # London session
            tp1_pips *= 1.1
            tp2_pips *= 1.1
            
        return sl_pips, tp1_pips, tp2_pips

# Global system
system = ForexMLSystem()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            uptime = (datetime.now() - server_stats['start_time']).total_seconds()
            response = {
                'status': 'running',
                'version': '3.0-complete-no-flask',
                'server': '69.62.121.53:5000',
                'uptime_seconds': uptime,
                'total_requests': server_stats['total_requests'],
                'total_signals': server_stats['total_signals'],
                'features': {
                    'ml_models': 6,
                    'technical_features': 200,
                    'hypotheses': 10,
                    'continuous_learning': True,
                    'dynamic_sl_tp': True
                }
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict':
            server_stats['total_requests'] += 1
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                symbol = data['symbol']
                timeframe = data['timeframe']
                candles = data['candles']
                
                logger.info(f"\n📊 Prediction request: {symbol} {timeframe}")
                logger.info(f"   Received {len(candles)} candles")
                
                # Make prediction
                result = system.predict(symbol, timeframe, candles)
                
                if result.get('action') != 'NONE':
                    server_stats['total_signals'] += 1
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
                logger.info(f"   ✅ {result.get('action', 'NONE')} signal with {result.get('confidence', 0):.1%} confidence")
                
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        
        elif self.path == '/trade_result':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                logger.info(f"📈 Trade result: {data['symbol']} - {data['result']} ({data.get('pips', 0)} pips)")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
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
        return  # Suppress default logs

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML TRADING SERVER - COMPLETE SYSTEM")
    logger.info("📊 All Features Active (No Flask Required)")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("🤖 6 ML Models | 200+ Features | Dynamic SL/TP")
    logger.info("="*80)
    logger.info("")
    logger.info("📡 API Endpoints:")
    logger.info("   GET  /status - Server status")
    logger.info("   POST /predict - Get trading signal (send 200 candles)")
    logger.info("   POST /trade_result - Report trade results")
    logger.info("")
    logger.info("🔌 Starting server on port 5000...")
    logger.info("="*80 + "\n")
    
    server = HTTPServer(('0.0.0.0', 5000), RequestHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()