#!/usr/bin/env python3
"""
MT5 Bridge Server (Linux Version) - خادم API مبسط للعمل على Linux
يستقبل طلبات من EA ويرد بإشارات بناءً على البيانات المرسلة
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Optional

# تكوين Flask لقبول أي content-type
class FlexibleRequest(Flask):
    def make_default_options_response(self):
        rv = super().make_default_options_response()
        rv.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return rv

# إضافة المسار
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json'

class SimpleBridgeServer:
    """خادم مبسط يعمل على Linux بدون MT5"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # ذاكرة للبيانات والإشارات
        self.price_history = {}
        self.recent_signals = {}
        self.active_trades = {}
        self.trade_results = []
        
        logger.add("logs/mt5_bridge_linux.log", rotation="1 day", retention="30 days")
        logger.info("Linux Bridge Server initialized")
        
        # تحميل النماذج المدربة إذا وُجدت
        self.models_loaded = self._load_models()
    
    def _load_models(self) -> bool:
        """تحميل النماذج المدربة"""
        try:
            models_path = "models/"
            if os.path.exists(models_path):
                # هنا يمكن تحميل النماذج المحفوظة
                logger.info("Models directory found")
                return True
            else:
                logger.warning("Models directory not found - using simple strategy")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_signal(self, symbol: str, price: float) -> Dict:
        """توليد إشارة تداول بناءً على البيانات المستلمة"""
        try:
            # حفظ السعر في التاريخ
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': price,
                'time': datetime.now()
            })
            
            # الاحتفاظ بآخر 100 سعر فقط
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # استخدام استراتيجية بسيطة إذا لم تكن النماذج متاحة
            signal = self._generate_simple_signal(symbol, price)
            
            # حفظ الإشارة
            self.recent_signals[symbol] = {
                'signal': signal,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Generated signal for {symbol}: {signal['action']} at {price}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {"action": "NO_TRADE", "confidence": 0, "error": str(e)}
    
    def _generate_simple_signal(self, symbol: str, current_price: float) -> Dict:
        """استراتيجية بسيطة للتداول"""
        
        # إذا لم يكن لدينا بيانات كافية
        if len(self.price_history.get(symbol, [])) < 20:
            return {
                "action": "NO_TRADE",
                "confidence": 0,
                "reason": "Not enough data"
            }
        
        # حساب المتوسطات
        prices = [p['price'] for p in self.price_history[symbol]]
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        # حساب RSI بسيط
        price_changes = np.diff(prices[-15:])
        gains = price_changes[price_changes > 0]
        losses = -price_changes[price_changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # توليد الإشارة
        action = "NO_TRADE"
        confidence = 0.5
        
        # شروط الشراء
        if sma_10 > sma_20 and rsi < 70 and current_price > sma_10:
            action = "BUY"
            confidence = 0.7 + (0.2 * (70 - rsi) / 70)  # ثقة أعلى مع RSI أقل
            
        # شروط البيع
        elif sma_10 < sma_20 and rsi > 30 and current_price < sma_10:
            action = "SELL"
            confidence = 0.7 + (0.2 * (rsi - 30) / 70)  # ثقة أعلى مع RSI أعلى
        
        # حساب SL و TP
        atr_estimate = np.std(prices[-20:]) * 2  # تقدير ATR
        
        if action == "BUY":
            sl = current_price - (atr_estimate * 2)
            tp = current_price + (atr_estimate * 3)
        elif action == "SELL":
            sl = current_price + (atr_estimate * 2)
            tp = current_price - (atr_estimate * 3)
        else:
            sl = current_price
            tp = current_price
        
        return {
            "action": action,
            "confidence": round(confidence, 2),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "lot": 0.01,  # حجم ثابت للأمان
            "reasons": [
                f"SMA Cross: {sma_10:.5f} vs {sma_20:.5f}",
                f"RSI: {rsi:.1f}",
                f"Price vs SMA10: {current_price:.5f} vs {sma_10:.5f}"
            ]
        }
    
    def confirm_trade(self, trade_data: Dict) -> Dict:
        """تأكيد تنفيذ صفقة"""
        try:
            trade_id = f"{trade_data['symbol']}_{datetime.now().timestamp()}"
            self.active_trades[trade_id] = {
                **trade_data,
                'timestamp': datetime.now(),
                'confirmed': True
            }
            
            logger.info(f"Trade confirmed: {trade_data['symbol']} {trade_data['action']}")
            
            return {
                "status": "confirmed",
                "trade_id": trade_id,
                "message": "Trade confirmed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error confirming trade: {e}")
            return {"status": "error", "message": str(e)}
    
    def report_trade_result(self, result_data: Dict) -> Dict:
        """استقبال نتيجة صفقة"""
        try:
            self.trade_results.append({
                **result_data,
                'received_at': datetime.now()
            })
            
            # تحليل بسيط للأداء
            profit = result_data.get('profit', 0)
            result_type = "ربح" if profit > 0 else "خسارة"
            
            # حساب معدل النجاح
            if len(self.trade_results) > 0:
                winning_trades = sum(1 for t in self.trade_results if t.get('profit', 0) > 0)
                win_rate = winning_trades / len(self.trade_results)
            else:
                win_rate = 0
            
            logger.info(f"Trade result: {result_data['symbol']} - {result_type} ${profit:.2f}")
            
            return {
                "status": "received",
                "analysis": {
                    "result": result_type,
                    "profit": profit,
                    "total_trades": len(self.trade_results),
                    "win_rate": round(win_rate * 100, 1)
                },
                "message": "Result processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error reporting result: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict:
        """حالة النظام"""
        return {
            "server_time": datetime.now().isoformat(),
            "active_trades": len(self.active_trades),
            "recent_signals": len(self.recent_signals),
            "total_results": len(self.trade_results),
            "models_loaded": self.models_loaded,
            "system_health": "healthy",
            "mode": "Linux Simple Mode"
        }


# إنشاء instance
bridge = SimpleBridgeServer()

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "Linux Compatible"
    })

@app.route('/test', methods=['GET', 'POST'])
def test():
    """Endpoint للاختبار وتشخيص المشاكل"""
    headers = dict(request.headers)
    data = request.get_data(as_text=True)
    
    response = {
        "method": request.method,
        "headers": headers,
        "data": data,
        "is_json": request.is_json,
        "content_type": request.content_type,
        "content_length": request.content_length,
        "remote_addr": request.remote_addr
    }
    
    logger.info(f"Test endpoint called: {response}")
    
    return jsonify(response)

@app.route('/echo', methods=['POST'])
def echo():
    """Echo endpoint - returns exactly what was sent"""
    raw_data = request.get_data(as_text=True)
    
    # Try to parse as JSON and echo back
    try:
        data = json.loads(raw_data)
        return jsonify({
            "received": data,
            "echo": data,
            "status": "ok"
        })
    except:
        return jsonify({
            "received_raw": raw_data,
            "status": "raw_data"
        })

@app.route('/api/test', methods=['POST'])
def api_test():
    """API test endpoint for EA connection testing"""
    try:
        data = request.get_json(force=True)
        api_key = data.get('api_key', '')
        
        return jsonify({
            "status": "success",
            "message": "Connection successful",
            "api_key_received": api_key,
            "timestamp": datetime.now().isoformat(),
            "server": "MT5 Bridge Server Linux"
        })
    except Exception as e:
        logger.error(f"API test error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical_data', methods=['POST'])
def receive_historical_data():
    """Receive historical data from EA"""
    try:
        data = request.get_json(force=True)
        
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        bars_data = data.get('data', [])
        
        if not symbol or not timeframe or not bars_data:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Save to database
        saved_count = save_historical_data(symbol, timeframe, bars_data)
        
        logger.info(f"Historical data: {symbol} {timeframe} - {len(bars_data)} bars, saved: {saved_count}")
        
        return jsonify({
            "status": "success",
            "received": len(bars_data),
            "saved": saved_count,
            "symbol": symbol,
            "timeframe": timeframe
        })
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_data', methods=['POST'])
def receive_live_data():
    """Receive live data updates from EA"""
    try:
        data = request.get_json(force=True)
        
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        bars_data = data.get('data', [])
        
        if not symbol or not timeframe or not bars_data:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Update price history
        key = f"{symbol}_{timeframe}"
        if hasattr(bridge, 'price_history'):
            bridge.price_history[key] = bars_data[-100:]  # Keep last 100 bars
        
        logger.info(f"Live data update: {symbol} {timeframe} - {len(bars_data)} bars")
        
        return jsonify({
            "status": "success",
            "received": len(bars_data),
            "symbol": symbol,
            "timeframe": timeframe
        })
        
    except Exception as e:
        logger.error(f"Live data error: {e}")
        return jsonify({"error": str(e)}), 500

def save_historical_data(symbol: str, timeframe: str, bars_data: list) -> int:
    """Save historical data to database"""
    try:
        import sqlite3
        from pathlib import Path
        
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                spread INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, time)
            )
        """)
        
        saved_count = 0
        for bar in bars_data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timeframe, time, open, high, low, close, volume, spread)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    timeframe,
                    int(bar['time']),
                    float(bar['open']),
                    float(bar['high']),
                    float(bar['low']),
                    float(bar['close']),
                    int(bar['volume']),
                    int(bar.get('spread', 0))
                ))
                saved_count += 1
            except Exception as e:
                logger.debug(f"Skip bar: {e}")
        
        conn.commit()
        conn.close()
        
        return saved_count
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return 0

@app.route('/get_signal', methods=['POST'])
def get_signal():
    """استقبال طلب الإشارة من MT5"""
    try:
        # قراءة البيانات الخام أولاً
        raw_data = request.get_data(as_text=True)
        logger.info(f"Raw data received: {raw_data}")
        
        data = None
        
        # محاولة parse كـ JSON
        if raw_data:
            try:
                # تنظيف البيانات من الأحرف الإضافية
                cleaned_data = raw_data.strip()
                # إزالة أي null bytes أو أحرف غير مرئية
                cleaned_data = cleaned_data.rstrip('\x00').rstrip()
                
                data = json.loads(cleaned_data)
                logger.info(f"Parsed JSON data: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                logger.warning(f"Raw data bytes: {raw_data.encode()}")
                
                # محاولة استخراج JSON من البيانات
                try:
                    # البحث عن بداية ونهاية JSON
                    start = raw_data.find('{')
                    end = raw_data.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = raw_data[start:end]
                        data = json.loads(json_str)
                        logger.info(f"Extracted JSON: {data}")
                except:
                    pass
        
        # إذا فشل كل شيء، جرب request.json
        if not data:
            try:
                data = request.get_json(force=True, silent=True)
            except:
                data = None
        
        # قيم افتراضية إذا فشل كل شيء
        if not data:
            logger.warning("Could not parse data, using defaults")
            data = {"symbol": "EURUSDm", "price": 1.1000}
        
        # استخراج البيانات
        symbol = str(data.get('symbol', 'EURUSDm'))
        price = float(data.get('price', 1.1000))
        
        logger.info(f"Processing signal for {symbol} at {price}")
        
        # توليد إشارة
        signal = bridge.get_signal(symbol, price)
        
        logger.info(f"Sending signal: {signal}")
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Error in get_signal: {str(e)}", exc_info=True)
        # Always return valid response
        return jsonify({
            "action": "NO_TRADE",
            "confidence": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "lot": 0.01,
            "error": str(e)
        }), 200

@app.route('/confirm_trade', methods=['POST'])
def confirm_trade():
    try:
        data = request.get_json()
        result = bridge.confirm_trade(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in confirm_trade: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/report_trade', methods=['POST'])
def report_trade():
    try:
        data = request.get_json()
        result = bridge.report_trade_result(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in report_trade: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    try:
        status = bridge.get_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_balance', methods=['POST'])
def update_balance():
    try:
        data = request.get_json()
        balance = data.get('balance')
        
        if balance:
            bridge.account_balance = balance
            logger.info(f"Balance updated: ${balance}")
            
        return jsonify({"status": "updated"})
        
    except Exception as e:
        logger.error(f"Error updating balance: {e}")
        return jsonify({"error": str(e)}), 500


def run_server(host='0.0.0.0', port=5000):
    """تشغيل الخادم"""
    logger.info(f"Starting Linux Bridge Server on {host}:{port}")
    logger.info("This is a simplified version that works without MT5")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
