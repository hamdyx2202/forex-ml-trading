#!/usr/bin/env python3
"""
Data Sync Server for Linux
يستقبل البيانات من EA ويحفظها في قاعدة البيانات
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import os
from datetime import datetime
from loguru import logger
from pathlib import Path
import hashlib
import hmac

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.getenv("FOREXTML_API_KEY", "your_secure_api_key")
DB_PATH = "data/forex_ml.db"
LOG_PATH = "logs/data_sync_server.log"

# Setup logging
logger.add(LOG_PATH, rotation="1 day", retention="30 days")

def verify_api_key(provided_key):
    """التحقق من مفتاح API"""
    return provided_key == API_KEY

def ensure_database():
    """التأكد من وجود قاعدة البيانات"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # جدول البيانات الرئيسي
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
    
    # جدول سجل المزامنة
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            bars_count INTEGER NOT NULL,
            sync_type TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    
    # فهرس للأداء
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_price_data_symbol_tf_time 
        ON price_data(symbol, timeframe, time DESC)
    """)
    
    conn.commit()
    conn.close()

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        "status": "healthy",
        "service": "ForexML Data Sync Server",
        "timestamp": datetime.now().isoformat(),
        "mode": "Data Sync Mode"
    })

@app.route('/', methods=['GET'])
def index():
    """الصفحة الرئيسية"""
    return jsonify({
        "service": "ForexML Data Sync Server",
        "status": "running",
        "endpoints": [
            "/health",
            "/api/test",
            "/api/historical_data",
            "/api/live_data",
            "/api/stats"
        ]
    })

@app.route('/api/test', methods=['POST'])
def test_connection():
    """اختبار الاتصال"""
    try:
        data = request.get_json()
        
        if not data or not verify_api_key(data.get('api_key')):
            return jsonify({"error": "Unauthorized"}), 401
            
        return jsonify({
            "status": "success",
            "message": "Connection successful",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Test connection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical_data', methods=['POST'])
def receive_historical_data():
    """استقبال البيانات التاريخية"""
    try:
        data = request.get_json()
        
        # التحقق من المصادقة
        if not data or not verify_api_key(data.get('api_key')):
            return jsonify({"error": "Unauthorized"}), 401
        
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        bars = data.get('data', [])
        
        if not symbol or not timeframe or not bars:
            return jsonify({"error": "Missing required fields"}), 400
            
        # حفظ البيانات
        saved_count = save_price_data(symbol, timeframe, bars)
        
        # تسجيل المزامنة
        log_sync(symbol, timeframe, len(bars), 'historical', 'success')
        
        logger.info(f"Historical data received: {symbol} {timeframe} - {len(bars)} bars, saved: {saved_count}")
        
        return jsonify({
            "status": "success",
            "received": len(bars),
            "saved": saved_count,
            "symbol": symbol,
            "timeframe": timeframe
        })
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_data', methods=['POST'])
def receive_live_data():
    """استقبال البيانات الحية"""
    try:
        data = request.get_json()
        
        # التحقق من المصادقة
        if not data or not verify_api_key(data.get('api_key')):
            return jsonify({"error": "Unauthorized"}), 401
        
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        bars = data.get('data', [])
        
        if not symbol or not timeframe or not bars:
            return jsonify({"error": "Missing required fields"}), 400
            
        # حفظ البيانات
        saved_count = save_price_data(symbol, timeframe, bars)
        
        # تسجيل المزامنة
        log_sync(symbol, timeframe, len(bars), 'live', 'success')
        
        # تشغيل التحليل إذا كان هناك بيانات جديدة
        if saved_count > 0:
            trigger_analysis(symbol, timeframe)
        
        return jsonify({
            "status": "success",
            "received": len(bars),
            "saved": saved_count,
            "symbol": symbol,
            "timeframe": timeframe
        })
        
    except Exception as e:
        logger.error(f"Live data error: {e}")
        return jsonify({"error": str(e)}), 500

def save_price_data(symbol: str, timeframe: str, bars: list) -> int:
    """حفظ البيانات في قاعدة البيانات"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    saved_count = 0
    
    try:
        for bar in bars:
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
        
    finally:
        conn.close()
        
    return saved_count

def log_sync(symbol: str, timeframe: str, bars_count: int, sync_type: str, status: str):
    """تسجيل عملية المزامنة"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO sync_log (symbol, timeframe, bars_count, sync_type, status)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, timeframe, bars_count, sync_type, status))
        conn.commit()
    finally:
        conn.close()

def trigger_analysis(symbol: str, timeframe: str):
    """تشغيل التحليل عند وصول بيانات جديدة"""
    # يمكن إضافة منطق لتشغيل التحليل هنا
    logger.debug(f"New data available for analysis: {symbol} {timeframe}")

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """الحصول على إحصائيات قاعدة البيانات"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # إحصائيات البيانات
        cursor.execute("""
            SELECT symbol, timeframe, 
                   COUNT(*) as bars_count,
                   MIN(time) as start_time,
                   MAX(time) as end_time
            FROM price_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        
        data_stats = []
        for row in cursor.fetchall():
            data_stats.append({
                "symbol": row[0],
                "timeframe": row[1],
                "bars_count": row[2],
                "start_date": datetime.fromtimestamp(row[3]).isoformat() if row[3] else None,
                "end_date": datetime.fromtimestamp(row[4]).isoformat() if row[4] else None
            })
        
        # إحصائيات المزامنة
        cursor.execute("""
            SELECT COUNT(*) as total_syncs,
                   SUM(bars_count) as total_bars,
                   MAX(sync_time) as last_sync
            FROM sync_log
            WHERE status = 'success'
        """)
        
        sync_stats = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            "data_statistics": data_stats,
            "sync_statistics": {
                "total_syncs": sync_stats[0],
                "total_bars": sync_stats[1],
                "last_sync": sync_stats[2]
            }
        })
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/<symbol>/<timeframe>', methods=['GET'])
def get_data(symbol, timeframe):
    """الحصول على البيانات من قاعدة البيانات"""
    try:
        limit = request.args.get('limit', 1000, type=int)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time DESC
            LIMIT ?
        """, (symbol, timeframe, limit))
        
        data = []
        for row in cursor.fetchall():
            data.append({
                "time": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "spread": row[6]
            })
        
        conn.close()
        
        return jsonify({
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(data),
            "data": data
        })
        
    except Exception as e:
        logger.error(f"Get data error: {e}")
        return jsonify({"error": str(e)}), 500

def run_server(host='0.0.0.0', port=5000):
    """تشغيل الخادم"""
    ensure_database()
    logger.info(f"Starting Data Sync Server on {host}:{port}")
    logger.info(f"Database: {DB_PATH}")
    logger.info("Ready to receive data from MT5...")
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_server()