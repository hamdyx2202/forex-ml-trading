#!/usr/bin/env python3
"""
MT5 Prediction Server - نسخة مبسطة للاختبار
Simple version for testing without full dependencies
"""

from flask import Flask, request, jsonify
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': 0,
        'version': '2.0-simple',
        'message': 'Server is running in simple test mode'
    })

@app.route('/api/predict_advanced', methods=['POST'])
def predict_advanced():
    """نقطة نهاية التنبؤ - نسخة اختبارية"""
    try:
        data = request.get_json()
        
        # استخراج البيانات
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        candles = data.get('candles', [])
        
        print(f"📊 Received prediction request: {symbol} {timeframe}")
        print(f"📈 Number of candles: {len(candles)}")
        
        # إرسال تنبؤ وهمي للاختبار
        current_price = float(candles[-1]['close']) if candles else 1.0850
        
        # تنبؤ اختباري
        predictions = {
            'scalping': {
                'signal': 2,  # Buy signal
                'confidence': 0.85,
                'stop_loss': current_price - 0.0020,
                'take_profit_1': current_price + 0.0020,
                'take_profit_2': current_price + 0.0040,
                'take_profit_3': current_price + 0.0060
            }
        }
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'message': 'Test prediction - not from real model'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/update_trade_result', methods=['POST'])
def update_trade_result():
    """تحديث نتيجة الصفقة - نسخة اختبارية"""
    try:
        data = request.get_json()
        print(f"✅ Received trade result: Ticket #{data.get('ticket')}")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_performance', methods=['GET'])
def get_performance():
    """الحصول على الأداء - نسخة اختبارية"""
    return jsonify({
        'overall': {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'avg_pips': 0,
            'best_trade': 0,
            'worst_trade': 0
        },
        'by_strategy': [],
        'message': 'Test performance data'
    })

if __name__ == '__main__':
    print("🚀 Starting MT5 Prediction Server (Simple Test Mode)")
    print("📊 Server URL: http://localhost:5000")
    print("⚠️  This is a test server without real ML models")
    print("\n✅ Available endpoints:")
    print("   - GET  /api/health")
    print("   - POST /api/predict_advanced")
    print("   - POST /api/update_trade_result")
    print("   - GET  /api/get_performance")
    print("\n🔍 Press Ctrl+C to stop the server\n")
    
    # إنشاء مجلد السجلات
    Path("logs").mkdir(exist_ok=True)
    
    # تشغيل الخادم
    app.run(host='0.0.0.0', port=5000, debug=True)