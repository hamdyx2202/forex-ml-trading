#!/usr/bin/env python3
"""
Advanced MT5 Bridge Server - Uses Trained ML Models
خادم متقدم يستخدم النماذج المدربة
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib
from loguru import logger
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/advanced_server.log", rotation="1 day", retention="30 days")

# Import advanced predictor
sys.path.append(str(Path(__file__).parent.parent))
from src.advanced_predictor_95 import AdvancedPredictor
from feature_engineer_adaptive import AdaptiveFeatureEngineer as FeatureEngineer

app = Flask(__name__)
CORS(app)

class AdvancedMLServer:
    def __init__(self):
        """تهيئة الخادم المتقدم"""
        self.predictor = AdvancedPredictor()
        self.feature_engineer = FeatureEngineer(target_features=68)
        self.models_loaded = len(self.predictor.models) > 0
        
        logger.info(f"Advanced ML Server initialized")
        logger.info(f"Models loaded: {len(self.predictor.models)}")
        
        # عرض النماذج المحملة
        for model_key in self.predictor.models.keys():
            metrics = self.predictor.metrics.get(model_key, {})
            logger.info(f"  • {model_key}: Accuracy={metrics.get('accuracy', 0):.2%}")
    
    def process_signal_request(self, data):
        """معالجة طلب الإشارة"""
        try:
            # التحقق من البيانات المطلوبة
            if not data or 'symbol' not in data:
                return {
                    'action': 'ERROR',
                    'confidence': 0,
                    'reason': 'Missing symbol'
                }
            
            symbol = data.get('symbol', '')
            timeframe = data.get('timeframe', 'M5')
            
            # تحويل الإطار الزمني للتوافق مع النماذج
            timeframe_map = {
                'M5': 'PERIOD_M5',
                'M15': 'PERIOD_M15',
                'H1': 'PERIOD_H1',
                'H4': 'PERIOD_H4'
            }
            model_timeframe = timeframe_map.get(timeframe, 'PERIOD_M5')
            
            # التحقق من وجود بيانات الشموع
            if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                # لدينا بيانات شموع كاملة
                bars_data = data['data']
                
                if len(bars_data) < 50:
                    return {
                        'action': 'NO_TRADE',
                        'confidence': 0,
                        'reason': f'Not enough bars: {len(bars_data)}, need at least 50'
                    }
                
                # تحويل البيانات إلى DataFrame
                df = pd.DataFrame(bars_data)
                
                # التأكد من أن الأعمدة بالأسماء الصحيحة
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('datetime', inplace=True)
                
                # التأكد من أن البيانات numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # إضافة المؤشرات الفنية
                logger.info(f"Processing {len(df)} bars for {symbol} {timeframe}")
                
                try:
                    # إنشاء الميزات
                    df_features = self.feature_engineer.create_features(df.copy())
                    
                    if df_features.empty:
                        logger.warning("No features created")
                        return {
                            'action': 'NO_TRADE',
                            'confidence': 0,
                            'reason': 'Failed to create features'
                        }
                    
                    # التنبؤ باستخدام النموذج المناسب
                    model_key = f"{symbol}_{model_timeframe}"
                    logger.info(f"🔍 Model key: {model_key} (symbol={symbol}, timeframe={model_timeframe})")
                    
                    logger.info(f"🔍 Searching for model: {model_key}")
                    if model_key not in self.predictor.models:
                        logger.warning(f"No model found for {model_key}")
                        # محاولة بدون suffix
                        alt_key = f"{symbol}_{model_timeframe}"
                        if alt_key in self.predictor.models:
                            model_key = alt_key
                        else:
                            return {
                                'action': 'NO_TRADE',
                                'confidence': 0,
                                'reason': f'No model for {model_key}'
                            }
                    
                    # التنبؤ
                    result = self.predictor.predict_with_confidence(
                        symbol=symbol,
                        timeframe=model_timeframe,
                        current_data=None,
                        historical_data=bars_data
                    )
                    
                    logger.info(f"Prediction for {symbol}: {result}")
                    
                    # تحضير الرد
                    response = {
                        'signal': result.get('signal', 'NO_TRADE'),
                        'action': result.get('action', 'NO_TRADE'),
                        'confidence': float(result.get('confidence', 0)),
                        'probability_up': float(result.get('probability_up', 0.5)),
                        'probability_down': float(result.get('probability_down', 0.5)),
                        'model_accuracy': float(result.get('model_accuracy', 0)),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # إضافة توصيات إدارة المخاطر
                    if 'risk_management' in result:
                        response['risk_management'] = result['risk_management']
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Feature engineering error: {e}")
                    return {
                        'action': 'ERROR',
                        'confidence': 0,
                        'reason': f'Feature engineering failed: {str(e)}'
                    }
                    
            else:
                # لا توجد بيانات شموع
                return {
                    'action': 'NO_TRADE',
                    'confidence': 0,
                    'reason': 'No candle data provided'
                }
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {
                'action': 'ERROR',
                'confidence': 0,
                'reason': str(e)
            }

# إنشاء instance من الخادم
ml_server = AdvancedMLServer()

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخادم"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ml_server.models_loaded,
        'models_count': len(ml_server.predictor.models),
        'models': list(ml_server.predictor.models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/get_signal', methods=['POST'])
def get_signal():
    """استقبال طلب الإشارة من MT5"""
    try:
        # قراءة البيانات
        raw_data = request.get_data(as_text=True)
        logger.info(f"Received request from {request.remote_addr}")
        
        # تحليل JSON
        try:
            data = json.loads(raw_data.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return jsonify({
                'action': 'ERROR',
                'confidence': 0,
                'reason': 'Invalid JSON'
            }), 400
        
        # معالجة الطلب
        response = ml_server.process_signal_request(data)
        
        logger.info(f"Response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({
            'action': 'ERROR',
            'confidence': 0,
            'reason': str(e)
        }), 500

@app.route('/model_performance', methods=['GET'])
def model_performance():
    """عرض أداء النماذج"""
    try:
        performance = ml_server.predictor.get_model_performance()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_server(host='0.0.0.0', port=5000):
    """تشغيل الخادم"""
    logger.info("="*60)
    logger.info("🚀 Starting Advanced ML Bridge Server")
    logger.info(f"📡 Listening on {host}:{port}")
    logger.info(f"🧠 Models loaded: {len(ml_server.predictor.models)}")
    logger.info("="*60)
    
    if not ml_server.models_loaded:
        logger.warning("⚠️ No models loaded! Please run train_advanced_95_percent.py first")
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_server()