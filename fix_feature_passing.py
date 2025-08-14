#!/usr/bin/env python3
"""
Fix Feature Passing to Predictor
إصلاح تمرير الميزات للمُتنبئ
"""

import os
import sys

print("🔧 Fixing feature passing issue...")
print("="*60)

print("\n📊 المشكلة:")
print("- Feature engineering ينتج 70 ميزة (49 + 21 padding)")
print("- لكن predictor يستقبل البيانات الخام بدلاً من df_features")
print("- RobustScaler يشتكي من 49 ميزة بدلاً من 70")

# إنشاء نسخة محدثة من advanced_predictor_95.py
print("\n📝 Creating fixed predictor...")

fixed_predictor = '''#!/usr/bin/env python3
"""
Fixed Advanced Predictor for 95% Accuracy
مُتنبئ متقدم محدث لدقة 95%
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import os
import sys
from loguru import logger

class AdvancedPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_names = {}
        self.load_latest_models()
        
    def load_latest_models(self):
        """تحميل أحدث النماذج المتقدمة"""
        model_dir = Path("models/advanced")
        if not model_dir.exists():
            print("⚠️ No advanced models found. Please run train_advanced_95_percent.py first")
            return
            
        # البحث عن أحدث ملف تدريب
        model_files = list(model_dir.glob("*_ensemble_*.pkl"))
        if not model_files:
            print("⚠️ No ensemble models found")
            return
            
        # تحميل كل نموذج
        loaded_count = 0
        for model_file in model_files:
            try:
                # استخراج اسم الزوج والإطار الزمني
                filename = model_file.stem
                
                # Extract model key
                if '_ensemble_' in filename:
                    key = filename.split('_ensemble_')[0]
                else:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        key = '_'.join(parts[:3])
                    else:
                        key = filename
                
                # تحميل النموذج
                model_data = joblib.load(model_file)
                self.models[key] = model_data['model']
                self.scalers[key] = model_data['scaler']
                self.metrics[key] = model_data.get('metrics', {})
                self.feature_names[key] = model_data.get('feature_names', [])
                
                loaded_count += 1
                print(f"✅ Loaded model: {key}")
                    
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
                
        print(f"✅ Loaded {loaded_count} advanced models")
        print(f"📊 Available models: {list(self.models.keys())}")
        
    def predict_with_features(self, symbol, timeframe, features_df):
        """التنبؤ باستخدام DataFrame الميزات المُعدة"""
        key = f"{symbol}_{timeframe}"
        
        logger.info(f"🔮 Predicting with key: {key}")
        logger.info(f"📊 Features shape: {features_df.shape}")
        
        if key not in self.models:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'message': f'No model available for {symbol} {timeframe}'
            }
            
        try:
            model = self.models[key]
            scaler = self.scalers[key]
            
            # استخدام آخر صف من الميزات
            if isinstance(features_df, pd.DataFrame):
                # إزالة الأعمدة غير المطلوبة
                feature_cols = [col for col in features_df.columns 
                               if col not in ['target', 'target_binary', 'target_3class', 
                                            'future_return', 'time', 'open', 'high', 
                                            'low', 'close', 'volume', 'spread', 'datetime']]
                
                X = features_df[feature_cols].iloc[-1:].values
                logger.info(f"🎯 Using {len(feature_cols)} features")
                logger.info(f"🎯 Feature shape for prediction: {X.shape}")
            else:
                X = features_df
                
            # التطبيع
            X_scaled = scaler.transform(X)
            
            # التنبؤ
            y_pred = model.predict(X_scaled)[0]
            y_proba = model.predict_proba(X_scaled)[0]
            
            # تحديد الإشارة
            signal = 'BUY' if y_pred == 1 else 'SELL'
            confidence = float(max(y_proba))
            
            # الحصول على دقة النموذج
            model_accuracy = self.metrics.get(key, {}).get('accuracy', 0.75)
            
            return {
                'signal': signal,
                'action': signal,
                'confidence': confidence,
                'probability_up': float(y_proba[1] if len(y_proba) > 1 else 0.5),
                'probability_down': float(y_proba[0] if len(y_proba) > 0 else 0.5),
                'model_accuracy': model_accuracy,
                'features_used': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'message': f'Prediction error: {str(e)}'
            }
    
    def predict_with_confidence(self, symbol, timeframe, current_data, historical_data=None):
        """التنبؤ مع مستوى الثقة - للتوافق مع الكود القديم"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.models:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'message': f'No model available for {symbol} {timeframe}'
            }
            
        try:
            # هذه الدالة للتوافق فقط
            # يجب استخدام predict_with_features مباشرة
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'message': 'Please use predict_with_features instead'
            }
            
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'message': f'Prediction error: {str(e)}'
            }
'''

with open('src/advanced_predictor_fixed.py', 'w') as f:
    f.write(fixed_predictor)

print("✅ Created src/advanced_predictor_fixed.py")

# إنشاء خادم محدث
print("\n🔧 Creating updated server...")

updated_server = '''#!/usr/bin/env python3
"""
Updated Server with Fixed Feature Passing
خادم محدث مع إصلاح تمرير الميزات
"""

import sys
import os
from pathlib import Path

# Replace predictor import
sys.path.append(str(Path(__file__).parent.parent))

# استيراد النسخة المحدثة
from src.advanced_predictor_fixed import AdvancedPredictor

# استيراد الخادم الأصلي
from src.mt5_bridge_server_advanced import app, logger
import src.mt5_bridge_server_advanced as original_server

# استبدال predictor
original_server.ml_server.predictor = AdvancedPredictor()

# تحديث دالة process_signal_request
original_process = original_server.ml_server.process_signal_request

def new_process_signal_request(self, data):
    """معالجة طلب الإشارة - محدث"""
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
        
        # تحويل الإطار الزمني
        timeframe_map = {
            'M5': 'PERIOD_M5',
            'M15': 'PERIOD_M15',
            'H1': 'PERIOD_H1',
            'H4': 'PERIOD_H4'
        }
        model_timeframe = timeframe_map.get(timeframe, 'PERIOD_M5')
        
        # التحقق من وجود بيانات الشموع
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            bars_data = data['data']
            
            # تحضير البيانات
            import pandas as pd
            df = pd.DataFrame(bars_data)
            
            # تحويل الوقت
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df = df.sort_values('datetime')
                df.set_index('datetime', inplace=True)
            
            # إنشاء الميزات
            df_features = self.feature_engineer.create_features(df.copy())
            
            if df_features.empty:
                logger.warning("No features created")
                return {
                    'action': 'NO_TRADE',
                    'confidence': 0,
                    'reason': 'Failed to create features'
                }
            
            logger.info(f"📊 Features created: {df_features.shape}")
            
            # التحقق من النموذج
            model_key = f"{symbol}_{model_timeframe}"
            if model_key not in self.predictor.models:
                return {
                    'action': 'NO_TRADE',
                    'confidence': 0,
                    'reason': f'No model for {model_key}'
                }
            
            # التنبؤ باستخدام الميزات المُعدة
            result = self.predictor.predict_with_features(
                symbol=symbol,
                timeframe=model_timeframe,
                features_df=df_features
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
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return response
            
        else:
            return {
                'action': 'NO_TRADE',
                'confidence': 0,
                'reason': 'No bar data provided'
            }
            
    except Exception as e:
        logger.error(f"Error processing signal: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'action': 'ERROR',
            'confidence': 0,
            'reason': str(e)
        }

# استبدال الدالة
original_server.ml_server.process_signal_request = lambda data: new_process_signal_request(original_server.ml_server, data)

print("🚀 Starting server with fixed feature passing...")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('server_fixed_features.py', 'w') as f:
    f.write(updated_server)

print("✅ Created server_fixed_features.py")

print("\n" + "="*60)
print("✅ Fix created!")
print("\n🚀 الحل:")
print("1. انسخ الملفات إلى VPS:")
print("   scp src/advanced_predictor_fixed.py server_fixed_features.py root@69.62.121.53:/home/forex-ml-trading/")
print("\n2. على VPS:")
print("   cd /home/forex-ml-trading")
print("   source venv_pro/bin/activate")
print("   python server_fixed_features.py &")
print("\n📊 هذا الإصلاح:")
print("- يمرر df_features (مع 70 ميزة) إلى predictor")
print("- يستخدم predict_with_features بدلاً من predict_with_confidence")
print("- يتعامل مع padding features بشكل صحيح")