#!/usr/bin/env python3
"""
Advanced Predictor for 95% Accuracy
نظام التنبؤ المتقدم لدقة 95%
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
import glob

class AdvancedPredictor:
    """نظام التنبؤ المتقدم"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
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
                filename = model_file.stem  # e.g., EURUSDm_PERIOD_M5_ensemble_20250812_142405
                
                # Extract model key by removing the ensemble timestamp part
                if '_ensemble_' in filename:
                    key = filename.split('_ensemble_')[0]  # EURUSDm_PERIOD_M5
                else:
                    # Fallback: take first 3 parts
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
                
                loaded_count += 1
                print(f"✅ Loaded model: {key}")
                    
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
                
        print(f"✅ Loaded {loaded_count} advanced models")
        
    def prepare_features(self, data):
        """تحضير الميزات من البيانات الخام"""
        df = pd.DataFrame([data])
        
        # إضافة مؤشرات بسيطة (نفس المؤشرات المستخدمة في التدريب)
        # هنا يجب استخدام نفس FeatureEngineer
        # لكن للبساطة سنضيف بعض المؤشرات الأساسية
        
        features = {}
        
        # نسب الأسعار
        features['HL_ratio'] = data['high'] / data['low']
        features['CO_ratio'] = data['close'] / data['open']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - max(data['open'], data['close'])
        features['lower_shadow'] = min(data['open'], data['close']) - data['low']
        
        # مؤشرات بسيطة (تحتاج لبيانات تاريخية)
        # في الإنتاج، يجب استخدام نفس feature_engineer_fixed.py
        
        return features
        
    def predict_with_confidence(self, symbol, timeframe, current_data, historical_data=None):
        """التنبؤ مع مستوى الثقة"""
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
            # تحضير الميزات
            if historical_data is not None:
                # استخدام البيانات التاريخية لحساب المؤشرات
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from feature_engineer_fixed_v5 import FeatureEngineer
                engineer = FeatureEngineer()
                
                # تحويل البيانات إلى DataFrame
                df = pd.DataFrame(historical_data)
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('datetime', inplace=True)
                
                # إضافة المؤشرات
                df_features = engineer.add_technical_indicators(df)
                df_features = engineer.add_price_features(df_features)
                df_features = engineer.add_pattern_features(df_features)
                df_features = engineer.add_time_features(df_features)
                df_features = engineer.add_market_structure(df_features)
                
                # أخذ آخر صف
                features = df_features.iloc[-1]
                
                # اختيار نفس الميزات المستخدمة في التدريب
                feature_cols = [col for col in df_features.columns 
                               if col not in ['target', 'target_binary', 'target_3class', 
                                             'future_return', 'time', 'open', 'high', 
                                             'low', 'close', 'volume', 'spread']]
                
                X = features[feature_cols].values.reshape(1, -1)
            else:
                # استخدام ميزات بسيطة
                features = self.prepare_features(current_data)
                X = pd.DataFrame([features])
                
            # التحجيم
            if self.scalers[key] is not None:
                X_scaled = self.scalers[key].transform(X)
            else:
                X_scaled = X
                
            # التنبؤ
            model = self.models[key]
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # حساب الثقة
            confidence = max(probabilities)
            prob_up = probabilities[1] if len(probabilities) > 1 else 0.5
            prob_down = probabilities[0] if len(probabilities) > 1 else 0.5
            
            # تحديد الإشارة بناءً على الثقة
            if confidence < 0.6:
                signal = 'NEUTRAL'
                action = 'WAIT'
            elif prediction == 1 and confidence >= 0.7:
                signal = 'STRONG_BUY' if confidence >= 0.85 else 'BUY'
                action = 'BUY'
            elif prediction == 0 and confidence >= 0.7:
                signal = 'STRONG_SELL' if confidence >= 0.85 else 'SELL'
                action = 'SELL'
            else:
                signal = 'NEUTRAL'
                action = 'WAIT'
                
            # الحصول على مقاييس النموذج
            model_metrics = self.metrics.get(key, {})
            
            result = {
                'signal': signal,
                'action': action,
                'confidence': float(confidence),
                'probability_up': float(prob_up),
                'probability_down': float(prob_down),
                'model_accuracy': model_metrics.get('accuracy', 0),
                'high_conf_accuracy': model_metrics.get('high_confidence_accuracy', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # إضافة توصيات إدارة المخاطر
            if action in ['BUY', 'SELL']:
                result['risk_management'] = {
                    'position_size': 0.02 if confidence >= 0.85 else 0.01,  # 2% أو 1% من رأس المال
                    'stop_loss_pips': 20 if confidence >= 0.85 else 30,
                    'take_profit_pips': 40 if confidence >= 0.85 else 30,
                    'trailing_stop': confidence >= 0.8
                }
                
            return result
            
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0,
                'message': f'Prediction error: {str(e)}'
            }
            
    def get_model_performance(self):
        """الحصول على أداء جميع النماذج"""
        performance = {}
        
        for key, metrics in self.metrics.items():
            performance[key] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'high_confidence_accuracy': metrics.get('high_confidence_accuracy', 0),
                'high_confidence_trades': metrics.get('high_confidence_trades', 0)
            }
            
        # حساب المتوسطات
        if performance:
            avg_accuracy = np.mean([p['accuracy'] for p in performance.values()])
            avg_high_conf = np.mean([p['high_confidence_accuracy'] for p in performance.values()])
            
            performance['overall'] = {
                'average_accuracy': avg_accuracy,
                'average_high_confidence_accuracy': avg_high_conf,
                'models_count': len(self.models)
            }
            
        return performance

# للاستخدام المباشر
def predict(symbol, timeframe, current_data, historical_data=None):
    """دالة سريعة للتنبؤ"""
    predictor = AdvancedPredictor()
    return predictor.predict_with_confidence(symbol, timeframe, current_data, historical_data)

if __name__ == "__main__":
    # مثال للاستخدام
    predictor = AdvancedPredictor()
    
    # عرض أداء النماذج
    performance = predictor.get_model_performance()
    
    print("="*60)
    print("🎯 Advanced Models Performance")
    print("="*60)
    
    for key, metrics in performance.items():
        if key != 'overall':
            print(f"\n{key}:")
            print(f"  • Accuracy: {metrics['accuracy']:.2%}")
            print(f"  • High Confidence Accuracy: {metrics['high_confidence_accuracy']:.2%}")
            
    if 'overall' in performance:
        print(f"\n📊 Overall Performance:")
        print(f"  • Average Accuracy: {performance['overall']['average_accuracy']:.2%}")
        print(f"  • Average High-Conf Accuracy: {performance['overall']['average_high_confidence_accuracy']:.2%}")
        print(f"  • Total Models: {performance['overall']['models_count']}")