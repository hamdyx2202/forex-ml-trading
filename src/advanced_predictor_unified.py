#!/usr/bin/env python3
"""
Advanced Predictor with Unified Features
المتنبئ المتقدم مع الميزات الموحدة
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
import glob
from loguru import logger
import sys

# إضافة المسار للوصول للملفات
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering_unified import UnifiedFeatureEngineer

class UnifiedAdvancedPredictor:
    """نظام التنبؤ المتقدم الموحد"""
    
    def __init__(self, models_dir: str = "models/unified"):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_info = {}
        self.models_dir = models_dir
        self.feature_engineer = UnifiedFeatureEngineer()
        self.load_latest_models()
        
    def load_latest_models(self):
        """تحميل أحدث النماذج الموحدة"""
        model_dir = Path(self.models_dir)
        if not model_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            # Try alternative paths
            alt_paths = [
                "models/advanced",
                "../models/unified",
                "../models/advanced"
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    model_dir = Path(alt_path)
                    logger.info(f"Using alternative model path: {alt_path}")
                    break
            else:
                logger.error("No models found in any location")
                return
            
        # البحث عن ملفات النماذج
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            logger.warning("No model files found")
            return
            
        # تحميل كل نموذج
        loaded_count = 0
        for model_file in model_files:
            try:
                # استخراج اسم النموذج
                filename = model_file.stem
                
                # تخطي ملفات non-model
                if 'config' in filename or 'summary' in filename:
                    continue
                
                # استخراج المفتاح (symbol_timeframe)
                # مثال: EURJPYm_PERIOD_H1_unified_v2 -> EURJPYm_PERIOD_H1
                parts = filename.split('_')
                if len(parts) >= 3:
                    # أخذ أول 3 أجزاء (Symbol_PERIOD_Timeframe)
                    key = '_'.join(parts[:3])
                else:
                    key = filename
                
                logger.info(f"Loading model: {key} from {filename}")
                
                # تحميل النموذج
                model_data = joblib.load(model_file)
                
                # التحقق من الإصدار الموحد
                if model_data.get('training_metadata', {}).get('unified_features', False):
                    self.models[key] = model_data['model']
                    self.scalers[key] = model_data['scaler']
                    self.metrics[key] = model_data.get('metrics', {})
                    self.feature_info[key] = {
                        'feature_names': model_data.get('feature_names', []),
                        'n_features': model_data.get('n_features', 0),
                        'feature_version': model_data.get('feature_version', 'unknown')
                    }
                    loaded_count += 1
                    logger.info(f"✅ Loaded {key}: {self.feature_info[key]['n_features']} features")
                else:
                    # محاولة تحميل النماذج القديمة مع تحذير
                    self.models[key] = model_data.get('model', model_data)
                    self.scalers[key] = model_data.get('scaler')
                    self.metrics[key] = model_data.get('metrics', {})
                    loaded_count += 1
                    logger.warning(f"⚠️ Loaded legacy model: {key}")
                    
            except Exception as e:
                logger.error(f"Error loading {model_file}: {e}")
                
        logger.info(f"✅ Loaded {loaded_count} models")
        
    def predict_with_confidence(self, symbol, timeframe, current_data, historical_data=None):
        """التنبؤ مع مستوى الثقة"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.models:
            logger.warning(f"No model found for {key}")
            logger.info(f"Available models: {list(self.models.keys())[:5]}...")
            return {
                'signal': 'NEUTRAL',
                'action': 'NO_TRADE',
                'confidence': 0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'message': f'No model available for {symbol} {timeframe}'
            }
            
        try:
            # تحضير البيانات
            if historical_data is not None and len(historical_data) > 0:
                # تحويل البيانات التاريخية إلى DataFrame
                df = pd.DataFrame(historical_data)
                
                # التأكد من وجود الأعمدة المطلوبة
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        logger.error(f"Missing required column: {col}")
                        return self._error_response(f"Missing data: {col}")
                
                # تحويل الوقت إلى datetime index
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('datetime', inplace=True)
                
                # إنشاء الميزات باستخدام الموحد
                X, feature_names = self.feature_engineer.create_features(df)
                
                # التحقق من عدد الميزات
                expected_features = self.feature_info.get(key, {}).get('n_features', 0)
                if expected_features > 0 and X.shape[1] != expected_features:
                    logger.error(
                        f"Feature mismatch for {key}: "
                        f"created {X.shape[1]}, expected {expected_features}"
                    )
                    
                    # محاولة المطابقة
                    if self.feature_info[key].get('feature_names'):
                        X = self._align_features(X, feature_names, key)
                    else:
                        return self._error_response(
                            f"Feature count mismatch: {X.shape[1]} vs {expected_features}"
                        )
                
                # أخذ آخر صف للتنبؤ
                X_current = X[-1:] if len(X) > 0 else X
                
            else:
                # استخدام البيانات الحالية فقط
                return self._error_response("Historical data required for prediction")
            
            # التحجيم
            if self.scalers[key] is not None:
                X_scaled = self.scalers[key].transform(X_current)
            else:
                X_scaled = X_current
                
            # التنبؤ
            model = self.models[key]
            prediction = model.predict(X_scaled)[0]
            
            # الحصول على الاحتماليات
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                prob_down = float(probabilities[0])
                prob_up = float(probabilities[1]) if len(probabilities) > 1 else 1 - prob_down
                confidence = float(max(probabilities))
            else:
                # للنماذج التي لا تدعم الاحتماليات
                prob_up = 1.0 if prediction == 1 else 0.0
                prob_down = 1.0 - prob_up
                confidence = 0.7  # ثقة افتراضية
                
            # تحديد الإشارة بناءً على الثقة
            if confidence < 0.6:
                signal = 'NEUTRAL'
                action = 'NO_TRADE'
            elif prediction == 1 and confidence >= 0.7:
                signal = 'STRONG_BUY' if confidence >= 0.85 else 'BUY'
                action = 'BUY'
            elif prediction == 0 and confidence >= 0.7:
                signal = 'STRONG_SELL' if confidence >= 0.85 else 'SELL'
                action = 'SELL'
            else:
                signal = 'NEUTRAL'
                action = 'NO_TRADE'
                
            # الحصول على مقاييس النموذج
            model_metrics = self.metrics.get(key, {})
            
            result = {
                'signal': signal,
                'action': action,
                'confidence': confidence,
                'probability_up': prob_up,
                'probability_down': prob_down,
                'model_accuracy': model_metrics.get('accuracy', 0),
                'high_conf_accuracy': model_metrics.get('high_confidence_accuracy', 0),
                'feature_count': X.shape[1] if 'X' in locals() else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # إضافة توصيات إدارة المخاطر
            if action in ['BUY', 'SELL']:
                result['risk_management'] = {
                    'position_size': 0.02 if confidence >= 0.85 else 0.01,
                    'stop_loss_pips': 20 if confidence >= 0.85 else 30,
                    'take_profit_pips': 40 if confidence >= 0.85 else 30,
                    'trailing_stop': confidence >= 0.8
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {key}: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._error_response(f"Prediction error: {str(e)}")
    
    def _align_features(self, X, current_names, model_key):
        """محاذاة الميزات مع النموذج المدرب"""
        expected_names = self.feature_info[model_key]['feature_names']
        
        # إنشاء DataFrame للمحاذاة
        df_current = pd.DataFrame(X, columns=current_names)
        aligned_data = []
        
        for feature in expected_names:
            if feature in df_current.columns:
                aligned_data.append(df_current[feature].values)
            else:
                # إضافة قيمة افتراضية للميزات المفقودة
                logger.warning(f"Missing feature: {feature}")
                aligned_data.append(np.zeros(X.shape[0]))
        
        return np.column_stack(aligned_data)
    
    def _error_response(self, message):
        """إرجاع استجابة خطأ موحدة"""
        return {
            'signal': 'ERROR',
            'action': 'NO_TRADE',
            'confidence': 0,
            'probability_up': 0.5,
            'probability_down': 0.5,
            'message': message,
            'timestamp': datetime.now().isoformat()
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
                'high_confidence_trades': metrics.get('high_confidence_trades', 0),
                'feature_count': self.feature_info.get(key, {}).get('n_features', 0),
                'feature_version': self.feature_info.get(key, {}).get('feature_version', 'unknown')
            }
            
        # حساب المتوسطات
        if performance:
            avg_accuracy = np.mean([p['accuracy'] for p in performance.values()])
            avg_high_conf = np.mean([p['high_confidence_accuracy'] for p in performance.values()])
            
            performance['overall'] = {
                'average_accuracy': avg_accuracy,
                'average_high_confidence_accuracy': avg_high_conf,
                'models_count': len(self.models),
                'unified_features': True
            }
            
        return performance

# للاستخدام المباشر
if __name__ == "__main__":
    # مثال للاستخدام
    predictor = UnifiedAdvancedPredictor()
    
    # عرض أداء النماذج
    performance = predictor.get_model_performance()
    
    print("="*60)
    print("🎯 Unified Models Performance")
    print("="*60)
    
    for key, metrics in performance.items():
        if key != 'overall':
            print(f"\n{key}:")
            print(f"  • Accuracy: {metrics['accuracy']:.2%}")
            print(f"  • High Confidence Accuracy: {metrics['high_confidence_accuracy']:.2%}")
            print(f"  • Feature Count: {metrics['feature_count']}")
            print(f"  • Feature Version: {metrics['feature_version']}")
            
    if 'overall' in performance:
        print(f"\n📊 Overall Performance:")
        print(f"  • Average Accuracy: {performance['overall']['average_accuracy']:.2%}")
        print(f"  • Average High-Conf Accuracy: {performance['overall']['average_high_confidence_accuracy']:.2%}")
        print(f"  • Total Models: {performance['overall']['models_count']}")
        print(f"  • Unified Features: {performance['overall']['unified_features']}")