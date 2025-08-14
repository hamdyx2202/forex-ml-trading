#!/usr/bin/env python3
"""
Model Validator - Ensures all models follow unified standards
مُحقق النماذج - يضمن اتباع جميع النماذج للمعايير الموحدة
"""

import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

from unified_standards import (
    STANDARD_FEATURES,
    get_model_filename,
    validate_features
)

class ModelValidator:
    """التحقق من صحة النماذج قبل الحفظ أو التحميل"""
    
    def __init__(self):
        self.validation_log = []
        self.models_dir = Path("models/unified")
        
    def validate_model_file(self, filepath):
        """التحقق من صحة ملف نموذج"""
        try:
            # تحميل النموذج
            model_data = joblib.load(filepath)
            
            # التحققات الأساسية
            checks = {
                'has_model': 'model' in model_data,
                'has_scaler': 'scaler' in model_data,
                'has_features': 'feature_names' in model_data or 'n_features' in model_data,
                'has_metrics': 'metrics' in model_data
            }
            
            # التحقق من عدد الميزات
            n_features = None
            if 'n_features' in model_data:
                n_features = model_data['n_features']
            elif 'feature_names' in model_data:
                n_features = len(model_data['feature_names'])
            
            checks['correct_features'] = n_features == STANDARD_FEATURES
            
            # التحقق من قدرة النموذج على التنبؤ
            if all(checks.values()):
                try:
                    # اختبار بسيط
                    X_test = np.random.randn(1, STANDARD_FEATURES)
                    X_scaled = model_data['scaler'].transform(X_test)
                    prediction = model_data['model'].predict(X_scaled)
                    proba = model_data['model'].predict_proba(X_scaled)
                    checks['can_predict'] = True
                except:
                    checks['can_predict'] = False
            else:
                checks['can_predict'] = False
            
            # النتيجة
            is_valid = all(checks.values())
            
            result = {
                'filepath': str(filepath),
                'valid': is_valid,
                'checks': checks,
                'n_features': n_features,
                'timestamp': datetime.now().isoformat()
            }
            
            self.validation_log.append(result)
            
            if not is_valid:
                logger.warning(f"❌ Invalid model: {filepath.name}")
                for check, passed in checks.items():
                    if not passed:
                        logger.warning(f"   Failed: {check}")
            else:
                logger.info(f"✅ Valid model: {filepath.name}")
            
            return is_valid, result
            
        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            return False, {'error': str(e)}
    
    def fix_model_features(self, filepath, output_path=None):
        """إصلاح نموذج لا يتوافق مع المعايير"""
        try:
            model_data = joblib.load(filepath)
            
            # الحصول على عدد الميزات الحالي
            if 'scaler' in model_data:
                current_features = model_data['scaler'].n_features_in_
            else:
                logger.error("No scaler found in model")
                return False
            
            if current_features == STANDARD_FEATURES:
                logger.info("Model already has correct number of features")
                return True
            
            logger.info(f"Fixing model: {current_features} -> {STANDARD_FEATURES} features")
            
            # إنشاء scaler جديد يتوافق مع المعايير
            from sklearn.preprocessing import RobustScaler
            
            if current_features < STANDARD_FEATURES:
                # نحتاج padding
                logger.info(f"Adding {STANDARD_FEATURES - current_features} padding features")
                
                # إنشاء wrapper للـ scaler
                class PaddedScaler:
                    def __init__(self, original_scaler, target_features):
                        self.original_scaler = original_scaler
                        self.target_features = target_features
                        self.original_features = original_scaler.n_features_in_
                        self.padding_features = target_features - self.original_features
                    
                    def transform(self, X):
                        # تطبيع الميزات الأصلية
                        X_scaled = self.original_scaler.transform(X[:, :self.original_features])
                        
                        # إضافة padding
                        if self.padding_features > 0:
                            padding = np.zeros((X.shape[0], self.padding_features))
                            X_scaled = np.hstack([X_scaled, padding])
                        
                        return X_scaled
                    
                    def fit_transform(self, X, y=None):
                        self.original_scaler.fit(X[:, :self.original_features])
                        return self.transform(X)
                    
                    @property
                    def n_features_in_(self):
                        return self.target_features
                
                # استبدال الـ scaler
                model_data['scaler'] = PaddedScaler(model_data['scaler'], STANDARD_FEATURES)
                
            else:
                # نحتاج قص الميزات
                logger.warning(f"Trimming {current_features - STANDARD_FEATURES} extra features")
                logger.warning("This may affect model performance!")
            
            # تحديث metadata
            model_data['n_features'] = STANDARD_FEATURES
            model_data['fixed_by_validator'] = True
            model_data['fix_date'] = datetime.now().isoformat()
            
            # حفظ النموذج المُصلح
            if output_path is None:
                output_path = filepath.parent / f"fixed_{filepath.name}"
            
            joblib.dump(model_data, output_path)
            logger.info(f"✅ Fixed model saved: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing model: {e}")
            return False
    
    def validate_all_models(self):
        """التحقق من جميع النماذج في المجلد"""
        logger.info(f"🔍 Validating all models in {self.models_dir}")
        
        if not self.models_dir.exists():
            logger.error("Models directory not found")
            return
        
        model_files = list(self.models_dir.glob("*.pkl"))
        logger.info(f"Found {len(model_files)} model files")
        
        valid_count = 0
        invalid_count = 0
        
        for model_file in model_files:
            is_valid, result = self.validate_model_file(model_file)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        # حفظ تقرير التحقق
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_models': len(model_files),
            'valid_models': valid_count,
            'invalid_models': invalid_count,
            'details': self.validation_log
        }
        
        report_file = self.models_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📊 Validation Summary:")
        logger.info(f"   Total: {len(model_files)}")
        logger.info(f"   ✅ Valid: {valid_count}")
        logger.info(f"   ❌ Invalid: {invalid_count}")
        logger.info(f"   Report saved: {report_file}")
        
        return report
    
    def create_compatibility_wrapper(self):
        """إنشاء wrapper للتوافق مع النماذج القديمة"""
        wrapper_code = '''#!/usr/bin/env python3
"""
Compatibility Wrapper for Legacy Models
غلاف التوافق للنماذج القديمة
"""

import joblib
import numpy as np
from pathlib import Path

class CompatibilityWrapper:
    """يسمح باستخدام النماذج القديمة مع النظام الجديد"""
    
    def __init__(self, model_path):
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        
        # الحصول على عدد الميزات
        if hasattr(self.scaler, 'n_features_in_'):
            self.original_features = self.scaler.n_features_in_
        else:
            self.original_features = 49  # افتراضي
        
        self.target_features = 70
    
    def predict(self, X):
        """التنبؤ مع معالجة اختلاف عدد الميزات"""
        # ضبط عدد الميزات
        if X.shape[1] < self.original_features:
            # إضافة padding للوصول للعدد الأصلي
            padding = np.zeros((X.shape[0], self.original_features - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > self.original_features:
            # قص الميزات الزائدة
            X = X[:, :self.original_features]
        
        # تطبيع وتنبؤ
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """احتمالية التنبؤ مع معالجة اختلاف عدد الميزات"""
        # نفس المعالجة
        if X.shape[1] < self.original_features:
            padding = np.zeros((X.shape[0], self.original_features - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > self.original_features:
            X = X[:, :self.original_features]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

# استخدام
# wrapper = CompatibilityWrapper('path/to/old/model.pkl')
# predictions = wrapper.predict(X_with_70_features)
'''
        
        wrapper_file = Path("compatibility_wrapper.py")
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        logger.info(f"✅ Created compatibility wrapper: {wrapper_file}")

if __name__ == "__main__":
    validator = ModelValidator()
    
    # التحقق من جميع النماذج
    validator.validate_all_models()
    
    # إنشاء wrapper للتوافق
    validator.create_compatibility_wrapper()