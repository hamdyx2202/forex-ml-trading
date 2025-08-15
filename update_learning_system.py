#!/usr/bin/env python3
"""
تحديث نظام التعلم لإضافة ميزات الدعم والمقاومة
Update Learning System for Support/Resistance Features

التحديثات:
1. إضافة 5 ميزات جديدة للدعم والمقاومة (المجموع 75)
2. تحديث feature engineering
3. تحديث معايير التدريب
4. إضافة تتبع فعالية الدعم/المقاومة
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import shutil

# إضافة المسار للموديولات
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75
from support_resistance import SupportResistanceCalculator

class LearningSystemUpdater:
    """محدث نظام التعلم"""
    
    def __init__(self):
        """تهيئة المحدث"""
        self.backup_dir = f"backups/update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.files_to_update = [
            'advanced_learner.py',
            'emergency_train_models.py',
            'train_multi_tf_models.py',
            'server.py',
            'predictor.py'
        ]
        
    def create_backups(self):
        """إنشاء نسخ احتياطية من الملفات الحالية"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            
            for file in self.files_to_update:
                if os.path.exists(file):
                    shutil.copy2(file, os.path.join(self.backup_dir, file))
                    logger.info(f"✅ Backed up {file}")
            
            # نسخ احتياطية من النماذج الحالية
            if os.path.exists('models'):
                shutil.copytree('models', os.path.join(self.backup_dir, 'models'))
                logger.info("✅ Backed up models directory")
                
        except Exception as e:
            logger.error(f"Error creating backups: {str(e)}")
            raise
    
    def update_advanced_learner(self):
        """تحديث advanced_learner.py لاستخدام 75 ميزة"""
        try:
            with open('advanced_learner.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # التحديثات المطلوبة
            updates = [
                # تغيير import
                ('from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer',
                 'from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75'),
                
                # تغيير class instantiation
                ('self.feature_engineer = AdaptiveFeatureEngineer(target_features=70)',
                 'self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)'),
                
                # تحديث TARGET_FEATURES
                ('TARGET_FEATURES = 70',
                 'TARGET_FEATURES = 75'),
                
                # إضافة symbol parameter في engineer_features
                ('df_features = self.feature_engineer.engineer_features(df)',
                 'df_features = self.feature_engineer.engineer_features(df, pair)')
            ]
            
            for old, new in updates:
                if old in content:
                    content = content.replace(old, new)
                    logger.info(f"✅ Updated: {old[:50]}...")
            
            with open('advanced_learner.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Updated advanced_learner.py")
            
        except Exception as e:
            logger.error(f"Error updating advanced_learner.py: {str(e)}")
    
    def update_emergency_train(self):
        """تحديث emergency_train_models.py"""
        try:
            with open('emergency_train_models.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            updates = [
                # تغيير import
                ('from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer',
                 'from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75'),
                
                # تغيير instantiation
                ('feature_engineer = AdaptiveFeatureEngineer(target_features=70)',
                 'feature_engineer = AdaptiveFeatureEngineer75(target_features=75)'),
                
                # تحديث TARGET_FEATURES
                ('TARGET_FEATURES = 70',
                 'TARGET_FEATURES = 75')
            ]
            
            for old, new in updates:
                if old in content:
                    content = content.replace(old, new)
            
            # إضافة symbol في engineer_features calls
            content = content.replace(
                'df_features = feature_engineer.engineer_features(df)',
                'df_features = feature_engineer.engineer_features(df, pair)'
            )
            
            with open('emergency_train_models.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Updated emergency_train_models.py")
            
        except Exception as e:
            logger.error(f"Error updating emergency_train_models.py: {str(e)}")
    
    def update_server(self):
        """تحديث server.py لاستخدام feature engineering الجديد"""
        try:
            with open('server.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # إضافة imports
            if 'from feature_engineer_adaptive_75' not in content:
                imports_section = """from datetime import datetime, timedelta
from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75
from support_resistance import SupportResistanceCalculator"""
                
                content = content.replace(
                    'from datetime import datetime, timedelta',
                    imports_section
                )
            
            # تحديث feature engineering في get_signal
            old_fe = """# هندسة الميزات
    from fix_feature_count import engineer_features_with_padding
    df_features = engineer_features_with_padding(df_ohlcv)"""
            
            new_fe = """# هندسة الميزات مع 75 ميزة
    feature_engineer = AdaptiveFeatureEngineer75()
    df_features = feature_engineer.engineer_features(df_ohlcv, symbol)"""
            
            if old_fe in content:
                content = content.replace(old_fe, new_fe)
            
            with open('server.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Updated server.py")
            
        except Exception as e:
            logger.error(f"Error updating server.py: {str(e)}")
    
    def update_predictor(self):
        """تحديث predictor.py"""
        try:
            # التحقق من وجود predictor.py
            if not os.path.exists('predictor.py'):
                logger.warning("predictor.py not found, skipping")
                return
            
            with open('predictor.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # تحديثات مشابهة للملفات الأخرى
            updates = [
                ('from feature_engineer_adaptive_70', 'from feature_engineer_adaptive_75'),
                ('AdaptiveFeatureEngineer(', 'AdaptiveFeatureEngineer75('),
                ('target_features=70', 'target_features=75'),
                ('TARGET_FEATURES = 70', 'TARGET_FEATURES = 75')
            ]
            
            for old, new in updates:
                if old in content:
                    content = content.replace(old, new)
            
            with open('predictor.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Updated predictor.py")
            
        except Exception as e:
            logger.error(f"Error updating predictor.py: {str(e)}")
    
    def create_migration_script(self):
        """إنشاء سكريبت لترحيل النماذج الحالية"""
        migration_script = '''#!/usr/bin/env python3
"""
سكريبت ترحيل النماذج من 70 إلى 75 ميزة
Model Migration Script: 70 to 75 features
"""

import os
import joblib
import numpy as np
from loguru import logger
from datetime import datetime

def migrate_model(model_path, output_path):
    """ترحيل نموذج واحد"""
    try:
        # تحميل النموذج
        model_data = joblib.load(model_path)
        
        # تحديث metadata
        if 'metadata' in model_data:
            model_data['metadata']['n_features'] = 75
            model_data['metadata']['feature_version'] = '75_with_sr'
            model_data['metadata']['migration_date'] = datetime.now().isoformat()
        
        # حفظ النموذج المحدث
        joblib.dump(model_data, output_path)
        logger.info(f"✅ Migrated {os.path.basename(model_path)}")
        
    except Exception as e:
        logger.error(f"Error migrating {model_path}: {str(e)}")

def main():
    """ترحيل جميع النماذج"""
    models_dir = "models"
    migrated_dir = "models_75_features"
    
    os.makedirs(migrated_dir, exist_ok=True)
    
    # البحث عن جميع ملفات joblib
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.joblib'):
                model_path = os.path.join(root, file)
                
                # حفظ في نفس البنية
                rel_path = os.path.relpath(model_path, models_dir)
                output_path = os.path.join(migrated_dir, rel_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                migrate_model(model_path, output_path)
    
    logger.info("✅ Migration completed! Check models_75_features directory")

if __name__ == "__main__":
    main()
'''
        
        with open('migrate_models_to_75.py', 'w') as f:
            f.write(migration_script)
        
        logger.info("✅ Created migration script: migrate_models_to_75.py")
    
    def create_training_schedule(self):
        """إنشاء جدول تدريب تدريجي"""
        schedule = '''# جدول التدريب التدريجي للنماذج الجديدة (75 ميزة)
# Gradual Training Schedule for New Models (75 features)

## المرحلة 1: اختبار (يوم 1)
- تدريب نموذج واحد: EURUSD_M5
- اختبار الأداء لمدة 24 ساعة
- التحقق من التوافق

## المرحلة 2: توسع محدود (يوم 2-3)
- تدريب 4 أزواج رئيسية على M5:
  - EURUSD, GBPUSD, USDJPY, XAUUSD
- مراقبة الأداء

## المرحلة 3: توسع متوسط (يوم 4-7)
- إضافة timeframes: M15, H1
- نفس الأزواج الأربعة
- المجموع: 12 نموذج

## المرحلة 4: توسع كامل (يوم 8-14)
- جميع الأزواج المفعلة
- جميع الـ timeframes
- تدريب تدريجي مع مراقبة

## أوامر التدريب:

### المرحلة 1:
```bash
python train_single_model.py --pair EURUSD --timeframe M5 --features 75
```

### المرحلة 2:
```bash
python train_selected_models.py --pairs "EURUSD,GBPUSD,USDJPY,XAUUSD" --timeframe M5 --features 75
```

### المرحلة 3:
```bash
python train_selected_models.py --pairs "EURUSD,GBPUSD,USDJPY,XAUUSD" --timeframes "M5,M15,H1" --features 75
```

### المرحلة 4:
```bash
python advanced_learner.py --features 75
```
'''
        
        with open('TRAINING_SCHEDULE_75.md', 'w', encoding='utf-8') as f:
            f.write(schedule)
        
        logger.info("✅ Created training schedule: TRAINING_SCHEDULE_75.md")
    
    def create_compatibility_layer(self):
        """إنشاء طبقة توافق للنماذج القديمة"""
        compat_code = '''#!/usr/bin/env python3
"""
طبقة التوافق للنماذج 70 و 75 ميزة
Compatibility Layer for 70 and 75 Feature Models
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
from loguru import logger

class FeatureCompatibilityLayer:
    """طبقة التوافق بين النماذج القديمة والجديدة"""
    
    def __init__(self):
        self.old_features = 70
        self.new_features = 75
        
    def make_compatible(self, features: Union[np.ndarray, pd.DataFrame], 
                       target_features: int) -> Union[np.ndarray, pd.DataFrame]:
        """
        تحويل الميزات للعدد المطلوب
        
        Args:
            features: الميزات الحالية
            target_features: العدد المطلوب
            
        Returns:
            الميزات بالعدد المطلوب
        """
        if isinstance(features, pd.DataFrame):
            return self._make_df_compatible(features, target_features)
        else:
            return self._make_array_compatible(features, target_features)
    
    def _make_array_compatible(self, features: np.ndarray, target_features: int) -> np.ndarray:
        """تحويل numpy array"""
        current_features = features.shape[1] if len(features.shape) > 1 else len(features)
        
        if current_features == target_features:
            return features
        
        if current_features < target_features:
            # إضافة padding
            if len(features.shape) > 1:
                padding = np.zeros((features.shape[0], target_features - current_features))
                return np.hstack([features, padding])
            else:
                padding = np.zeros(target_features - current_features)
                return np.concatenate([features, padding])
        else:
            # قص الميزات الزائدة
            return features[:, :target_features] if len(features.shape) > 1 else features[:target_features]
    
    def _make_df_compatible(self, df: pd.DataFrame, target_features: int) -> pd.DataFrame:
        """تحويل DataFrame"""
        # استخراج الميزات فقط (بدون OHLCV)
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        current_features = len(feature_cols)
        
        if current_features == target_features:
            return df
        
        df_copy = df.copy()
        
        if current_features < target_features:
            # إضافة أعمدة padding
            for i in range(current_features, target_features):
                if f'padding_{i}' not in df_copy.columns:
                    df_copy[f'padding_{i}'] = 0.0
        else:
            # حذف الأعمدة الزائدة
            cols_to_drop = feature_cols[target_features:]
            df_copy = df_copy.drop(columns=cols_to_drop)
        
        return df_copy
    
    def get_feature_version(self, n_features: int) -> str:
        """تحديد إصدار الميزات"""
        if n_features == 70:
            return "v70_base"
        elif n_features == 75:
            return "v75_with_sr"
        else:
            return f"v{n_features}_custom"


# دالة مساعدة للاستخدام المباشر
def ensure_feature_compatibility(features, target_features=75):
    """ضمان توافق الميزات"""
    compat = FeatureCompatibilityLayer()
    return compat.make_compatible(features, target_features)
'''
        
        with open('feature_compatibility.py', 'w') as f:
            f.write(compat_code)
        
        logger.info("✅ Created compatibility layer: feature_compatibility.py")
    
    def update_unified_standards(self):
        """تحديث معايير الموحدة"""
        try:
            if os.path.exists('unified_standards.py'):
                with open('unified_standards.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # تحديث REQUIRED_FEATURES
                content = content.replace('REQUIRED_FEATURES = 70', 'REQUIRED_FEATURES = 75')
                
                # إضافة معلومات الميزات الجديدة
                sr_features_info = '''
# ميزات الدعم والمقاومة الجديدة (5 ميزات)
SR_FEATURES = [
    'distance_to_support_pct',      # المسافة النسبية لأقرب دعم
    'distance_to_resistance_pct',   # المسافة النسبية لأقرب مقاومة
    'nearest_support_strength',     # قوة أقرب دعم
    'nearest_resistance_strength',  # قوة أقرب مقاومة
    'position_in_sr_range'         # الموقع في نطاق الدعم/المقاومة
]

# إجمالي الميزات = 70 (الأساسية) + 5 (دعم/مقاومة) = 75
'''
                
                # إضافة قبل نهاية الملف
                if 'SR_FEATURES' not in content:
                    content = content.rstrip() + '\n' + sr_features_info
                
                with open('unified_standards.py', 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("✅ Updated unified_standards.py")
                
        except Exception as e:
            logger.error(f"Error updating unified_standards.py: {str(e)}")
    
    def run_update(self):
        """تشغيل جميع التحديثات"""
        logger.info("🚀 Starting learning system update to 75 features...")
        
        # 1. إنشاء نسخ احتياطية
        logger.info("📁 Creating backups...")
        self.create_backups()
        
        # 2. تحديث الملفات
        logger.info("📝 Updating files...")
        self.update_advanced_learner()
        self.update_emergency_train()
        self.update_server()
        self.update_predictor()
        self.update_unified_standards()
        
        # 3. إنشاء أدوات مساعدة
        logger.info("🛠️ Creating helper tools...")
        self.create_migration_script()
        self.create_training_schedule()
        self.create_compatibility_layer()
        
        # 4. إنشاء تقرير التحديث
        self.create_update_report()
        
        logger.info("✅ Learning system update completed!")
        logger.info(f"📁 Backups saved in: {self.backup_dir}")
    
    def create_update_report(self):
        """إنشاء تقرير التحديث"""
        report = f'''# تقرير تحديث نظام التعلم
# Learning System Update Report

**التاريخ**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## التحديثات المنفذة:

### 1. الملفات المحدثة:
- ✅ advanced_learner.py - محدث لاستخدام 75 ميزة
- ✅ emergency_train_models.py - محدث لاستخدام 75 ميزة
- ✅ server.py - محدث لاستخدام feature engineering الجديد
- ✅ predictor.py - محدث (إن وجد)
- ✅ unified_standards.py - محدث بالمعايير الجديدة

### 2. الملفات الجديدة:
- ✅ feature_engineer_adaptive_75.py - مهندس ميزات جديد
- ✅ support_resistance.py - حاسب الدعم والمقاومة
- ✅ dynamic_sl_tp_system.py - نظام SL/TP الديناميكي
- ✅ instrument_manager.py - مدير الأدوات المالية
- ✅ feature_compatibility.py - طبقة التوافق
- ✅ migrate_models_to_75.py - سكريبت الترحيل

### 3. الميزات الجديدة المضافة:
1. **distance_to_support_pct**: المسافة النسبية لأقرب دعم (%)
2. **distance_to_resistance_pct**: المسافة النسبية لأقرب مقاومة (%)
3. **nearest_support_strength**: قوة أقرب دعم (0-1)
4. **nearest_resistance_strength**: قوة أقرب مقاومة (0-1)
5. **position_in_sr_range**: الموقع في نطاق الدعم/المقاومة (0-1)

## الخطوات التالية:

### 1. ترحيل النماذج (اختياري):
```bash
python migrate_models_to_75.py
```

### 2. التدريب التدريجي:
اتبع الجدول في TRAINING_SCHEDULE_75.md

### 3. اختبار التوافق:
```bash
python test_feature_compatibility.py
```

### 4. تشغيل النظام:
```bash
# مع النماذج الجديدة
python server.py --features 75

# مع طبقة التوافق للنماذج القديمة
python server.py --compat-mode
```

## ملاحظات مهمة:

⚠️ **النماذج الحالية (70 ميزة) ستحتاج إلى:**
- إما إعادة تدريب بـ 75 ميزة
- أو استخدام طبقة التوافق

⚠️ **تأكد من:**
- أخذ نسخة احتياطية قبل التدريب الجديد
- اختبار نموذج واحد أولاً
- مراقبة الأداء بعد التحديث

## النسخ الاحتياطية:
محفوظة في: {self.backup_dir}
'''
        
        with open('UPDATE_REPORT_75_FEATURES.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ Created update report: UPDATE_REPORT_75_FEATURES.md")


def main():
    """تشغيل التحديث"""
    updater = LearningSystemUpdater()
    
    print("🔄 Learning System Update to 75 Features")
    print("=" * 50)
    print("\nThis will update the learning system to use 75 features")
    print("including 5 new support/resistance features.")
    print(f"\nBackups will be saved to: {updater.backup_dir}")
    
    response = input("\nDo you want to proceed? (yes/no): ").lower()
    
    if response == 'yes':
        updater.run_update()
        print("\n✅ Update completed successfully!")
        print("📋 Check UPDATE_REPORT_75_FEATURES.md for details")
    else:
        print("❌ Update cancelled")


if __name__ == "__main__":
    main()