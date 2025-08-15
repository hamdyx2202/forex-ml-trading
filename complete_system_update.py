#!/usr/bin/env python3
"""
تحديث شامل للنظام - إضافة ميزات الدعم والمقاومة وجميع الأزواج
Complete System Update - Add S/R Features and All Instruments
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
from loguru import logger

# إعداد logging
logger.add("system_update_{time}.log", rotation="500 MB")

class CompleteSystemUpdater:
    """محدث النظام الشامل"""
    
    def __init__(self):
        self.backup_dir = f"backups/complete_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.updates_completed = []
        self.updates_failed = []
        
    def run_update(self):
        """تشغيل التحديث الكامل"""
        print("🚀 Starting Complete System Update...")
        print("=" * 60)
        
        try:
            # 1. إنشاء نسخ احتياطية
            print("\n📦 Creating backups...")
            self.create_backups()
            
            # 2. تحديث الملفات الأساسية
            print("\n🔧 Updating core files...")
            self.update_core_files()
            
            # 3. تحديث ملفات التعلم
            print("\n🧠 Updating learning system...")
            self.update_learning_system()
            
            # 4. تحديث إعدادات الأدوات
            print("\n💹 Updating instruments configuration...")
            self.update_instruments_config()
            
            # 5. تحديث unified_standards.py
            print("\n📏 Updating unified standards...")
            self.update_unified_standards()
            
            # 6. تحديث ملفات الخادم
            print("\n🖥️ Updating server files...")
            self.update_server_files()
            
            # 7. التحقق من التحديثات
            print("\n✅ Verifying updates...")
            self.verify_updates()
            
            # 8. عرض التقرير
            self.show_report()
            
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            print(f"\n❌ Update failed: {str(e)}")
            print("\n⚠️ Restoring from backup...")
            self.restore_backup()
            raise
    
    def create_backups(self):
        """إنشاء نسخ احتياطية"""
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # قائمة الملفات والمجلدات للنسخ الاحتياطي
        items_to_backup = [
            'src',
            'models',
            'config',
            'unified_standards.py',
            'feature_engineer_adaptive_70.py',
            'instrument_manager.py',
            'support_resistance.py',
            'dynamic_sl_tp_system.py'
        ]
        
        for item in items_to_backup:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.copytree(item, os.path.join(self.backup_dir, item))
                else:
                    shutil.copy2(item, self.backup_dir)
                print(f"  ✅ Backed up: {item}")
    
    def update_core_files(self):
        """تحديث الملفات الأساسية"""
        updates = [
            {
                'file': 'unified_standards.py',
                'changes': [
                    ('STANDARD_FEATURES = 70', 'STANDARD_FEATURES = 75'),
                    ('# Standard feature count', '# Standard feature count (70 base + 5 S/R)')
                ]
            }
        ]
        
        for update in updates:
            if self.update_file(update['file'], update['changes']):
                self.updates_completed.append(update['file'])
            else:
                self.updates_failed.append(update['file'])
    
    def update_learning_system(self):
        """تحديث نظام التعلم"""
        # تحديث src/advanced_learner_unified.py
        learner_updates = {
            'file': 'src/advanced_learner_unified.py',
            'changes': [
                ('from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer',
                 'from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75'),
                ('self.feature_engineer = AdaptiveFeatureEngineer(target_features=70)',
                 'self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)'),
                ('df_features = self.feature_engineer.engineer_features(df)',
                 'df_features = self.feature_engineer.engineer_features(df, pair)'),
                ('expected_features = 70', 'expected_features = 75')
            ]
        }
        
        if self.update_file(learner_updates['file'], learner_updates['changes']):
            self.updates_completed.append(learner_updates['file'])
        else:
            self.updates_failed.append(learner_updates['file'])
        
        # تحديث src/continuous_learner_unified.py
        continuous_updates = {
            'file': 'src/continuous_learner_unified.py',
            'changes': [
                ('from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer',
                 'from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75'),
                ('feature_engineer = AdaptiveFeatureEngineer(target_features=70)',
                 'feature_engineer = AdaptiveFeatureEngineer75(target_features=75)'),
                ('TARGET_FEATURES = 70', 'TARGET_FEATURES = 75')
            ]
        }
        
        if self.update_file(continuous_updates['file'], continuous_updates['changes']):
            self.updates_completed.append(continuous_updates['file'])
        else:
            self.updates_failed.append(continuous_updates['file'])
    
    def update_server_files(self):
        """تحديث ملفات الخادم"""
        # تحديث src/mt5_bridge_server_advanced.py
        server_updates = {
            'file': 'src/mt5_bridge_server_advanced.py',
            'changes': [
                ('self.feature_engineer = AdaptiveFeatureEngineer(target_features=70)',
                 'self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)'),
                ('from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer',
                 'from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75'),
                ('expected_features = 70', 'expected_features = 75'),
                # إضافة دعم لميزات S/R
                ('# Feature extraction', '''# Feature extraction
        # Extract symbol from request if available
        symbol = data.get('symbol', 'UNKNOWN')'''),
                ('df_features = self.feature_engineer.engineer_features(df)',
                 'df_features = self.feature_engineer.engineer_features(df, symbol)')
            ]
        }
        
        if self.update_file(server_updates['file'], server_updates['changes']):
            self.updates_completed.append(server_updates['file'])
        else:
            self.updates_failed.append(server_updates['file'])
    
    def update_instruments_config(self):
        """تحديث إعدادات الأدوات"""
        # إنشاء ملف إعدادات للأدوات المفعلة
        instruments_config = {
            "forex_majors": [
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
                "AUDUSD", "NZDUSD", "USDCAD"
            ],
            "forex_minors": [
                "EURJPY", "GBPJPY", "EURGBP", "EURAUD", 
                "GBPAUD", "AUDCAD", "NZDCAD"
            ],
            "metals": [
                "XAUUSD", "XAGUSD"
            ],
            "energy": [
                "USOIL", "UKOIL"
            ],
            "indices": [
                "US30", "NAS100", "SP500", "DAX", "FTSE100"
            ],
            "crypto": [
                "BTCUSD", "ETHUSD"
            ],
            "enabled_for_training": {
                "immediate": [
                    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", 
                    "NZDUSD", "USDCAD", "USDCHF", "XAUUSD", 
                    "XAGUSD", "USOIL", "US30", "NAS100"
                ],
                "phase2": [
                    "EURJPY", "GBPJPY", "EURGBP", "SP500", 
                    "DAX", "BTCUSD", "ETHUSD"
                ]
            },
            "features_version": 75,
            "use_support_resistance": True
        }
        
        config_path = "config/instruments_enabled.json"
        os.makedirs("config", exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(instruments_config, f, indent=4)
        
        print(f"  ✅ Created: {config_path}")
        self.updates_completed.append(config_path)
    
    def update_unified_standards(self):
        """تحديث معايير موحدة لدعم 75 ميزة"""
        # إنشاء ملف unified_standards_75.py
        content = '''#!/usr/bin/env python3
"""
Unified Standards for 75 Features System
المعايير الموحدة لنظام 75 ميزة
"""

# Standard feature count (70 base + 5 S/R)
STANDARD_FEATURES = 75

# Base features (70)
BASE_FEATURES = 70

# Support/Resistance features (5)
SR_FEATURES = 5

# S/R feature names
SR_FEATURE_NAMES = [
    'distance_to_support_pct',
    'distance_to_resistance_pct',
    'nearest_support_strength',
    'nearest_resistance_strength',
    'position_in_sr_range'
]

# Model naming convention
def get_model_filename(symbol, timeframe, features=75):
    """Generate standard model filename"""
    return f"{symbol}_{timeframe}.pkl"

# Feature validation
def validate_features(features_count):
    """Validate feature count"""
    return features_count in [70, 75]  # Support both versions

# Get feature version
def get_feature_version(n_features):
    """Get feature version from count"""
    if n_features == 70:
        return "base"
    elif n_features == 75:
        return "base_with_sr"
    else:
        return "unknown"
'''
        
        with open('unified_standards_75.py', 'w') as f:
            f.write(content)
        
        print("  ✅ Created: unified_standards_75.py")
        self.updates_completed.append('unified_standards_75.py')
    
    def update_file(self, filepath, changes):
        """تحديث ملف مع التغييرات المحددة"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            for old, new in changes:
                if old in content:
                    content = content.replace(old, new)
                    logger.info(f"Updated in {filepath}: {old[:50]}...")
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Updated: {filepath}")
                return True
            else:
                print(f"  ℹ️ No changes needed: {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating {filepath}: {str(e)}")
            return False
    
    def verify_updates(self):
        """التحقق من التحديثات"""
        print("\n🔍 Verifying updates...")
        
        # التحقق من وجود الملفات الجديدة
        required_files = [
            'feature_engineer_adaptive_75.py',
            'support_resistance.py',
            'dynamic_sl_tp_system.py',
            'instrument_manager.py',
            'unified_standards_75.py',
            'config/instruments_enabled.json'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                print(f"  ✅ Found: {file}")
            else:
                print(f"  ❌ Missing: {file}")
                self.updates_failed.append(file)
    
    def restore_backup(self):
        """استرجاع النسخ الاحتياطية"""
        try:
            for item in os.listdir(self.backup_dir):
                source = os.path.join(self.backup_dir, item)
                dest = item
                
                if os.path.isdir(source):
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(source, dest)
                else:
                    shutil.copy2(source, dest)
                
                print(f"  ✅ Restored: {item}")
                
        except Exception as e:
            print(f"  ❌ Failed to restore backup: {str(e)}")
    
    def show_report(self):
        """عرض تقرير التحديث"""
        print("\n" + "=" * 60)
        print("📊 UPDATE REPORT")
        print("=" * 60)
        
        print(f"\n✅ Successful updates: {len(self.updates_completed)}")
        for item in self.updates_completed:
            print(f"  • {item}")
        
        if self.updates_failed:
            print(f"\n❌ Failed updates: {len(self.updates_failed)}")
            for item in self.updates_failed:
                print(f"  • {item}")
        else:
            print("\n🎉 All updates completed successfully!")
        
        print(f"\n📁 Backup location: {self.backup_dir}")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    updater = CompleteSystemUpdater()
    
    print("⚠️ COMPLETE SYSTEM UPDATE")
    print("This will update the system to use 75 features with S/R support")
    print("and add configuration for all trading instruments.")
    print("\nA backup will be created before making changes.")
    
    response = input("\nProceed with update? (yes/no): ")
    
    if response.lower() == 'yes':
        updater.run_update()
    else:
        print("Update cancelled.")