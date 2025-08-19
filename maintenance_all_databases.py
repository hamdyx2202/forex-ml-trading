#!/usr/bin/env python3
"""
🧹 Maintenance Script for Multiple Databases
📊 يعالج جميع قواعد البيانات الموجودة
🗄️ يستخدم نظام النسخ الاحتياطية الذكي
"""

import os
import subprocess
import logging
from smart_backup_manager import SmartBackupManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """صيانة جميع قواعد البيانات"""
    logger.info("="*70)
    logger.info("🧹 Starting Maintenance for All Databases")
    logger.info("="*70)
    
    # قواعد البيانات المعروفة
    databases = {
        './data/forex_ml.db': {
            'name': 'Main Forex ML Database',
            'priority': 1,
            'clean_data': True,
            'clean_models': True
        },
        './data/forex_data.db': {
            'name': 'Secondary Forex Database',
            'priority': 2,
            'clean_data': True,
            'clean_models': False  # قاعدة صغيرة، ربما للاختبار
        }
    }
    
    # البحث عن قواعد بيانات إضافية
    if os.path.exists('./data'):
        for file in os.listdir('./data'):
            if file.endswith('.db'):
                db_path = f'./data/{file}'
                if db_path not in databases:
                    size_mb = os.path.getsize(db_path) / (1024 * 1024)
                    databases[db_path] = {
                        'name': f'Unknown Database ({file})',
                        'priority': 3,
                        'clean_data': size_mb > 100,  # نظف فقط إذا كانت كبيرة
                        'clean_models': False
                    }
    
    # ترتيب حسب الأولوية
    sorted_dbs = sorted(databases.items(), key=lambda x: x[1]['priority'])
    
    logger.info(f"\n📊 Found {len(databases)} databases:")
    for db_path, info in sorted_dbs:
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            logger.info(f"   - {info['name']}: {size_mb:.2f} MB")
    
    # صيانة كل قاعدة
    for db_path, info in sorted_dbs:
        if not os.path.exists(db_path):
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Processing: {info['name']}")
        logger.info(f"   Path: {db_path}")
        
        try:
            # نسخ احتياطي أولاً
            logger.info("   📦 Creating backup...")
            subprocess.run([
                'python3', 'maintenance_script.py',
                '--db', db_path,
                '--backup'
            ], check=True)
            
            # تنظيف البيانات إذا لزم
            if info['clean_data']:
                size_before = os.path.getsize(db_path) / (1024 * 1024)
                logger.info(f"   🧹 Cleaning old data (current size: {size_before:.2f} MB)...")
                
                subprocess.run([
                    'python3', 'maintenance_script.py',
                    '--db', db_path,
                    '--data'
                ], check=True)
                
                # تحسين قاعدة البيانات
                logger.info("   🔧 Optimizing database...")
                subprocess.run([
                    'python3', 'maintenance_script.py',
                    '--db', db_path,
                    '--optimize'
                ], check=True)
                
                size_after = os.path.getsize(db_path) / (1024 * 1024)
                logger.info(f"   ✅ Size reduced: {size_before:.2f} MB → {size_after:.2f} MB")
            
            # تنظيف النماذج إذا كانت القاعدة الرئيسية
            if info['clean_models'] and db_path == './data/forex_ml.db':
                logger.info("   🤖 Cleaning old models...")
                subprocess.run([
                    'python3', 'maintenance_script.py',
                    '--models'
                ], check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Error processing {info['name']}: {e}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")
    
    # تنظيف السجلات (مرة واحدة فقط)
    logger.info(f"\n{'='*60}")
    logger.info("📝 Cleaning logs...")
    try:
        subprocess.run([
            'python3', 'maintenance_script.py',
            '--logs'
        ], check=True)
    except:
        pass
    
    # تقرير نهائي
    logger.info(f"\n{'='*70}")
    logger.info("✅ Maintenance completed for all databases!")
    
    # عرض الأحجام النهائية
    logger.info("\n📊 Final sizes:")
    total_size = 0
    for db_path, info in sorted_dbs:
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            logger.info(f"   - {info['name']}: {size_mb:.2f} MB")
            total_size += size_mb
    
    logger.info(f"\n💾 Total database size: {total_size:.2f} MB")
    
    # نصائح
    if total_size > 1000:  # أكثر من 1 جيجا
        logger.warning("\n⚠️  Total size exceeds 1 GB!")
        logger.info("   Consider:")
        logger.info("   - Reducing data retention periods")
        logger.info("   - Moving old data to archive")
        logger.info("   - Using separate databases for different years")
    
    # عرض تقرير النسخ الاحتياطية الذكية
    logger.info("\n📦 Smart Backup System Report:")
    backup_manager = SmartBackupManager()
    all_backups = backup_manager.list_backups()
    
    if all_backups:
        logger.info(f"   Total backups: {len(all_backups)}")
        
        # تجميع النسخ حسب قاعدة البيانات
        db_backups = {}
        for backup in all_backups:
            db_name = backup['name'].split('_backup_')[0]
            if db_name not in db_backups:
                db_backups[db_name] = []
            db_backups[db_name].append(backup)
        
        # عرض تفاصيل كل قاعدة
        for db_name, backups in db_backups.items():
            total_size = sum(b['size_mb'] for b in backups)
            logger.info(f"\n   📊 {db_name}:")
            logger.info(f"      - Backups: {len(backups)}")
            logger.info(f"      - Total size: {total_size:.1f} MB")
            logger.info(f"      - Compressed: {sum(1 for b in backups if b['compressed'])}")
            
            # عرض سياسة الاحتفاظ
            daily = sum(1 for b in backups if b['age_days'] <= 7)
            weekly = sum(1 for b in backups if 7 < b['age_days'] <= 28)
            monthly = sum(1 for b in backups if b['age_days'] > 28)
            
            logger.info(f"      - Daily backups: {daily}")
            logger.info(f"      - Weekly backups: {weekly}")
            logger.info(f"      - Monthly backups: {monthly}")
    else:
        logger.info("   No backups found")

if __name__ == "__main__":
    main()