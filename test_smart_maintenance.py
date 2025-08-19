#!/usr/bin/env python3
"""
🧪 Test Smart Maintenance System
📦 يختبر نظام الصيانة مع النسخ الاحتياطية الذكية
"""

import os
import time
import sqlite3
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_database():
    """إنشاء قاعدة بيانات تجريبية"""
    test_db = './data/test_maintenance.db'
    os.makedirs('./data', exist_ok=True)
    
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    
    # إنشاء جدول تجريبي
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # إضافة بيانات تجريبية
    for i in range(10):
        cursor.execute("INSERT INTO test_data (data) VALUES (?)", (f"Test data {i}",))
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Created test database: {test_db}")
    return test_db

def modify_test_database(db_path):
    """تعديل قاعدة البيانات التجريبية"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # إضافة بيانات جديدة
    cursor.execute("INSERT INTO test_data (data) VALUES (?)", ("Modified data",))
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Modified test database")

def main():
    logger.info("="*70)
    logger.info("🧪 Testing Smart Maintenance System")
    logger.info("="*70)
    
    # 1. إنشاء قاعدة بيانات تجريبية
    test_db = create_test_database()
    
    # 2. تشغيل الصيانة مع النسخ الاحتياطي الذكي
    logger.info("\n📦 First maintenance run (should create backup)...")
    os.system(f"python3 maintenance_script.py --db {test_db} --backup")
    
    # 3. تشغيل مرة أخرى بدون تغييرات
    logger.info("\n📦 Second maintenance run (no changes - should skip backup)...")
    time.sleep(2)
    os.system(f"python3 maintenance_script.py --db {test_db} --backup")
    
    # 4. تعديل قاعدة البيانات
    logger.info("\n✏️ Modifying database...")
    modify_test_database(test_db)
    
    # 5. تشغيل مرة أخرى بعد التغييرات
    logger.info("\n📦 Third maintenance run (with changes - should create backup)...")
    os.system(f"python3 maintenance_script.py --db {test_db} --backup")
    
    # 6. عرض النسخ الاحتياطية
    logger.info("\n📊 Listing all backups...")
    os.system("python3 smart_backup_manager.py list")
    
    # 7. تشغيل الصيانة الكاملة
    logger.info("\n🧹 Running full maintenance...")
    os.system(f"python3 maintenance_script.py --db {test_db}")
    
    # تنظيف
    logger.info("\n🧹 Cleaning up test files...")
    if os.path.exists(test_db):
        os.remove(test_db)
    
    logger.info("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()