#!/usr/bin/env python3
"""
📦 Smart Backup Manager - مدير النسخ الاحتياطية الذكي
🗄️ يحتفظ بنسخ ذكية دون إهدار المساحة
"""

import os
import shutil
import sqlite3
import hashlib
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SmartBackupManager:
    """مدير ذكي للنسخ الاحتياطية"""
    
    def __init__(self):
        self.backup_dir = './backups'
        self.archive_dir = './backups/archive'
        
        # سياسة الاحتفاظ بالنسخ
        self.retention_policy = {
            'daily': 7,      # احتفظ بـ 7 نسخ يومية
            'weekly': 4,     # احتفظ بـ 4 نسخ أسبوعية
            'monthly': 3     # احتفظ بـ 3 نسخ شهرية
        }
        
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
    
    def smart_backup(self, db_path):
        """نسخ احتياطي ذكي"""
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return None
        
        db_name = os.path.basename(db_path).replace('.db', '')
        
        # 1. تحقق من التغييرات
        if not self._has_changes(db_path):
            logger.info("📊 No changes detected, skipping backup")
            return None
        
        # 2. إنشاء نسخة احتياطية
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{db_name}_backup_{timestamp}.db"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        logger.info(f"📦 Creating backup: {backup_name}")
        shutil.copy2(db_path, backup_path)
        
        # 3. ضغط النسخة إذا كانت كبيرة
        size_mb = os.path.getsize(backup_path) / (1024 * 1024)
        if size_mb > 100:  # أكبر من 100 ميجا
            self._compress_backup(backup_path)
        
        # 4. تنظيف النسخ القديمة بذكاء
        self._smart_cleanup(db_name)
        
        return backup_path
    
    def _has_changes(self, db_path):
        """التحقق من وجود تغييرات منذ آخر نسخة"""
        try:
            # حساب checksum للملف الحالي
            current_hash = self._calculate_checksum(db_path)
            
            # البحث عن آخر نسخة
            db_name = os.path.basename(db_path).replace('.db', '')
            last_backup = self._get_last_backup(db_name)
            
            if not last_backup:
                return True  # لا توجد نسخ سابقة
            
            # مقارنة checksum
            last_hash = self._calculate_checksum(last_backup)
            return current_hash != last_hash
            
        except Exception as e:
            logger.error(f"Error checking changes: {e}")
            return True  # افترض وجود تغييرات
    
    def _calculate_checksum(self, file_path):
        """حساب checksum للملف"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # قراءة أول 10 ميجا فقط للسرعة
            for chunk in iter(lambda: f.read(10 * 1024 * 1024), b""):
                hash_md5.update(chunk)
                break  # فقط أول chunk
        return hash_md5.hexdigest()
    
    def _get_last_backup(self, db_name):
        """الحصول على آخر نسخة احتياطية"""
        backups = []
        for file in os.listdir(self.backup_dir):
            if file.startswith(f"{db_name}_backup_") and (file.endswith('.db') or file.endswith('.db.gz')):
                file_path = os.path.join(self.backup_dir, file)
                backups.append((file_path, os.path.getmtime(file_path)))
        
        if backups:
            backups.sort(key=lambda x: x[1], reverse=True)
            return backups[0][0]
        return None
    
    def _compress_backup(self, backup_path):
        """ضغط النسخة الاحتياطية"""
        try:
            import gzip
            compressed_path = backup_path + '.gz'
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # حذف النسخة غير المضغوطة
            os.remove(backup_path)
            
            original_size = os.path.getsize(backup_path) / (1024 * 1024)
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
            logger.info(f"   🗜️ Compressed: {original_size:.1f} MB → {compressed_size:.1f} MB")
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
    
    def _smart_cleanup(self, db_name):
        """تنظيف ذكي للنسخ القديمة"""
        logger.info("   🧹 Smart cleanup of old backups...")
        
        # جمع كل النسخ
        all_backups = []
        for file in os.listdir(self.backup_dir):
            if file.startswith(f"{db_name}_backup_"):
                file_path = os.path.join(self.backup_dir, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                all_backups.append({
                    'path': file_path,
                    'time': file_time,
                    'age_days': (datetime.now() - file_time).days
                })
        
        # ترتيب حسب التاريخ (الأحدث أولاً)
        all_backups.sort(key=lambda x: x['time'], reverse=True)
        
        # تطبيق سياسة الاحتفاظ
        to_keep = set()
        
        # 1. احتفظ بالنسخ اليومية (آخر 7 أيام)
        daily_count = 0
        for backup in all_backups:
            if backup['age_days'] <= 7 and daily_count < self.retention_policy['daily']:
                to_keep.add(backup['path'])
                daily_count += 1
        
        # 2. احتفظ بنسخة أسبوعية (آخر 4 أسابيع)
        weekly_count = 0
        for week in range(4):
            week_start = datetime.now() - timedelta(weeks=week+1)
            week_end = week_start + timedelta(days=7)
            
            for backup in all_backups:
                if week_start <= backup['time'] < week_end and weekly_count < self.retention_policy['weekly']:
                    to_keep.add(backup['path'])
                    weekly_count += 1
                    break
        
        # 3. احتفظ بنسخة شهرية (آخر 3 شهور)
        monthly_count = 0
        for month in range(3):
            month_start = datetime.now() - timedelta(days=30*(month+1))
            month_end = month_start + timedelta(days=30)
            
            for backup in all_backups:
                if month_start <= backup['time'] < month_end and monthly_count < self.retention_policy['monthly']:
                    to_keep.add(backup['path'])
                    monthly_count += 1
                    break
        
        # 4. احذف الباقي
        deleted_count = 0
        saved_space = 0
        
        for backup in all_backups:
            if backup['path'] not in to_keep:
                try:
                    size_mb = os.path.getsize(backup['path']) / (1024 * 1024)
                    os.remove(backup['path'])
                    deleted_count += 1
                    saved_space += size_mb
                    logger.info(f"      ❌ Deleted: {os.path.basename(backup['path'])} ({backup['age_days']} days old)")
                except Exception as e:
                    logger.error(f"      Failed to delete {backup['path']}: {e}")
        
        if deleted_count > 0:
            logger.info(f"   ✅ Deleted {deleted_count} old backups, saved {saved_space:.1f} MB")
        else:
            logger.info(f"   ✅ No old backups to delete")
        
        # عرض ملخص النسخ المحتفظ بها
        logger.info(f"   📦 Keeping {len(to_keep)} backups:")
        logger.info(f"      - Daily: {daily_count}")
        logger.info(f"      - Weekly: {weekly_count}")
        logger.info(f"      - Monthly: {monthly_count}")
    
    def restore_backup(self, backup_path, target_path):
        """استعادة نسخة احتياطية"""
        try:
            if backup_path.endswith('.gz'):
                import gzip
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(target_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, target_path)
            
            logger.info(f"✅ Restored backup to: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            return False
    
    def list_backups(self, db_name=None):
        """عرض قائمة النسخ الاحتياطية"""
        backups = []
        
        for file in os.listdir(self.backup_dir):
            if file.endswith('.db') or file.endswith('.db.gz'):
                if db_name and not file.startswith(f"{db_name}_backup_"):
                    continue
                    
                file_path = os.path.join(self.backup_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_days = (datetime.now() - mod_time).days
                
                backups.append({
                    'name': file,
                    'path': file_path,
                    'size_mb': size_mb,
                    'date': mod_time,
                    'age_days': age_days,
                    'compressed': file.endswith('.gz')
                })
        
        # ترتيب حسب التاريخ
        backups.sort(key=lambda x: x['date'], reverse=True)
        
        return backups

def main():
    """أمثلة على الاستخدام"""
    manager = SmartBackupManager()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'backup':
            # نسخ احتياطي ذكي
            db_path = sys.argv[2] if len(sys.argv) > 2 else './data/forex_ml.db'
            manager.smart_backup(db_path)
            
        elif command == 'list':
            # عرض النسخ
            backups = manager.list_backups()
            print(f"\n📦 Total backups: {len(backups)}")
            for b in backups[:10]:  # أول 10 فقط
                print(f"   - {b['name']} ({b['size_mb']:.1f} MB, {b['age_days']} days old)")
            
        elif command == 'restore':
            # استعادة نسخة
            if len(sys.argv) > 3:
                manager.restore_backup(sys.argv[2], sys.argv[3])
            else:
                print("Usage: python3 smart_backup_manager.py restore <backup_path> <target_path>")
    else:
        print("Smart Backup Manager")
        print("Usage:")
        print("  python3 smart_backup_manager.py backup [db_path]")
        print("  python3 smart_backup_manager.py list")
        print("  python3 smart_backup_manager.py restore <backup_path> <target_path>")

if __name__ == "__main__":
    main()