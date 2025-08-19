#!/usr/bin/env python3
"""
🔄 استرجاع النسخة الاحتياطية
📦 يسترجع آخر نسخة احتياطية أو نسخة محددة
"""

import os
import shutil
import sqlite3
from datetime import datetime
import gzip

def list_available_backups():
    """عرض النسخ الاحتياطية المتاحة"""
    backup_dir = './backups'
    if not os.path.exists(backup_dir):
        print("❌ No backup directory found!")
        return []
    
    backups = []
    for file in os.listdir(backup_dir):
        if file.endswith('.db.gz') or file.endswith('.db'):
            file_path = os.path.join(backup_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = os.path.getmtime(file_path)
            
            backups.append({
                'file': file,
                'path': file_path,
                'size_mb': size_mb,
                'date': datetime.fromtimestamp(mod_time)
            })
    
    # ترتيب حسب التاريخ (الأحدث أولاً)
    backups.sort(key=lambda x: x['date'], reverse=True)
    return backups

def restore_backup(backup_path, target_path='./data/forex_ml.db'):
    """استرجاع نسخة احتياطية"""
    print(f"\n🔄 Restoring backup: {os.path.basename(backup_path)}")
    print(f"📍 Target: {target_path}")
    
    # التأكد من وجود النسخة الاحتياطية
    if not os.path.exists(backup_path):
        print(f"❌ Backup file not found: {backup_path}")
        return False
    
    # إنشاء مجلد البيانات إذا لم يكن موجود
    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # نسخ احتياطي للملف الحالي إذا كان موجود
    if os.path.exists(target_path):
        current_backup = f"{target_path}.before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📦 Backing up current database to: {current_backup}")
        shutil.copy2(target_path, current_backup)
    
    try:
        # إذا كان الملف مضغوط
        if backup_path.endswith('.gz'):
            print("📦 Decompressing backup...")
            with gzip.open(backup_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # نسخ مباشر
            shutil.copy2(backup_path, target_path)
        
        # التحقق من سلامة قاعدة البيانات
        print("🔍 Verifying database integrity...")
        conn = sqlite3.connect(target_path)
        cursor = conn.cursor()
        
        # فحص سلامة قاعدة البيانات
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        
        if integrity == 'ok':
            print("✅ Database integrity check passed")
            
            # عرض معلومات قاعدة البيانات
            cursor.execute("SELECT COUNT(*) FROM price_data")
            record_count = cursor.fetchone()[0]
            print(f"📊 Total records: {record_count:,}")
            
            # عرض الأزواج المتاحة
            cursor.execute("SELECT DISTINCT symbol FROM price_data")
            symbols = [row[0] for row in cursor.fetchall()]
            print(f"💹 Available pairs: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                print(f"   ... and {len(symbols) - 10} more")
        else:
            print(f"⚠️ Database integrity check failed: {integrity}")
        
        conn.close()
        
        print("\n✅ Backup restored successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error restoring backup: {e}")
        return False

def main():
    """القائمة الرئيسية"""
    print("="*60)
    print("🔄 Database Backup Restore Tool")
    print("="*60)
    
    # عرض النسخ الاحتياطية المتاحة
    backups = list_available_backups()
    
    if not backups:
        print("❌ No backups found!")
        return
    
    print("\n📦 Available backups:")
    for i, backup in enumerate(backups):
        age_days = (datetime.now() - backup['date']).days
        print(f"{i+1}. {backup['file']}")
        print(f"   📅 Date: {backup['date'].strftime('%Y-%m-%d %H:%M:%S')} ({age_days} days ago)")
        print(f"   💾 Size: {backup['size_mb']:.2f} MB")
        print()
    
    # اختيار النسخة الاحتياطية
    try:
        choice = input("\n🔢 Enter backup number to restore (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            print("👋 Exiting...")
            return
        
        backup_index = int(choice) - 1
        if 0 <= backup_index < len(backups):
            selected_backup = backups[backup_index]
            
            # تأكيد الاستعادة
            print(f"\n⚠️ WARNING: This will replace the current database!")
            print(f"📦 Selected backup: {selected_backup['file']}")
            confirm = input("Are you sure? (yes/no): ")
            
            if confirm.lower() == 'yes':
                restore_backup(selected_backup['path'])
            else:
                print("❌ Restore cancelled")
        else:
            print("❌ Invalid selection")
            
    except ValueError:
        print("❌ Invalid input")
    except KeyboardInterrupt:
        print("\n❌ Cancelled")

if __name__ == "__main__":
    # السماح بتحديد ملف معين مباشرة
    import sys
    if len(sys.argv) > 1:
        # استعادة ملف محدد
        backup_file = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else './data/forex_ml.db'
        restore_backup(backup_file, target)
    else:
        # القائمة التفاعلية
        main()