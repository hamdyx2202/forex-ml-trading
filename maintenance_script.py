#!/usr/bin/env python3
"""
🧹 Maintenance Script - سكريبت الصيانة الدورية
📊 ينظف البيانات القديمة والنماذج غير المستخدمة
💾 يحافظ على أداء النظام وتوفير المساحة
🗄️ يستخدم نظام النسخ الاحتياطية الذكي
"""

import os
import sqlite3
import json
import shutil
from datetime import datetime, timedelta
import logging
from smart_backup_manager import SmartBackupManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMaintenance:
    """نظام الصيانة الدورية"""
    
    def __init__(self, db_path=None):
        # السماح بتحديد قاعدة البيانات
        self.db_path = db_path or './data/forex_ml.db'
        self.models_dir = './trained_models'
        self.backup_dir = './backups'
        self.logs_dir = './logs'
        self.model_performance_file = './model_performance.json'
        
        # إعدادات الاحتفاظ بالبيانات
        self.data_retention = {
            'M1': 90,    # 3 شهور
            'M5': 180,   # 6 شهور
            'M15': 365,  # سنة
            'M30': 365,  # سنة
            'H1': 730,   # سنتين
            'H4': 730,   # سنتين
            'D1': 1095   # 3 سنوات
        }
        
        # إنشاء مجلد النسخ الاحتياطية
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # مدير النسخ الاحتياطية الذكي
        self.backup_manager = SmartBackupManager()
        
    def run_full_maintenance(self):
        """تشغيل الصيانة الكاملة"""
        logger.info("="*70)
        logger.info("🧹 Starting Full System Maintenance")
        logger.info("="*70)
        
        # 1. نسخ احتياطي قبل الصيانة
        self.backup_database()
        
        # 2. تنظيف البيانات القديمة
        self.cleanup_old_data()
        
        # 3. تنظيف النماذج القديمة
        self.cleanup_old_models()
        
        # 4. تنظيف السجلات القديمة
        self.cleanup_old_logs()
        
        # 5. تحسين قاعدة البيانات
        self.optimize_database()
        
        # 6. تقرير الصيانة
        self.generate_maintenance_report()
        
        logger.info("✅ Maintenance completed successfully!")
        
    def backup_database(self):
        """نسخ احتياطي ذكي لقاعدة البيانات"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning("Database not found, skipping backup")
                return
            
            logger.info("📦 Creating smart backup...")
            
            # استخدام مدير النسخ الاحتياطية الذكي
            backup_path = self.backup_manager.smart_backup(self.db_path)
            
            if backup_path:
                logger.info(f"✅ Smart backup created successfully")
            else:
                logger.info("📊 No backup needed (no changes detected)")
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            
    def cleanup_old_data(self):
        """تنظيف البيانات القديمة من قاعدة البيانات"""
        logger.info("\n📊 Cleaning old price data...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            total_deleted = 0
            
            # حذف البيانات القديمة حسب الفريم
            for timeframe, days in self.data_retention.items():
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # عد السجلات قبل الحذف
                cursor.execute(
                    "SELECT COUNT(*) FROM price_data WHERE timeframe = ? AND time < ?",
                    (timeframe, cutoff_date.strftime('%Y-%m-%d'))
                )
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # حذف السجلات القديمة
                    cursor.execute(
                        "DELETE FROM price_data WHERE timeframe = ? AND time < ?",
                        (timeframe, cutoff_date.strftime('%Y-%m-%d'))
                    )
                    
                    logger.info(f"   ✅ {timeframe}: Deleted {count:,} records older than {days} days")
                    total_deleted += count
            
            # حذف البيانات المكررة (إن وجدت)
            cursor.execute("""
                DELETE FROM price_data 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM price_data 
                    GROUP BY symbol, timeframe, time
                )
            """)
            duplicates = cursor.rowcount
            if duplicates > 0:
                logger.info(f"   ✅ Removed {duplicates:,} duplicate records")
                total_deleted += duplicates
            
            conn.commit()
            conn.close()
            
            logger.info(f"   📊 Total records deleted: {total_deleted:,}")
            
        except Exception as e:
            logger.error(f"❌ Data cleanup failed: {e}")
            
    def cleanup_old_models(self):
        """تنظيف النماذج القديمة والاحتفاظ بأفضل النماذج حسب الأداء"""
        logger.info("\n🤖 Cleaning old models and keeping best performers...")
        
        try:
            if not os.path.exists(self.models_dir):
                logger.warning("Models directory not found")
                return
            
            # قراءة سجل أداء النماذج
            model_performance = self._load_model_performance()
            
            # جمع معلومات النماذج
            models_info = {}
            
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl') and 'scaler' not in file:
                    file_path = os.path.join(self.models_dir, file)
                    
                    # تحليل اسم الملف
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:])
                        
                        # معلومات الملف
                        file_stat = os.stat(file_path)
                        age_days = (datetime.now() - datetime.fromtimestamp(file_stat.st_mtime)).days
                        size_mb = file_stat.st_size / (1024 * 1024)
                        
                        # الحصول على أداء النموذج
                        performance_key = f"{symbol}_{timeframe}_{model_type}"
                        performance = model_performance.get(performance_key, {})
                        accuracy = performance.get('accuracy', 0.5)
                        win_rate = performance.get('win_rate', 50)
                        profit_factor = performance.get('profit_factor', 1.0)
                        
                        # حساب نقاط الأداء المركبة
                        performance_score = (accuracy * 100 * 0.4) + (win_rate * 0.3) + (profit_factor * 10 * 0.3)
                        
                        key = f"{symbol}_{timeframe}"
                        if key not in models_info:
                            models_info[key] = []
                        
                        models_info[key].append({
                            'file': file,
                            'path': file_path,
                            'model_type': model_type,
                            'age_days': age_days,
                            'size_mb': size_mb,
                            'modified': datetime.fromtimestamp(file_stat.st_mtime),
                            'accuracy': accuracy,
                            'win_rate': win_rate,
                            'profit_factor': profit_factor,
                            'performance_score': performance_score
                        })
            
            total_deleted = 0
            total_size_saved = 0
            kept_models = []
            
            # تنظيف النماذج لكل زوج
            for pair_key, models in models_info.items():
                logger.info(f"\n   📊 Processing {pair_key}...")
                
                # ترتيب حسب الأداء (الأفضل أولاً)
                models.sort(key=lambda x: x['performance_score'], reverse=True)
                
                # طباعة أفضل النماذج
                if models:
                    logger.info(f"      Best model: {models[0]['model_type']} - Score: {models[0]['performance_score']:.2f}")
                
                # الاحتفاظ بأفضل 3 نماذج فقط (مع شروط)
                models_to_keep = []
                models_to_delete = []
                
                for i, model in enumerate(models):
                    # الاحتفاظ بأفضل 3 نماذج إذا كانت:
                    # 1. من أفضل 3 في الأداء
                    # 2. أحدث من 60 يوم
                    # 3. لها performance_score > 50
                    if i < 3 and model['age_days'] < 60 and model['performance_score'] > 50:
                        models_to_keep.append(model)
                        kept_models.append({
                            'pair': pair_key,
                            'model': model['model_type'],
                            'score': model['performance_score'],
                            'accuracy': model['accuracy'],
                            'win_rate': model['win_rate']
                        })
                    else:
                        models_to_delete.append(model)
                
                # حذف النماذج الضعيفة أو القديمة
                for model in models_to_delete:
                    reason = ""
                    if model['age_days'] > 60:
                        reason = f"too old ({model['age_days']} days)"
                    elif model['performance_score'] <= 50:
                        reason = f"low performance (score: {model['performance_score']:.1f})"
                    else:
                        reason = "excess model"
                    
                    os.remove(model['path'])
                    logger.info(f"      ❌ Deleted: {model['model_type']} - {reason}")
                    total_deleted += 1
                    total_size_saved += model['size_mb']
                
                # إذا لم نحتفظ بأي نموذج، احتفظ بالأفضل حتى لو كان ضعيف
                if not models_to_keep and models:
                    best_model = models[0]
                    logger.info(f"      ⚠️  Keeping best available model despite low performance")
                    models_to_delete.remove(best_model)
                    total_deleted -= 1
                    total_size_saved -= best_model['size_mb']
            
            # حذف ملفات scalers القديمة
            for file in os.listdir(self.models_dir):
                if 'scaler' in file and file.endswith('.pkl'):
                    file_path = os.path.join(self.models_dir, file)
                    file_stat = os.stat(file_path)
                    age_days = (datetime.now() - datetime.fromtimestamp(file_stat.st_mtime)).days
                    
                    if age_days > 30:
                        os.remove(file_path)
                        logger.info(f"   ✅ Deleted old scaler: {file}")
                        total_deleted += 1
            
            logger.info(f"   🤖 Total models deleted: {total_deleted}")
            logger.info(f"   💾 Space saved: {total_size_saved:.2f} MB")
            
            # حفظ تقرير بالنماذج المحتفظ بها
            if kept_models:
                self._save_kept_models_report(kept_models)
            
        except Exception as e:
            logger.error(f"❌ Model cleanup failed: {e}")
            
    def cleanup_old_logs(self):
        """تنظيف ملفات السجلات القديمة"""
        logger.info("\n📝 Cleaning old logs...")
        
        try:
            log_files = [
                'enhanced_ml_server.log',
                'server.log',
                'advanced_ml_server.log',
                'server_output.log'
            ]
            
            total_cleaned = 0
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    file_size = os.path.getsize(log_file) / (1024 * 1024)  # MB
                    
                    if file_size > 100:  # إذا كان أكبر من 100 ميجا
                        # احتفظ بآخر 1000 سطر فقط
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                        
                        if len(lines) > 1000:
                            with open(log_file, 'w') as f:
                                f.writelines(lines[-1000:])
                            
                            logger.info(f"   ✅ Truncated {log_file} (was {file_size:.1f} MB)")
                            total_cleaned += 1
            
            # حذف ملفات nohup.out القديمة
            if os.path.exists('nohup.out'):
                os.remove('nohup.out')
                logger.info("   ✅ Deleted nohup.out")
                total_cleaned += 1
            
            logger.info(f"   📝 Total log files cleaned: {total_cleaned}")
            
        except Exception as e:
            logger.error(f"❌ Log cleanup failed: {e}")
            
    def optimize_database(self):
        """تحسين أداء قاعدة البيانات"""
        logger.info("\n🔧 Optimizing database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # حجم قاعدة البيانات قبل التحسين
            cursor.execute("SELECT page_count * page_size / 1024 / 1024 FROM pragma_page_count(), pragma_page_size()")
            size_before = cursor.fetchone()[0]
            
            # تشغيل VACUUM لتحسين وضغط قاعدة البيانات
            cursor.execute("VACUUM")
            
            # إعادة بناء الفهارس
            cursor.execute("REINDEX")
            
            # تحليل الجداول لتحسين الاستعلامات
            cursor.execute("ANALYZE")
            
            # حجم قاعدة البيانات بعد التحسين
            cursor.execute("SELECT page_count * page_size / 1024 / 1024 FROM pragma_page_count(), pragma_page_size()")
            size_after = cursor.fetchone()[0]
            
            conn.close()
            
            space_saved = size_before - size_after
            logger.info(f"   ✅ Database optimized")
            logger.info(f"   💾 Size before: {size_before:.2f} MB")
            logger.info(f"   💾 Size after: {size_after:.2f} MB")
            logger.info(f"   💾 Space saved: {space_saved:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Database optimization failed: {e}")
            
    def _cleanup_old_backups(self):
        """حذف النسخ الاحتياطية القديمة - لم تعد مطلوبة مع النظام الذكي"""
        # هذه الدالة لم تعد مطلوبة لأن SmartBackupManager يتولى التنظيف الذكي
        pass
            
    def generate_maintenance_report(self):
        """توليد تقرير الصيانة"""
        logger.info("\n📊 Generating maintenance report...")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'database_size': 0,
                'models_count': 0,
                'total_records': 0,
                'disk_usage': {}
            }
            
            # حجم قاعدة البيانات
            if os.path.exists(self.db_path):
                report['database_size'] = os.path.getsize(self.db_path) / (1024 * 1024)
                
                # عدد السجلات
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM price_data")
                report['total_records'] = cursor.fetchone()[0]
                
                # إحصائيات لكل زوج
                cursor.execute("""
                    SELECT symbol, timeframe, COUNT(*) as count 
                    FROM price_data 
                    GROUP BY symbol, timeframe 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                report['top_pairs'] = cursor.fetchall()
                conn.close()
            
            # عدد النماذج
            if os.path.exists(self.models_dir):
                report['models_count'] = len([f for f in os.listdir(self.models_dir) if f.endswith('.pkl')])
            
            # استخدام القرص
            report['disk_usage'] = {
                'database_mb': report['database_size'],
                'models_mb': sum(os.path.getsize(os.path.join(self.models_dir, f)) 
                               for f in os.listdir(self.models_dir) if f.endswith('.pkl')) / (1024 * 1024) if os.path.exists(self.models_dir) else 0,
                'backups_mb': sum(os.path.getsize(os.path.join(self.backup_dir, f)) 
                               for f in os.listdir(self.backup_dir)) / (1024 * 1024) if os.path.exists(self.backup_dir) else 0
            }
            
            # حفظ التقرير
            report_path = f"maintenance_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"   ✅ Report saved to: {report_path}")
            logger.info(f"   📊 Database: {report['database_size']:.2f} MB ({report['total_records']:,} records)")
            logger.info(f"   🤖 Models: {report['models_count']} files ({report['disk_usage']['models_mb']:.2f} MB)")
            logger.info(f"   💾 Total disk usage: {sum(report['disk_usage'].values()):.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")

    def _load_model_performance(self):
        """قراءة سجل أداء النماذج"""
        if os.path.exists(self.model_performance_file):
            try:
                with open(self.model_performance_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_kept_models_report(self, kept_models):
        """حفظ تقرير بالنماذج المحتفظ بها"""
        report_file = f"kept_models_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'kept_models': kept_models,
                'total_models': len(kept_models)
            }, f, indent=2)
        logger.info(f"   📊 Kept models report saved to: {report_file}")

def main():
    """تشغيل الصيانة"""
    # السماح بتحديد قاعدة البيانات
    import argparse
    parser = argparse.ArgumentParser(description='System Maintenance Script')
    parser.add_argument('--db', help='Database path', default='./data/forex_ml.db')
    parser.add_argument('--data', action='store_true', help='Clean old data only')
    parser.add_argument('--models', action='store_true', help='Clean old models only')
    parser.add_argument('--logs', action='store_true', help='Clean old logs only')
    parser.add_argument('--optimize', action='store_true', help='Optimize database only')
    parser.add_argument('--backup', action='store_true', help='Backup only')
    parser.add_argument('--list-dbs', action='store_true', help='List available databases')
    
    args = parser.parse_args()
    
    # عرض قواعد البيانات المتاحة
    if args.list_dbs:
        print("\n📊 Available databases:")
        if os.path.exists('./data'):
            for file in os.listdir('./data'):
                if file.endswith('.db'):
                    size_mb = os.path.getsize(f'./data/{file}') / (1024 * 1024)
                    print(f"   - ./data/{file} ({size_mb:.2f} MB)")
        return
    
    maintenance = SystemMaintenance(db_path=args.db)
    
    # تنفيذ الأوامر
    if args.data:
        maintenance.cleanup_old_data()
    elif args.models:
        maintenance.cleanup_old_models()
    elif args.logs:
        maintenance.cleanup_old_logs()
    elif args.optimize:
        maintenance.optimize_database()
    elif args.backup:
        maintenance.backup_database()
    else:
        # تشغيل الصيانة الكاملة
        maintenance.run_full_maintenance()
        
        # عرض معلومات النسخ الاحتياطية
        logger.info("\n📦 Backup Information:")
        backups = maintenance.backup_manager.list_backups()
        logger.info(f"   Total backups: {len(backups)}")
        if backups:
            total_size = sum(b['size_mb'] for b in backups)
            logger.info(f"   Total backup size: {total_size:.1f} MB")
            logger.info(f"   Oldest backup: {backups[-1]['age_days']} days old")
            logger.info(f"   Newest backup: {backups[0]['age_days']} days old")

if __name__ == "__main__":
    main()