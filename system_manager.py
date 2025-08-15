#!/usr/bin/env python3
"""
🎯 مدير النظام الموحد - أوامر سهلة ومباشرة
Unified System Manager - Simple Direct Commands
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import glob
import shutil
from loguru import logger

# إعداد logging
logger.add("system_manager_{time}.log", rotation="500 MB")

class SystemManager:
    """مدير النظام الرئيسي"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "models" / "unified"
        self.logs_dir = self.base_dir / "logs"
        
        # قائمة الأزواج
        self.all_pairs = {
            "phase1": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "XAUUSD"],
            "phase2": ["EURJPY", "GBPJPY", "EURGBP", "XAGUSD", "USOIL", "US30", "NAS100"],
            "phase3": ["SP500", "DAX", "BTCUSD", "ETHUSD", "EURAUD", "GBPAUD"]
        }
        
        self.timeframes = ["M5", "M15", "H1", "H4"]
    
    def train_new_models(self, pairs=None, phase="all"):
        """
        تدريب نماذج جديدة
        
        Usage:
            python3 system_manager.py train --phase phase1
            python3 system_manager.py train --pairs EURUSD,GBPUSD
            python3 system_manager.py train  # تدريب الكل
        """
        print("🚀 بدء تدريب النماذج الجديدة...")
        
        # تحديد الأزواج للتدريب
        if pairs:
            pairs_list = [p.strip() for p in pairs.split(",")]
        elif phase == "all":
            pairs_list = []
            for p in self.all_pairs.values():
                pairs_list.extend(p)
        else:
            pairs_list = self.all_pairs.get(phase, [])
        
        if not pairs_list:
            print("❌ لا توجد أزواج للتدريب!")
            return
        
        print(f"📊 سيتم تدريب {len(pairs_list)} زوج: {', '.join(pairs_list)}")
        
        # إنشاء ملف تدريب مؤقت
        train_script = f"""
import sys
sys.path.append('{self.base_dir}/src')
from advanced_learner_unified import AdvancedLearner

pairs = {pairs_list}
timeframes = {self.timeframes}

learner = AdvancedLearner()
total = len(pairs) * len(timeframes)
completed = 0

for pair in pairs:
    for tf in timeframes:
        completed += 1
        print(f"\\n[{completed}/{total}] Training {pair} {tf}...")
        try:
            learner.train_model(pair, tf)
            print(f"✅ {pair} {tf} - Success")
        except Exception as e:
            print(f"❌ {pair} {tf} - Failed: {str(e)}")

print("\\n✅ Training completed!")
"""
        
        # حفظ وتشغيل السكريبت
        with open("temp_train.py", "w") as f:
            f.write(train_script)
        
        try:
            subprocess.run([sys.executable, "temp_train.py"], check=True)
        finally:
            if os.path.exists("temp_train.py"):
                os.remove("temp_train.py")
    
    def retrain_existing(self):
        """إعادة تدريب النماذج الموجودة"""
        print("🔄 إعادة تدريب النماذج الموجودة...")
        
        # جمع الأزواج من النماذج الموجودة
        existing_pairs = set()
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*.pkl"):
                # استخراج الزوج من اسم الملف (EURUSD_H1.pkl)
                parts = model_file.stem.split("_")
                if len(parts) >= 2:
                    pair = parts[0]
                    existing_pairs.add(pair)
        
        if existing_pairs:
            pairs_str = ",".join(sorted(existing_pairs))
            print(f"📊 وُجد {len(existing_pairs)} زوج للإعادة التدريب")
            self.train_new_models(pairs=pairs_str)
        else:
            print("❌ لا توجد نماذج موجودة لإعادة التدريب!")
    
    def start_continuous_learning(self):
        """تشغيل التعلم المستمر"""
        print("🧠 تشغيل التعلم المستمر...")
        
        # إيقاف الجلسة القديمة إن وجدت
        subprocess.run(["tmux", "kill-session", "-t", "learning"], 
                      capture_output=True)
        
        # تشغيل جلسة جديدة
        cmd = f"""
tmux new-session -d -s learning "cd {self.base_dir} && {sys.executable} src/continuous_learner_unified.py"
"""
        subprocess.run(cmd, shell=True)
        print("✅ التعلم المستمر يعمل في tmux session: learning")
        print("   للمشاهدة: tmux attach -t learning")
    
    def start_advanced_learning(self):
        """تشغيل التعلم المتقدم"""
        print("🎯 تشغيل التعلم المتقدم...")
        
        # إيقاف الجلسة القديمة إن وجدت
        subprocess.run(["tmux", "kill-session", "-t", "advanced"], 
                      capture_output=True)
        
        # تشغيل جلسة جديدة
        cmd = f"""
tmux new-session -d -s advanced "cd {self.base_dir} && {sys.executable} src/advanced_learner_unified.py --auto-improve"
"""
        subprocess.run(cmd, shell=True)
        print("✅ التعلم المتقدم يعمل في tmux session: advanced")
        print("   للمشاهدة: tmux attach -t advanced")
    
    def start_server(self):
        """تشغيل السيرفر الرئيسي"""
        print("🖥️ تشغيل السيرفر الرئيسي...")
        
        # إيقاف الجلسة القديمة إن وجدت
        subprocess.run(["tmux", "kill-session", "-t", "server"], 
                      capture_output=True)
        
        # تشغيل جلسة جديدة
        cmd = f"""
tmux new-session -d -s server "cd {self.base_dir} && {sys.executable} src/mt5_bridge_server_advanced.py"
"""
        subprocess.run(cmd, shell=True)
        print("✅ السيرفر يعمل في tmux session: server")
        print("   للمشاهدة: tmux attach -t server")
        print(f"   URL: http://localhost:5000")
    
    def clean_old_models(self, days=180):
        """حذف النماذج القديمة"""
        print(f"🗑️ حذف النماذج الأقدم من {days} يوم...")
        
        if not self.models_dir.exists():
            print("❌ مجلد النماذج غير موجود!")
            return
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # مجلدات النماذج المحتملة
        model_dirs = [
            self.models_dir,
            self.base_dir / "models",
            self.base_dir / "models" / "backup"
        ]
        
        for model_dir in model_dirs:
            if not model_dir.exists():
                continue
                
            for model_file in model_dir.glob("**/*.pkl"):
                try:
                    # الحصول على تاريخ آخر تعديل
                    mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        print(f"  🗑️ حذف: {model_file.name} (آخر تعديل: {mtime.strftime('%Y-%m-%d')})")
                        model_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️ خطأ في حذف {model_file}: {str(e)}")
        
        print(f"✅ تم حذف {deleted_count} نموذج قديم")
    
    def remove_70_feature_models(self):
        """حذف النماذج ذات 70 ميزة"""
        print("🗑️ حذف النماذج القديمة (70 ميزة)...")
        
        import joblib
        deleted_count = 0
        checked_count = 0
        
        for model_dir in [self.models_dir, self.base_dir / "models"]:
            if not model_dir.exists():
                continue
                
            for model_file in model_dir.glob("**/*.pkl"):
                checked_count += 1
                try:
                    # تحميل النموذج للتحقق من عدد الميزات
                    model_data = joblib.load(model_file)
                    
                    n_features = None
                    if 'n_features' in model_data:
                        n_features = model_data['n_features']
                    elif 'scaler' in model_data and hasattr(model_data['scaler'], 'n_features_in_'):
                        n_features = model_data['scaler'].n_features_in_
                    
                    if n_features == 70:
                        print(f"  🗑️ حذف: {model_file.name} (70 features)")
                        model_file.unlink()
                        deleted_count += 1
                    elif n_features == 75:
                        print(f"  ✅ الإبقاء: {model_file.name} (75 features)")
                    else:
                        print(f"  ⚠️ غير معروف: {model_file.name} ({n_features} features)")
                        
                except Exception as e:
                    print(f"  ⚠️ خطأ في فحص {model_file}: {str(e)}")
        
        print(f"✅ تم فحص {checked_count} نموذج، حذف {deleted_count}")
    
    def status(self):
        """عرض حالة النظام"""
        print("\n" + "="*60)
        print("📊 حالة النظام")
        print("="*60)
        
        # فحص العمليات
        print("\n🔄 العمليات النشطة:")
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("  لا توجد جلسات tmux نشطة")
        
        # فحص النماذج
        print("\n📦 النماذج:")
        if self.models_dir.exists():
            models_75 = 0
            models_70 = 0
            
            for model_file in self.models_dir.glob("*.pkl"):
                try:
                    import joblib
                    model_data = joblib.load(model_file)
                    n_features = model_data.get('n_features', 0)
                    if n_features == 75:
                        models_75 += 1
                    elif n_features == 70:
                        models_70 += 1
                except:
                    pass
            
            print(f"  نماذج 75 ميزة: {models_75}")
            print(f"  نماذج 70 ميزة: {models_70}")
            print(f"  إجمالي النماذج: {len(list(self.models_dir.glob('*.pkl')))}")
        
        # فحص قاعدة البيانات
        print("\n💾 قاعدة البيانات:")
        db_path = self.base_dir / "trading_data.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            trades_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM signals")
            signals_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"  الصفقات: {trades_count}")
            print(f"  الإشارات: {signals_count}")
        
        print("\n" + "="*60)
    
    def quick_start(self):
        """تشغيل سريع للنظام"""
        print("🚀 تشغيل سريع للنظام...")
        
        # 1. تنظيف النماذج القديمة
        print("\n1️⃣ تنظيف النماذج القديمة...")
        self.remove_70_feature_models()
        
        # 2. تشغيل السيرفر
        print("\n2️⃣ تشغيل السيرفر...")
        self.start_server()
        
        # 3. تشغيل التعلم المستمر
        print("\n3️⃣ تشغيل التعلم المستمر...")
        self.start_continuous_learning()
        
        # 4. عرض الحالة
        print("\n4️⃣ حالة النظام:")
        self.status()
        
        print("\n✅ النظام جاهز للعمل!")


def main():
    parser = argparse.ArgumentParser(description="مدير النظام الموحد")
    
    subparsers = parser.add_subparsers(dest='command', help='الأوامر المتاحة')
    
    # أمر التدريب
    train_parser = subparsers.add_parser('train', help='تدريب نماذج جديدة')
    train_parser.add_argument('--pairs', type=str, help='أزواج محددة (مثل: EURUSD,GBPUSD)')
    train_parser.add_argument('--phase', type=str, default='all', 
                            choices=['all', 'phase1', 'phase2', 'phase3'],
                            help='مرحلة التدريب')
    
    # أمر إعادة التدريب
    subparsers.add_parser('retrain', help='إعادة تدريب النماذج الموجودة')
    
    # أمر التعلم المستمر
    subparsers.add_parser('learning', help='تشغيل التعلم المستمر')
    
    # أمر التعلم المتقدم
    subparsers.add_parser('advanced', help='تشغيل التعلم المتقدم')
    
    # أمر السيرفر
    subparsers.add_parser('server', help='تشغيل السيرفر الرئيسي')
    
    # أمر التنظيف
    clean_parser = subparsers.add_parser('clean', help='حذف النماذج القديمة')
    clean_parser.add_argument('--days', type=int, default=180, 
                            help='حذف النماذج الأقدم من X يوم')
    
    # أمر حذف نماذج 70
    subparsers.add_parser('remove70', help='حذف النماذج ذات 70 ميزة')
    
    # أمر الحالة
    subparsers.add_parser('status', help='عرض حالة النظام')
    
    # أمر التشغيل السريع
    subparsers.add_parser('quickstart', help='تشغيل سريع للنظام')
    
    args = parser.parse_args()
    
    # إنشاء مدير النظام
    manager = SystemManager()
    
    # تنفيذ الأمر
    if args.command == 'train':
        manager.train_new_models(args.pairs, args.phase)
    elif args.command == 'retrain':
        manager.retrain_existing()
    elif args.command == 'learning':
        manager.start_continuous_learning()
    elif args.command == 'advanced':
        manager.start_advanced_learning()
    elif args.command == 'server':
        manager.start_server()
    elif args.command == 'clean':
        manager.clean_old_models(args.days)
    elif args.command == 'remove70':
        manager.remove_70_feature_models()
    elif args.command == 'status':
        manager.status()
    elif args.command == 'quickstart':
        manager.quick_start()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()