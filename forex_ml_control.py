#!/usr/bin/env python3
"""
Forex ML Trading System Control Center
مركز التحكم في نظام التداول
"""

import sys
import os
import subprocess
import psutil
import time
from pathlib import Path
from datetime import datetime
import signal
import json
from loguru import logger

class ForexMLControl:
    """مركز التحكم الرئيسي للنظام"""
    
    def __init__(self):
        self.processes = {}
        self.config_file = "system_config.json"
        self.load_config()
        
    def load_config(self):
        """تحميل الإعدادات"""
        default_config = {
            "server_port": 5000,
            "venv_path": "venv_pro",
            "pairs": ["EURUSD", "GBPUSD", "XAUUSD", "US30"],
            "timeframes": ["H1", "H4"]
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
            
    def save_config(self):
        """حفظ الإعدادات"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def stop_all(self):
        """إيقاف جميع العمليات"""
        print("🛑 إيقاف جميع العمليات...")
        
        # قتل العمليات بالاسم
        processes_to_kill = [
            "python.*mt5_bridge_server",
            "python.*continuous_learner",
            "python.*automated_training",
            "python.*integrated_training"
        ]
        
        for proc_pattern in processes_to_kill:
            try:
                # استخدام pkill
                subprocess.run(["pkill", "-f", proc_pattern], capture_output=True)
            except:
                pass
                
        # قتل العمليات المحفوظة
        for name, pid in self.processes.items():
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"✅ تم إيقاف {name} (PID: {pid})")
            except:
                pass
                
        self.processes.clear()
        print("✅ تم إيقاف جميع العمليات")
        
    def start_server(self):
        """بدء السيرفر"""
        print("🚀 بدء سيرفر MT5...")
        
        # التأكد من البيئة الافتراضية
        venv_python = f"{self.config['venv_path']}/bin/python"
        
        if not os.path.exists(venv_python):
            print("❌ البيئة الافتراضية غير موجودة!")
            return False
            
        # بدء السيرفر
        try:
            process = subprocess.Popen(
                [venv_python, "src/mt5_bridge_server_advanced.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            self.processes['server'] = process.pid
            print(f"✅ السيرفر يعمل على المنفذ {self.config['server_port']} (PID: {process.pid})")
            
            # انتظار للتأكد من البدء
            time.sleep(3)
            
            # التحقق من أن السيرفر يعمل
            if process.poll() is None:
                print("✅ السيرفر يعمل بنجاح")
                return True
            else:
                print("❌ فشل بدء السيرفر")
                return False
                
        except Exception as e:
            print(f"❌ خطأ في بدء السيرفر: {str(e)}")
            return False
            
    def start_training(self, mode="basic"):
        """بدء التدريب"""
        print(f"🎯 بدء التدريب - النمط: {mode}")
        
        venv_python = f"{self.config['venv_path']}/bin/python"
        
        scripts = {
            "basic": "advanced_learner_unified_sltp.py",
            "continuous": "continuous_learner_unified_sltp.py",
            "integrated": "integrated_training_sltp.py",
            "auto": "automated_training_sltp.py"
        }
        
        script = scripts.get(mode, scripts["basic"])
        
        try:
            process = subprocess.Popen(
                [venv_python, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            self.processes[f'training_{mode}'] = process.pid
            print(f"✅ بدء التدريب {mode} (PID: {process.pid})")
            
            return True
            
        except Exception as e:
            print(f"❌ خطأ في بدء التدريب: {str(e)}")
            return False
            
    def train_pair(self, pair, timeframe="H1"):
        """تدريب زوج واحد"""
        print(f"📊 تدريب {pair} {timeframe}...")
        
        venv_python = f"{self.config['venv_path']}/bin/python"
        
        # سكريبت بسيط للتدريب
        train_script = f"""
import sys
sys.path.append('.')
from advanced_learner_unified_sltp import AdvancedLearnerWithSLTP

learner = AdvancedLearnerWithSLTP()
success = learner.train_model_with_sltp("{pair}", "{timeframe}")
print("✅ نجح التدريب" if success else "❌ فشل التدريب")
"""
        
        # حفظ السكريبت مؤقتاً
        with open("temp_train.py", "w") as f:
            f.write(train_script)
            
        try:
            result = subprocess.run(
                [venv_python, "temp_train.py"],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            if result.stderr:
                print(f"⚠️ تحذيرات: {result.stderr}")
                
        except Exception as e:
            print(f"❌ خطأ: {str(e)}")
            
        finally:
            # حذف الملف المؤقت
            if os.path.exists("temp_train.py"):
                os.remove("temp_train.py")
                
    def status(self):
        """عرض حالة النظام"""
        print("\n" + "="*50)
        print("📊 حالة النظام")
        print("="*50)
        
        # فحص العمليات
        for name, pid in self.processes.items():
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    cpu = process.cpu_percent(interval=0.1)
                    memory = process.memory_info().rss / 1024 / 1024  # MB
                    print(f"✅ {name}: يعمل (PID: {pid}, CPU: {cpu:.1f}%, RAM: {memory:.1f}MB)")
                else:
                    print(f"❌ {name}: متوقف")
            except:
                print(f"❌ {name}: غير موجود")
                
        # فحص المنافذ
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.config['server_port']))
            sock.close()
            
            if result == 0:
                print(f"✅ السيرفر يستمع على المنفذ {self.config['server_port']}")
            else:
                print(f"❌ المنفذ {self.config['server_port']} غير مستخدم")
        except:
            pass
            
        # إحصائيات النماذج
        models_dir = Path("models/unified_sltp")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*.pkl")))
            print(f"📈 عدد النماذج المدربة: {model_count}")
            
        print("="*50 + "\n")
        
    def quick_start(self):
        """بدء سريع للنظام"""
        print("🚀 بدء سريع للنظام...")
        
        # إيقاف أي عمليات سابقة
        self.stop_all()
        time.sleep(2)
        
        # بدء السيرفر
        if self.start_server():
            time.sleep(3)
            
            # بدء التدريب الآلي
            self.start_training("auto")
            time.sleep(2)
            
            print("✅ النظام يعمل بالكامل!")
            self.status()
        else:
            print("❌ فشل بدء النظام")
            
    def menu(self):
        """القائمة التفاعلية"""
        while True:
            print("\n" + "="*50)
            print("🎛️  مركز التحكم - Forex ML Trading")
            print("="*50)
            print("1. إيقاف جميع العمليات")
            print("2. بدء السيرفر")
            print("3. بدء التدريب الأساسي")
            print("4. بدء التعلم المستمر")
            print("5. بدء النظام الآلي الكامل")
            print("6. تدريب زوج محدد")
            print("7. عرض حالة النظام")
            print("8. بدء سريع (recommended)")
            print("0. خروج")
            print("="*50)
            
            choice = input("اختر رقم: ").strip()
            
            if choice == "1":
                self.stop_all()
            elif choice == "2":
                self.start_server()
            elif choice == "3":
                self.start_training("basic")
            elif choice == "4":
                self.start_training("continuous")
            elif choice == "5":
                self.start_training("auto")
            elif choice == "6":
                pair = input("أدخل الزوج (مثال: EURUSD): ").strip().upper()
                timeframe = input("أدخل الإطار الزمني (H1/H4/M15): ").strip().upper()
                self.train_pair(pair, timeframe)
            elif choice == "7":
                self.status()
            elif choice == "8":
                self.quick_start()
            elif choice == "0":
                print("👋 إلى اللقاء!")
                self.stop_all()
                break
            else:
                print("❌ اختيار غير صحيح")
                
            input("\nاضغط Enter للمتابعة...")


def main():
    """الدالة الرئيسية"""
    control = ForexMLControl()
    
    # معالجة أوامر سطر الأوامر
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stop":
            control.stop_all()
        elif command == "start":
            control.quick_start()
        elif command == "server":
            control.start_server()
        elif command == "train":
            if len(sys.argv) > 2:
                control.start_training(sys.argv[2])
            else:
                control.start_training()
        elif command == "status":
            control.status()
        elif command == "train-pair":
            if len(sys.argv) > 3:
                control.train_pair(sys.argv[2], sys.argv[3])
            else:
                print("❌ يجب تحديد الزوج والإطار الزمني")
        else:
            print(f"❌ أمر غير معروف: {command}")
            print("\nالأوامر المتاحة:")
            print("  stop     - إيقاف جميع العمليات")
            print("  start    - بدء سريع للنظام")
            print("  server   - بدء السيرفر فقط")
            print("  train    - بدء التدريب (basic/continuous/integrated/auto)")
            print("  status   - عرض حالة النظام")
            print("  train-pair SYMBOL TIMEFRAME - تدريب زوج محدد")
    else:
        # القائمة التفاعلية
        control.menu()


if __name__ == "__main__":
    main()