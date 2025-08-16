#!/usr/bin/env python3
"""
🚀 نظام التشغيل الموحد للنظام المتقدم
يدمج جميع المكونات الأساسية في ملف واحد سهل التشغيل
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from colorama import init, Fore, Style
import threading

init()

class AdvancedSystemRunner:
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def print_header(self):
        """طباعة رأس النظام"""
        print(f"{Fore.CYAN}{'='*80}")
        print(f"🚀 نظام التداول الآلي المتقدم - الإصدار الكامل")
        print(f"📊 دقة مستهدفة: 95%+ | 🤖 6 نماذج AI | 📈 217+ مؤشر")
        print(f"{'='*80}{Style.RESET_ALL}\n")
    
    def check_requirements(self):
        """فحص المتطلبات الأساسية"""
        print(f"{Fore.YELLOW}🔍 فحص المتطلبات...{Style.RESET_ALL}")
        
        # فحص Python
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print(f"{Fore.RED}❌ يجب تثبيت Python 3.8 أو أحدث{Style.RESET_ALL}")
            return False
        print(f"{Fore.GREEN}✅ Python {python_version.major}.{python_version.minor}{Style.RESET_ALL}")
        
        # فحص المجلدات
        required_dirs = ['data', 'models', 'logs', 'results', 'src']
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                Path(dir_name).mkdir(parents=True, exist_ok=True)
                print(f"{Fore.GREEN}✅ تم إنشاء مجلد {dir_name}{Style.RESET_ALL}")
        
        # فحص قاعدة البيانات
        if not Path('data/forex_ml.db').exists():
            print(f"{Fore.YELLOW}⚠️ قاعدة البيانات غير موجودة - سيتم إنشاؤها عند أول استخدام{Style.RESET_ALL}")
        
        return True
    
    def install_requirements(self):
        """تثبيت المكتبات المطلوبة"""
        print(f"\n{Fore.YELLOW}📦 تثبيت المكتبات...{Style.RESET_ALL}")
        
        requirements = [
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'xgboost>=1.7.0',
            'tensorflow>=2.13.0',
            'lightgbm>=4.0.0',
            'ta>=0.10.0',
            'MetaTrader5>=5.0.45',
            'joblib>=1.3.0',
            'colorama>=0.4.6',
            'tqdm>=4.65.0',
            'streamlit>=1.25.0',
            'plotly>=5.15.0',
            'schedule>=1.2.0',
            'psutil>=5.9.0'
        ]
        
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"{Fore.GREEN}✅ {package.split('>=')[0]}{Style.RESET_ALL}")
            except:
                print(f"{Fore.RED}❌ فشل تثبيت {package}{Style.RESET_ALL}")
    
    def start_training_system(self):
        """بدء نظام التدريب المتقدم"""
        print(f"\n{Fore.CYAN}🤖 بدء نظام التدريب المتقدم...{Style.RESET_ALL}")
        
        if Path('train_advanced_complete.py').exists():
            self.processes['training'] = subprocess.Popen(
                [sys.executable, 'train_advanced_complete.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Fore.GREEN}✅ تم بدء التدريب المتقدم{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ ملف التدريب غير موجود{Style.RESET_ALL}")
    
    def start_continuous_training(self):
        """بدء نظام التدريب المستمر"""
        print(f"\n{Fore.CYAN}♻️ بدء نظام التدريب المستمر...{Style.RESET_ALL}")
        
        if Path('continuous_training_system.py').exists():
            self.processes['continuous'] = subprocess.Popen(
                [sys.executable, 'continuous_training_system.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Fore.GREEN}✅ تم بدء التدريب المستمر{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ ملف التدريب المستمر غير موجود{Style.RESET_ALL}")
    
    def start_prediction_server(self):
        """بدء خادم التنبؤات"""
        print(f"\n{Fore.CYAN}🔮 بدء خادم التنبؤات...{Style.RESET_ALL}")
        
        if Path('mt5_prediction_server.py').exists():
            self.processes['prediction'] = subprocess.Popen(
                [sys.executable, 'mt5_prediction_server.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Fore.GREEN}✅ تم بدء خادم التنبؤات{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ ملف خادم التنبؤات غير موجود{Style.RESET_ALL}")
    
    def start_trading_system(self):
        """بدء نظام التداول"""
        print(f"\n{Fore.CYAN}💰 بدء نظام التداول...{Style.RESET_ALL}")
        
        if Path('main.py').exists():
            self.processes['trading'] = subprocess.Popen(
                [sys.executable, 'main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Fore.GREEN}✅ تم بدء نظام التداول{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ ملف التداول الرئيسي غير موجود{Style.RESET_ALL}")
    
    def start_dashboard(self):
        """بدء لوحة التحكم"""
        print(f"\n{Fore.CYAN}📊 بدء لوحة التحكم...{Style.RESET_ALL}")
        
        if Path('dashboard.py').exists():
            self.processes['dashboard'] = subprocess.Popen(
                ['streamlit', 'run', 'dashboard.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Fore.GREEN}✅ تم بدء لوحة التحكم على http://localhost:8501{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ ملف لوحة التحكم غير موجود{Style.RESET_ALL}")
    
    def monitor_processes(self):
        """مراقبة العمليات"""
        while self.running:
            time.sleep(5)
            for name, process in self.processes.items():
                if process and process.poll() is not None:
                    print(f"{Fore.YELLOW}⚠️ توقفت عملية {name}{Style.RESET_ALL}")
    
    def stop_all(self):
        """إيقاف جميع العمليات"""
        print(f"\n{Fore.YELLOW}⏹️ إيقاف جميع العمليات...{Style.RESET_ALL}")
        self.running = False
        
        for name, process in self.processes.items():
            if process:
                process.terminate()
                print(f"{Fore.GREEN}✅ تم إيقاف {name}{Style.RESET_ALL}")
    
    def show_menu(self):
        """عرض القائمة الرئيسية"""
        print(f"\n{Fore.CYAN}📋 القائمة الرئيسية:{Style.RESET_ALL}")
        print("1. التدريب الكامل (مرة واحدة)")
        print("2. التدريب المستمر (24/7)")
        print("3. التداول الآلي")
        print("4. النظام الكامل (الكل معاً)")
        print("5. لوحة التحكم فقط")
        print("6. إيقاف كل شيء")
        print("0. خروج")
    
    def run(self):
        """تشغيل النظام"""
        self.print_header()
        
        if not self.check_requirements():
            return
        
        # السؤال عن تثبيت المكتبات
        install = input(f"\n{Fore.YELLOW}هل تريد تثبيت/تحديث المكتبات؟ (y/n): {Style.RESET_ALL}").strip().lower()
        if install == 'y':
            self.install_requirements()
        
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        while True:
            self.show_menu()
            choice = input(f"\n{Fore.CYAN}اختيارك: {Style.RESET_ALL}").strip()
            
            if choice == '1':
                # التدريب الكامل
                print(f"\n{Fore.YELLOW}🎯 وضع التدريب الكامل{Style.RESET_ALL}")
                print("سيتم تدريب جميع الأزواج بالنظام المتقدم")
                confirm = input("متابعة؟ (y/n): ").strip().lower()
                if confirm == 'y':
                    self.start_training_system()
                
            elif choice == '2':
                # التدريب المستمر
                print(f"\n{Fore.YELLOW}♻️ وضع التدريب المستمر{Style.RESET_ALL}")
                print("سيعمل النظام 24/7 مع مراقبة وتحديث النماذج")
                confirm = input("متابعة؟ (y/n): ").strip().lower()
                if confirm == 'y':
                    self.start_continuous_training()
                
            elif choice == '3':
                # التداول الآلي
                print(f"\n{Fore.YELLOW}💰 وضع التداول الآلي{Style.RESET_ALL}")
                print("تأكد من وجود نماذج مدربة وفتح MT5")
                confirm = input("متابعة؟ (y/n): ").strip().lower()
                if confirm == 'y':
                    self.start_prediction_server()
                    time.sleep(2)
                    self.start_trading_system()
                
            elif choice == '4':
                # النظام الكامل
                print(f"\n{Fore.YELLOW}🚀 تشغيل النظام الكامل{Style.RESET_ALL}")
                print("سيتم تشغيل:")
                print("• التدريب المستمر")
                print("• خادم التنبؤات")
                print("• نظام التداول")
                print("• لوحة التحكم")
                confirm = input("\nمتابعة؟ (y/n): ").strip().lower()
                if confirm == 'y':
                    self.start_continuous_training()
                    time.sleep(2)
                    self.start_prediction_server()
                    time.sleep(2)
                    self.start_trading_system()
                    time.sleep(2)
                    self.start_dashboard()
                
            elif choice == '5':
                # لوحة التحكم
                self.start_dashboard()
                
            elif choice == '6':
                # إيقاف الكل
                self.stop_all()
                
            elif choice == '0':
                # خروج
                self.stop_all()
                print(f"\n{Fore.GREEN}👋 وداعاً!{Style.RESET_ALL}")
                break
            
            else:
                print(f"{Fore.RED}❌ اختيار غير صحيح{Style.RESET_ALL}")

def main():
    try:
        runner = AdvancedSystemRunner()
        runner.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️ تم الإيقاف بواسطة المستخدم{Style.RESET_ALL}")
        runner.stop_all()
    except Exception as e:
        print(f"{Fore.RED}❌ خطأ: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()