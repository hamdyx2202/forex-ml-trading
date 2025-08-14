#!/usr/bin/env python3
"""
Auto Restart System
نظام إعادة التشغيل التلقائي
"""

import subprocess
import time
import os
import json
from datetime import datetime
from pathlib import Path
import logging

# إعداد logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_restart.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# المسارات
BASE_DIR = Path(__file__).parent.parent
VENV_PATH = BASE_DIR / "venv_pro" / "bin" / "activate"
STATE_FILE = BASE_DIR / "web_dashboard" / "system_state.json"

# الأنظمة المراقبة
SYSTEMS = {
    'server': {
        'process_name': 'mt5_bridge_server_advanced.py',
        'command': ['python3', 'src/mt5_bridge_server_advanced.py'],
        'cwd': str(BASE_DIR),
        'check_type': 'process'
    },
    'advanced_learner': {
        'screen_name': 'advanced_unified',
        'command': ['screen', '-dmS', 'advanced_unified', 'python3', 'src/advanced_learner_unified.py'],
        'cwd': str(BASE_DIR),
        'check_type': 'screen'
    },
    'continuous_learner': {
        'screen_name': 'continuous_unified',
        'command': ['screen', '-dmS', 'continuous_unified', 'python3', 'src/continuous_learner_unified.py'],
        'cwd': str(BASE_DIR),
        'check_type': 'screen'
    }
}

class AutoRestartManager:
    def __init__(self):
        self.state = self.load_state()
        self.restart_counts = {}
        self.max_restarts_per_hour = 5
        
    def load_state(self):
        """تحميل حالة النظام"""
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {'systems': {}, 'last_check': None}
    
    def save_state(self):
        """حفظ حالة النظام"""
        self.state['last_check'] = datetime.now().isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def check_process(self, process_name):
        """فحص العملية"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', process_name],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except Exception as e:
            logger.error(f"Error checking process {process_name}: {e}")
            return False
    
    def check_screen(self, screen_name):
        """فحص screen session"""
        try:
            result = subprocess.run(
                ['screen', '-ls'],
                capture_output=True,
                text=True
            )
            return screen_name in result.stdout
        except Exception as e:
            logger.error(f"Error checking screen {screen_name}: {e}")
            return False
    
    def check_port(self, port):
        """فحص المنفذ"""
        try:
            result = subprocess.run(
                ['netstat', '-tuln'],
                capture_output=True,
                text=True
            )
            return f':{port}' in result.stdout
        except:
            # محاولة بديلة
            try:
                result = subprocess.run(
                    ['ss', '-tuln'],
                    capture_output=True,
                    text=True
                )
                return f':{port}' in result.stdout
            except Exception as e:
                logger.error(f"Error checking port {port}: {e}")
                return False
    
    def restart_system(self, system_name, config):
        """إعادة تشغيل النظام"""
        logger.info(f"Restarting {system_name}...")
        
        # التحقق من عدد مرات إعادة التشغيل
        current_hour = datetime.now().hour
        if system_name not in self.restart_counts:
            self.restart_counts[system_name] = {'hour': current_hour, 'count': 0}
        
        if self.restart_counts[system_name]['hour'] != current_hour:
            # ساعة جديدة، إعادة تعيين العداد
            self.restart_counts[system_name] = {'hour': current_hour, 'count': 0}
        
        if self.restart_counts[system_name]['count'] >= self.max_restarts_per_hour:
            logger.error(f"{system_name} exceeded max restarts per hour!")
            self.send_alert(f"System {system_name} is failing repeatedly!")
            return False
        
        try:
            # إيقاف النظام أولاً
            if config['check_type'] == 'process':
                subprocess.run(['pkill', '-f', config['process_name']], check=False)
                time.sleep(2)
            elif config['check_type'] == 'screen':
                subprocess.run(['screen', '-X', '-S', config['screen_name'], 'quit'], check=False)
                time.sleep(2)
            
            # بدء النظام
            subprocess.run(config['command'], cwd=config['cwd'], check=True)
            
            self.restart_counts[system_name]['count'] += 1
            
            # تسجيل إعادة التشغيل
            if system_name not in self.state['systems']:
                self.state['systems'][system_name] = {}
            
            self.state['systems'][system_name]['last_restart'] = datetime.now().isoformat()
            self.state['systems'][system_name]['restart_count'] = \
                self.state['systems'][system_name].get('restart_count', 0) + 1
            
            logger.info(f"Successfully restarted {system_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart {system_name}: {e}")
            return False
    
    def check_system_health(self):
        """فحص صحة النظام"""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # فحص استخدام الموارد
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                health['issues'].append(f"High CPU usage: {cpu_percent}%")
                health['healthy'] = False
            
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health['issues'].append(f"High memory usage: {memory.percent}%")
                health['healthy'] = False
            
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                health['issues'].append(f"Low disk space: {disk.percent}% used")
                health['healthy'] = False
                
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
        
        return health
    
    def send_alert(self, message):
        """إرسال تنبيه"""
        logger.warning(f"ALERT: {message}")
        
        # يمكن إضافة إرسال email أو Telegram هنا
        alert_file = BASE_DIR / "logs" / "alerts.log"
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} | {message}\n")
    
    def monitor_and_restart(self):
        """المراقبة وإعادة التشغيل"""
        logger.info("Starting auto-restart monitor...")
        
        while True:
            try:
                # فحص صحة النظام
                health = self.check_system_health()
                if not health['healthy']:
                    for issue in health['issues']:
                        self.send_alert(issue)
                
                # فحص الأنظمة
                for system_name, config in SYSTEMS.items():
                    is_running = False
                    
                    if config['check_type'] == 'process':
                        is_running = self.check_process(config['process_name'])
                    elif config['check_type'] == 'screen':
                        is_running = self.check_screen(config['screen_name'])
                    
                    if not is_running:
                        logger.warning(f"{system_name} is not running!")
                        self.send_alert(f"{system_name} stopped - attempting restart")
                        
                        if self.restart_system(system_name, config):
                            self.send_alert(f"{system_name} restarted successfully")
                        else:
                            self.send_alert(f"Failed to restart {system_name}")
                    else:
                        logger.debug(f"{system_name} is running")
                
                # فحص المنافذ
                if not self.check_port(5000):
                    self.send_alert("Server port 5000 is not listening!")
                
                if not self.check_port(8080):
                    logger.warning("Dashboard port 8080 is not listening")
                
                # حفظ الحالة
                self.save_state()
                
                # الانتظار 5 دقائق
                time.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("Auto-restart monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)  # انتظار دقيقة في حالة الخطأ

def main():
    """النقطة الرئيسية"""
    manager = AutoRestartManager()
    
    # التحقق من وجود مجلد logs
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # التحقق من البيئة الافتراضية
    if not VENV_PATH.exists():
        logger.error(f"Virtual environment not found at {VENV_PATH}")
        logger.info("Trying alternative path...")
        alt_venv = BASE_DIR / "venv" / "bin" / "activate"
        if alt_venv.exists():
            logger.info("Found alternative venv")
        else:
            logger.error("No virtual environment found!")
    
    # بدء المراقبة
    manager.monitor_and_restart()

if __name__ == "__main__":
    main()