#!/usr/bin/env python3
"""
Forex ML Trading Web Dashboard
واجهة ويب للتحكم والمراقبة
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timedelta
import subprocess
import psutil
import json
import os
import sqlite3
import time
import threading
from pathlib import Path
from loguru import logger

# إعداد Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# المسارات
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "forex_data.db"
TRADES_LOG = MODELS_DIR / "unified" / "trades_log.json"

# المستخدمون (في الإنتاج استخدم قاعدة بيانات)
USERS = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin'
    },
    'viewer': {
        'password': generate_password_hash('viewer123'),
        'role': 'viewer'
    }
}

class User(UserMixin):
    def __init__(self, username, role):
        self.id = username
        self.role = role

@login_manager.user_loader
def load_user(username):
    if username in USERS:
        return User(username, USERS[username]['role'])
    return None

# حالة النظام
system_status = {
    'server': 'unknown',
    'advanced_learner': 'unknown',
    'continuous_learner': 'unknown',
    'last_check': None
}

def check_process_status(process_name):
    """فحص حالة العملية"""
    try:
        result = subprocess.run(['pgrep', '-f', process_name], 
                              capture_output=True, text=True)
        return 'running' if result.stdout.strip() else 'stopped'
    except:
        return 'error'

def check_screen_status(screen_name):
    """فحص حالة screen"""
    try:
        result = subprocess.run(['screen', '-ls'], 
                              capture_output=True, text=True)
        return 'running' if screen_name in result.stdout else 'stopped'
    except:
        return 'error'

def update_system_status():
    """تحديث حالة النظام"""
    global system_status
    
    # فحص السيرفر
    system_status['server'] = check_process_status('mt5_bridge_server_advanced.py')
    
    # فحص أنظمة التعلم
    system_status['advanced_learner'] = check_screen_status('advanced_unified')
    system_status['continuous_learner'] = check_screen_status('continuous_unified')
    
    system_status['last_check'] = datetime.now().isoformat()
    
    # بث التحديث للعملاء
    socketio.emit('status_update', system_status)

def continuous_monitoring():
    """مراقبة مستمرة"""
    while True:
        update_system_status()
        time.sleep(5)  # كل 5 ثواني

# بدء المراقبة في thread منفصل
monitoring_thread = threading.Thread(target=continuous_monitoring, daemon=True)
monitoring_thread.start()

# Routes
@app.route('/')
@login_required
def index():
    """الصفحة الرئيسية"""
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """صفحة تسجيل الدخول"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and check_password_hash(USERS[username]['password'], password):
            user = User(username, USERS[username]['role'])
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """تسجيل الخروج"""
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/status')
@login_required
def get_status():
    """الحصول على حالة النظام"""
    update_system_status()
    return jsonify(system_status)

@app.route('/api/stats')
@login_required
def get_stats():
    """الحصول على الإحصائيات"""
    stats = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'models_count': len(list(MODELS_DIR.glob('**/*.pkl'))),
        'uptime': get_system_uptime()
    }
    
    # إحصائيات التداول
    if TRADES_LOG.exists():
        with open(TRADES_LOG) as f:
            trades_data = json.load(f)
            stats['total_trades'] = len(trades_data.get('trades', []))
            
            # حساب معدل النجاح
            if stats['total_trades'] > 0:
                wins = sum(1 for t in trades_data['trades'] if t.get('result') == 'win')
                stats['win_rate'] = (wins / stats['total_trades']) * 100
            else:
                stats['win_rate'] = 0
    else:
        stats['total_trades'] = 0
        stats['win_rate'] = 0
    
    return jsonify(stats)

@app.route('/api/signals/latest')
@login_required
def get_latest_signals():
    """الحصول على آخر الإشارات"""
    # قراءة من السجلات أو قاعدة البيانات
    signals = []
    
    # هذا مثال - اقرأ من المصدر الفعلي
    log_file = LOGS_DIR / "server.log"
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()[-100:]  # آخر 100 سطر
            for line in lines:
                if 'signal' in line.lower() and ('BUY' in line or 'SELL' in line):
                    # استخراج معلومات الإشارة
                    try:
                        # تحليل السطر واستخراج البيانات
                        signals.append({
                            'time': datetime.now().isoformat(),
                            'symbol': 'EURUSD',  # استخرج من السطر
                            'signal': 'BUY',     # استخرج من السطر
                            'confidence': 0.85    # استخرج من السطر
                        })
                    except:
                        pass
    
    return jsonify(signals[-20:])  # آخر 20 إشارة

@app.route('/api/control/<action>', methods=['POST'])
@login_required
def control_system(action):
    """التحكم في النظام"""
    if current_user.role == 'viewer':
        return jsonify({'error': 'Insufficient permissions'}), 403
    
    result = {'success': False, 'message': ''}
    
    try:
        if action == 'restart_server':
            # إيقاف السيرفر
            subprocess.run(['pkill', '-f', 'mt5_bridge_server_advanced.py'])
            time.sleep(2)
            # تشغيل السيرفر
            subprocess.Popen(['python3', 'src/mt5_bridge_server_advanced.py'], 
                           cwd=str(BASE_DIR))
            result = {'success': True, 'message': 'Server restarted'}
            
        elif action == 'restart_advanced':
            # إعادة تشغيل advanced learner
            subprocess.run(['screen', '-X', '-S', 'advanced_unified', 'quit'])
            time.sleep(2)
            subprocess.run(['screen', '-dmS', 'advanced_unified', 
                          'python3', 'src/advanced_learner_unified.py'], 
                         cwd=str(BASE_DIR))
            result = {'success': True, 'message': 'Advanced learner restarted'}
            
        elif action == 'restart_continuous':
            # إعادة تشغيل continuous learner
            subprocess.run(['screen', '-X', '-S', 'continuous_unified', 'quit'])
            time.sleep(2)
            subprocess.run(['screen', '-dmS', 'continuous_unified', 
                          'python3', 'src/continuous_learner_unified.py'], 
                         cwd=str(BASE_DIR))
            result = {'success': True, 'message': 'Continuous learner restarted'}
            
        elif action == 'emergency_stop':
            # إيقاف كل شيء
            subprocess.run(['pkill', '-f', 'mt5_bridge_server_advanced.py'])
            subprocess.run(['screen', '-X', '-S', 'advanced_unified', 'quit'])
            subprocess.run(['screen', '-X', '-S', 'continuous_unified', 'quit'])
            result = {'success': True, 'message': 'All systems stopped'}
            
        elif action == 'restart_all':
            # إعادة تشغيل كل شيء
            control_system('emergency_stop')
            time.sleep(3)
            control_system('restart_server')
            control_system('restart_advanced')
            control_system('restart_continuous')
            result = {'success': True, 'message': 'All systems restarted'}
            
        elif action == 'backup_now':
            # نسخ احتياطي فوري
            backup_script = BASE_DIR / "daily_maintenance.sh"
            if backup_script.exists():
                subprocess.run(['bash', str(backup_script)])
                result = {'success': True, 'message': 'Backup completed'}
            else:
                result = {'success': False, 'message': 'Backup script not found'}
                
    except Exception as e:
        result = {'success': False, 'message': str(e)}
    
    # تسجيل العملية
    logger.info(f"Control action: {action} by {current_user.id} - {result}")
    
    return jsonify(result)

@app.route('/api/logs/<log_type>')
@login_required
def get_logs(log_type):
    """الحصول على السجلات"""
    logs = []
    
    if log_type == 'server':
        log_file = LOGS_DIR / "server.log"
    elif log_type == 'error':
        log_file = LOGS_DIR / "error.log"
    else:
        return jsonify({'error': 'Invalid log type'}), 400
    
    if log_file.exists():
        with open(log_file) as f:
            # آخر 1000 سطر
            lines = f.readlines()[-1000:]
            for line in lines:
                try:
                    # تحليل السطر
                    parts = line.split('|')
                    if len(parts) >= 3:
                        logs.append({
                            'time': parts[0].strip(),
                            'level': parts[1].strip(),
                            'message': '|'.join(parts[2:]).strip()
                        })
                except:
                    logs.append({
                        'time': '',
                        'level': 'INFO',
                        'message': line.strip()
                    })
    
    return jsonify(logs[-200:])  # آخر 200 سطر

def get_system_uptime():
    """حساب وقت التشغيل"""
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    days = uptime.days
    hours = uptime.seconds // 3600
    minutes = (uptime.seconds % 3600) // 60
    return f"{days}d {hours}h {minutes}m"

# WebSocket events
@socketio.on('connect')
@login_required
def handle_connect():
    """عند الاتصال"""
    emit('connected', {'data': 'Connected to dashboard'})
    update_system_status()

@socketio.on('request_update')
@login_required
def handle_update_request():
    """طلب تحديث"""
    update_system_status()

if __name__ == '__main__':
    # تشغيل في وضع التطوير
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)