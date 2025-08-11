# 📚 دليل النشر على Hostinger VPS

دليل شامل لنشر نظام التداول الآلي على Hostinger VPS خطوة بخطوة.

## 📋 المحتويات

1. [شراء وإعداد VPS من Hostinger](#1-شراء-وإعداد-vps-من-hostinger)
2. [الاتصال بالخادم](#2-الاتصال-بالخادم)
3. [إعداد البيئة الأساسية](#3-إعداد-البيئة-الأساسية)
4. [تثبيت المشروع](#4-تثبيت-المشروع)
5. [إعداد قاعدة البيانات](#5-إعداد-قاعدة-البيانات)
6. [تدريب النماذج](#6-تدريب-النماذج)
7. [إعداد الخدمات](#7-إعداد-الخدمات)
8. [المراقبة والصيانة](#8-المراقبة-والصيانة)

## 1. شراء وإعداد VPS من Hostinger

### الخطوات:

1. **اذهب إلى** [Hostinger VPS](https://www.hostinger.com/vps-hosting)

2. **اختر خطة VPS**:
   - الحد الأدنى: VPS 2 (2 vCPU, 4 GB RAM)
   - الموصى به: VPS 4 (4 vCPU, 8 GB RAM)

3. **اختر نظام التشغيل**:
   - Ubuntu 22.04 LTS (64-bit)

4. **اختر موقع الخادم**:
   - اختر الأقرب لخادم البروكر
   - أوروبا أو أمريكا عادة الأفضل

5. **أكمل عملية الشراء**

### بعد الشراء:

ستحصل على:
- عنوان IP للخادم
- اسم المستخدم (عادة root)
- كلمة المرور

## 2. الاتصال بالخادم

### على Windows:

1. **استخدم PuTTY**:
   - حمل من: https://www.putty.org/
   - أدخل IP في Host Name
   - اضغط Open

2. **أو استخدم Windows Terminal**:
```bash
ssh root@your_server_ip
```

### على Mac/Linux:
```bash
ssh root@your_server_ip
```

## 3. إعداد البيئة الأساسية

### تحديث النظام:
```bash
apt update && apt upgrade -y
```

### إنشاء مستخدم جديد (أكثر أماناً):
```bash
adduser trader
usermod -aG sudo trader
su - trader
```

### تثبيت الأدوات الأساسية:
```bash
sudo apt install -y \
    python3.9 \
    python3-pip \
    python3-venv \
    git \
    screen \
    htop \
    supervisor \
    nginx \
    ufw
```

## 4. تثبيت المشروع

### استنساخ المشروع:
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading
```

### إنشاء البيئة الافتراضية:
```bash
python3 -m venv venv
source venv/bin/activate
```

### تثبيت المكتبات:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### إعداد الملفات:
```bash
cp .env.example .env
nano .env
```

أضف بياناتك:
```env
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 5. إعداد قاعدة البيانات

### إنشاء المجلدات:
```bash
mkdir -p data/{raw,processed,models}
mkdir -p logs
```

### جمع البيانات الأولية:
```bash
source venv/bin/activate
python main.py collect
```

## 6. تدريب النماذج

### تدريب النماذج لأول مرة:
```bash
python train_models.py
```

هذا قد يستغرق 30-60 دقيقة حسب كمية البيانات.

## 7. إعداد الخدمات

### إنشاء خدمة systemd للبوت:

```bash
sudo nano /etc/systemd/system/forex-bot.service
```

أضف:
```ini
[Unit]
Description=Forex ML Trading Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/forex-ml-trading
Environment="PATH=/home/trader/forex-ml-trading/venv/bin"
ExecStart=/home/trader/forex-ml-trading/venv/bin/python main.py trade
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### تفعيل وبدء الخدمة:
```bash
sudo systemctl daemon-reload
sudo systemctl enable forex-bot
sudo systemctl start forex-bot
```

### إعداد خدمة المراقبة:
```bash
sudo nano /etc/systemd/system/forex-monitor.service
```

أضف نفس المحتوى مع تغيير:
```ini
ExecStart=/home/trader/forex-ml-trading/venv/bin/python main.py monitor
```

### إعداد لوحة التحكم (اختياري):
```bash
sudo nano /etc/systemd/system/forex-dashboard.service
```

```ini
[Unit]
Description=Forex Trading Dashboard
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/forex-ml-trading
Environment="PATH=/home/trader/forex-ml-trading/venv/bin"
ExecStart=/home/trader/forex-ml-trading/venv/bin/streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

## 8. المراقبة والصيانة

### فحص حالة الخدمات:
```bash
sudo systemctl status forex-bot
sudo systemctl status forex-monitor
```

### عرض السجلات:
```bash
# سجلات النظام
journalctl -u forex-bot -f

# سجلات التطبيق
tail -f ~/forex-ml-trading/logs/trader.log
```

### إعداد جدار الحماية:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8501  # للوحة التحكم
sudo ufw enable
```

### إعداد النسخ الاحتياطي التلقائي:

1. **إنشاء سكريبت النسخ الاحتياطي**:
```bash
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR=~/backups
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# نسخ قاعدة البيانات
cp ~/forex-ml-trading/data/trading.db $BACKUP_DIR/trading_$DATE.db

# نسخ النماذج
tar -czf $BACKUP_DIR/models_$DATE.tar.gz ~/forex-ml-trading/data/models/

# حذف النسخ القديمة (أكثر من 7 أيام)
find $BACKUP_DIR -mtime +7 -delete
```

2. **جعل السكريبت قابل للتنفيذ**:
```bash
chmod +x ~/backup.sh
```

3. **إضافة إلى crontab**:
```bash
crontab -e
```

أضف:
```
0 2 * * * /home/trader/backup.sh
```

### مراقبة الأداء:

1. **استخدام htop**:
```bash
htop
```

2. **مراقبة استخدام القرص**:
```bash
df -h
```

3. **مراقبة الذاكرة**:
```bash
free -h
```

## 🔧 نصائح للأداء الأمثل

### 1. تحسين Python:
```bash
# تثبيت مكتبات محسنة
pip install numpy --upgrade --force-reinstall
```

### 2. إعداد Swap (إذا كانت الذاكرة محدودة):
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. مراقبة الاتصال:
```bash
# اختبار سرعة الإنترنت
curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python3 -
```

## 🚨 استكشاف الأخطاء الشائعة

### 1. خطأ في الاتصال بـ MT5:
- تحقق من أن MT5 يسمح بالاتصالات الخارجية
- تأكد من صحة بيانات الحساب
- جرب استخدام خادم MT5 مختلف

### 2. نفاد الذاكرة:
- قلل عدد الأزواج المتداولة
- قلل حجم البيانات التاريخية
- فعل ملف Swap

### 3. توقف الخدمة:
```bash
# إعادة تشغيل الخدمة
sudo systemctl restart forex-bot

# فحص السجلات
journalctl -u forex-bot -n 100
```

## 📱 إعداد تطبيق Hostinger للمراقبة

1. حمل تطبيق Hostinger من متجر التطبيقات
2. سجل دخول بحسابك
3. يمكنك مراقبة:
   - استخدام CPU والذاكرة
   - حالة الخادم
   - إعادة تشغيل الخادم عند الحاجة

## 🔄 التحديثات الدورية

### تحديث الكود:
```bash
cd ~/forex-ml-trading
git pull origin main
pip install -r requirements.txt
sudo systemctl restart forex-bot
```

### إعادة تدريب النماذج (شهرياً):
```bash
python train_models.py
sudo systemctl restart forex-bot
```

## 📞 الدعم الفني

### دعم Hostinger:
- الدردشة المباشرة 24/7
- التذاكر عبر لوحة التحكم
- قاعدة المعرفة

### مجتمع المشروع:
- GitHub Issues
- Telegram Group

---

**ملاحظة**: احرص على حفظ نسخة من هذا الدليل وبيانات الوصول في مكان آمن!