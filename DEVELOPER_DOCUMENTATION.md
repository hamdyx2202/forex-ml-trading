# 📘 دليل المطور الشامل - Forex ML Trading Bot

## 📋 فهرس المحتويات

1. [نظرة عامة على المشروع](#1-نظرة-عامة-على-المشروع)
2. [خريطة الملفات الكاملة](#2-خريطة-الملفات-الكاملة)
3. [دليل التشغيل على Hostinger VPS](#3-دليل-التشغيل-على-hostinger-vps)
4. [الصيانة والمتابعة اليومية](#4-الصيانة-والمتابعة-اليومية)
5. [دليل التحديث والنشر](#5-دليل-التحديث-والنشر)
6. [حل المشاكل الشائعة](#6-حل-المشاكل-الشائعة)
7. [الأمان والنسخ الاحتياطي](#7-الأمان-والنسخ-الاحتياطي)

---

## 1. نظرة عامة على المشروع

### الوصف:
نظام تداول آلي يستخدم التعلم الآلي للتنبؤ بحركة أزواج الفوركس والتداول تلقائياً عبر MetaTrader 5.

### المميزات الرئيسية:
- تعلم آلي متقدم (LightGBM + XGBoost)
- تعلم مستمر من كل صفقة
- إدارة مخاطر ذكية
- مراقبة 24/7 مع تنبيهات Telegram
- لوحة تحكم ويب

### التقنيات المستخدمة:
- **اللغة:** Python 3.9+
- **قاعدة البيانات:** SQLite
- **التعلم الآلي:** LightGBM, XGBoost, Scikit-learn
- **التداول:** MetaTrader5 Python API
- **واجهة المستخدم:** Streamlit
- **المراقبة:** Telegram Bot API

---

## 2. خريطة الملفات الكاملة

```
forex-ml-trading/
│
├── 📁 config/                    # ملفات الإعدادات
│   ├── config.json              # الإعدادات الرئيسية
│   ├── pairs.json               # إعدادات أزواج العملات
│   └── credentials.json         # بيانات الحسابات (يُنشأ من .env)
│
├── 📁 src/                      # الكود المصدري الأساسي
│   ├── data_collector.py        # جمع البيانات من MT5
│   ├── feature_engineer.py      # استخراج المؤشرات الفنية
│   ├── model_trainer.py         # تدريب نماذج ML
│   ├── predictor.py             # التنبؤ بالحركة
│   ├── trader.py                # تنفيذ الصفقات
│   ├── risk_manager.py          # إدارة المخاطر
│   ├── monitor.py               # المراقبة والتنبيهات
│   ├── advanced_learner.py      # التعلم من التاريخ
│   └── continuous_learner.py    # التعلم المستمر
│
├── 📁 data/                     # البيانات
│   ├── 📁 raw/                  # البيانات الخام (يُملأ تلقائياً)
│   ├── 📁 processed/            # البيانات المعالجة
│   ├── 📁 models/               # النماذج المدربة
│   ├── trading.db               # قاعدة البيانات الرئيسية
│   ├── learning_memory.json     # ذاكرة التعلم
│   ├── quality_criteria.json    # معايير الجودة
│   └── blacklisted_patterns.json # الأنماط المحظورة
│
├── 📁 logs/                     # سجلات النظام
│   ├── trader.log               # سجل التداول
│   ├── data_collector.log       # سجل جمع البيانات
│   ├── model_trainer.log        # سجل التدريب
│   ├── monitor.log              # سجل المراقبة
│   └── continuous_learning.log  # سجل التعلم المستمر
│
├── 📁 deployment/               # ملفات النشر
│   ├── setup_vps.sh            # سكريبت إعداد VPS
│   ├── deploy.sh               # سكريبت النشر
│   └── vps_setup_commands.txt  # أوامر الإعداد اليدوي
│
├── 📁 docs/                     # الوثائق
│   ├── HOSTINGER_DEPLOYMENT_GUIDE.md
│   ├── BEGINNER_SETUP_GUIDE.md
│   └── DEVELOPER_DOCUMENTATION.md
│
├── 📁 notebooks/                # Jupyter notebooks للتحليل
│   ├── data_analysis.ipynb
│   └── model_testing.ipynb
│
├── 📁 tests/                    # اختبارات الوحدة
│   └── test_*.py
│
├── 📄 main.py                   # نقطة البداية الرئيسية
├── 📄 train_models.py           # سكريبت تدريب النماذج
├── 📄 learn_from_history.py     # سكريبت التعلم من التاريخ
├── 📄 run_daily_analysis.py     # التحليل اليومي للفرص
├── 📄 auto_improve.py           # التحسين التلقائي
├── 📄 dashboard.py              # لوحة التحكم Streamlit
├── 📄 requirements.txt          # المكتبات المطلوبة
├── 📄 .env.example              # مثال لملف البيئة
├── 📄 .gitignore               # الملفات المتجاهلة
├── 📄 README.md                # الوثائق الرئيسية
└── 📄 LICENSE                  # الترخيص
```

### شرح الملفات الرئيسية:

#### ملفات الإعداد:
- **config.json**: جميع إعدادات النظام (أزواج، مخاطر، مؤشرات)
- **pairs.json**: تفاصيل كل زوج عملات (pip value, spread, etc)
- **.env**: بيانات حساسة (كلمات مرور، API keys)

#### الوحدات الأساسية:
- **data_collector.py**: يتصل بـ MT5 ويجمع البيانات التاريخية والحية
- **feature_engineer.py**: يحسب 50+ مؤشر فني (RSI, MACD, Bollinger, etc)
- **model_trainer.py**: يدرب نماذج LightGBM و XGBoost
- **predictor.py**: يتنبأ بحركة السعر (صعود/هبوط)
- **trader.py**: ينفذ الصفقات ويديرها
- **risk_manager.py**: يحسب أحجام الصفقات ويدير المخاطر
- **advanced_learner.py**: يحلل التاريخ ويجد أفضل الأنماط
- **continuous_learner.py**: يتعلم من كل صفقة ويحسن النظام

#### السكريبتات التنفيذية:
- **main.py**: النقطة الرئيسية (collect, train, trade, monitor)
- **train_models.py**: تدريب جميع النماذج
- **learn_from_history.py**: التعلم من 365 يوم ماضية
- **run_daily_analysis.py**: البحث عن فرص اليوم
- **auto_improve.py**: التحسين التلقائي اليومي

---

## 3. دليل التشغيل على Hostinger VPS

### المتطلبات:
- Hostinger VPS (الحد الأدنى VPS 2)
- Ubuntu 22.04 LTS
- 4GB RAM, 2 vCPU
- Python 3.9+

### خطوات التشغيل الكاملة:

#### 1. الاتصال بـ VPS:
```bash
# من Windows PowerShell أو Terminal
ssh root@YOUR_VPS_IP
# أدخل كلمة المرور
```

#### 2. الإعداد الأولي (مرة واحدة فقط):
```bash
# تحديث النظام
apt update && apt upgrade -y

# تثبيت المتطلبات
apt install -y python3.9 python3.9-dev python3-pip python3-venv git screen htop nano

# إنشاء مستخدم (أكثر أماناً من root)
adduser trader
usermod -aG sudo trader
su - trader
```

#### 3. تثبيت المشروع:
```bash
# استنساخ المشروع
cd ~
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading

# إنشاء البيئة الافتراضية
python3.9 -m venv venv
source venv/bin/activate

# تثبيت المكتبات
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. إعداد البيانات:
```bash
# نسخ ملف البيئة
cp .env.example .env

# تحرير البيانات
nano .env
```

أضف:
```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo
MT5_PATH=

TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

DATABASE_URL=sqlite:///data/trading.db
ENVIRONMENT=production
DEBUG=False
```

#### 5. الإعداد الأولي للبيانات:
```bash
# اختبار الاتصال
python main.py test

# جمع البيانات التاريخية (20-30 دقيقة)
python main.py collect

# التعلم من التاريخ (30-45 دقيقة)
python learn_from_history.py

# تدريب النماذج (45-60 دقيقة)
python train_models.py
```

#### 6. إنشاء خدمات systemd:

**خدمة التداول:**
```bash
sudo nano /etc/systemd/system/forex-trading.service
```

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
StandardOutput=append:/home/trader/forex-ml-trading/logs/systemd.log
StandardError=append:/home/trader/forex-ml-trading/logs/systemd-error.log

[Install]
WantedBy=multi-user.target
```

**خدمة المراقبة:**
```bash
sudo nano /etc/systemd/system/forex-monitor.service
```

```ini
[Unit]
Description=Forex Trading Monitor
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/forex-ml-trading
Environment="PATH=/home/trader/forex-ml-trading/venv/bin"
ExecStart=/home/trader/forex-ml-trading/venv/bin/python main.py monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**خدمة لوحة التحكم (اختياري):**
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

#### 7. تفعيل وبدء الخدمات:
```bash
# إعادة تحميل systemd
sudo systemctl daemon-reload

# تفعيل الخدمات
sudo systemctl enable forex-trading
sudo systemctl enable forex-monitor
sudo systemctl enable forex-dashboard  # اختياري

# بدء الخدمات
sudo systemctl start forex-trading
sudo systemctl start forex-monitor
sudo systemctl start forex-dashboard  # اختياري

# التحقق من الحالة
sudo systemctl status forex-trading
sudo systemctl status forex-monitor
```

#### 8. إعداد جدار الحماية:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8501  # للوحة التحكم
sudo ufw enable
```

#### 9. إعداد المهام المجدولة (cron):
```bash
crontab -e
```

أضف:
```cron
# تحليل يومي للفرص (9 صباحاً)
0 9 * * * cd /home/trader/forex-ml-trading && /home/trader/forex-ml-trading/venv/bin/python run_daily_analysis.py

# تحسين تلقائي (2 صباحاً)
0 2 * * * cd /home/trader/forex-ml-trading && /home/trader/forex-ml-trading/venv/bin/python auto_improve.py --once

# نسخ احتياطي (3 صباحاً)
0 3 * * * /home/trader/backup.sh

# تحديث البيانات (كل 6 ساعات)
0 */6 * * * cd /home/trader/forex-ml-trading && /home/trader/forex-ml-trading/venv/bin/python main.py collect

# إعادة تدريب النماذج (أول يوم من الشهر)
0 4 1 * * cd /home/trader/forex-ml-trading && /home/trader/forex-ml-trading/venv/bin/python train_models.py
```

---

## 4. الصيانة والمتابعة اليومية

### المهام اليومية:

#### 1. فحص حالة النظام:
```bash
# حالة الخدمات
sudo systemctl status forex-trading
sudo systemctl status forex-monitor

# استخدام الموارد
htop
df -h  # المساحة المتاحة
free -h  # الذاكرة المتاحة
```

#### 2. مراجعة السجلات:
```bash
# سجلات التداول
tail -f logs/trader.log

# سجلات النظام
journalctl -u forex-trading -f

# البحث عن أخطاء
grep ERROR logs/*.log
```

#### 3. فحص الأداء:
```bash
# تشغيل تحليل الفرص
python run_daily_analysis.py

# عرض إحصائيات الأداء
python -c "from src.risk_manager import RiskManager; rm = RiskManager(); print(rm.get_performance_stats())"
```

#### 4. النسخ الاحتياطي:
```bash
# سكريبت النسخ الاحتياطي
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR=~/backups
DATE=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR=~/forex-ml-trading

mkdir -p $BACKUP_DIR

# نسخ قاعدة البيانات
cp $PROJECT_DIR/data/trading.db $BACKUP_DIR/trading_$DATE.db

# نسخ النماذج
tar -czf $BACKUP_DIR/models_$DATE.tar.gz $PROJECT_DIR/data/models/

# نسخ السجلات
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz $PROJECT_DIR/logs/

# نسخ الإعدادات
cp $PROJECT_DIR/.env $BACKUP_DIR/env_$DATE.txt
cp -r $PROJECT_DIR/config $BACKUP_DIR/config_$DATE/

# حذف النسخ القديمة (أكثر من 7 أيام)
find $BACKUP_DIR -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
chmod +x ~/backup.sh
```

### المهام الأسبوعية:

1. **تحديث البيانات الكاملة:**
```bash
cd ~/forex-ml-trading
source venv/bin/activate
python main.py collect
```

2. **مراجعة التعلم:**
```bash
python -c "from src.continuous_learner import ContinuousLearner; cl = ContinuousLearner(); print(cl.get_learning_insights())"
```

3. **تنظيف السجلات:**
```bash
# حذف السجلات القديمة
find logs/ -name "*.log" -mtime +30 -delete
```

### المهام الشهرية:

1. **إعادة تدريب النماذج:**
```bash
python train_models.py
```

2. **تحديث النظام:**
```bash
sudo apt update && sudo apt upgrade
pip install --upgrade -r requirements.txt
```

3. **مراجعة الأداء الشامل:**
```bash
# تقرير شهري
python -c "
from src.risk_manager import RiskManager
from src.continuous_learner import ContinuousLearner
rm = RiskManager()
cl = ContinuousLearner()
print('=== Performance Report ===')
print(rm.get_performance_stats(days=30))
print('=== Learning Insights ===')
print(cl.get_learning_insights())
"
```

---

## 5. دليل التحديث والنشر

### تحديث الكود:

#### 1. على جهازك المحلي:
```bash
# أجرِ التعديلات
git add .
git commit -m "وصف التحديث"
git push origin main
```

#### 2. على VPS:
```bash
# إيقاف الخدمات
sudo systemctl stop forex-trading
sudo systemctl stop forex-monitor

# تحديث الكود
cd ~/forex-ml-trading
git pull origin main

# تحديث المكتبات إذا لزم
source venv/bin/activate
pip install -r requirements.txt

# إعادة تشغيل الخدمات
sudo systemctl start forex-trading
sudo systemctl start forex-monitor
```

### نشر تحديث كبير:

#### 1. إنشاء سكريبت النشر:
```bash
nano ~/deploy.sh
```

```bash
#!/bin/bash
set -e  # إيقاف عند أي خطأ

echo "Starting deployment..."

# Variables
PROJECT_DIR=~/forex-ml-trading
BACKUP_DIR=~/backups/deploy_$(date +%Y%m%d_%H%M%S)

# Create backup
echo "Creating backup..."
mkdir -p $BACKUP_DIR
cp -r $PROJECT_DIR/data $BACKUP_DIR/
cp $PROJECT_DIR/.env $BACKUP_DIR/

# Stop services
echo "Stopping services..."
sudo systemctl stop forex-trading
sudo systemctl stop forex-monitor
sudo systemctl stop forex-dashboard

# Update code
echo "Updating code..."
cd $PROJECT_DIR
git stash  # حفظ التغييرات المحلية
git pull origin main
git stash pop  # استرجاع التغييرات

# Update dependencies
echo "Updating dependencies..."
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Run migrations if any
# python manage.py migrate

# Restart services
echo "Restarting services..."
sudo systemctl start forex-trading
sudo systemctl start forex-monitor
sudo systemctl start forex-dashboard

# Verify
echo "Verifying deployment..."
sleep 5
sudo systemctl status forex-trading --no-pager

echo "Deployment completed successfully!"
```

```bash
chmod +x ~/deploy.sh
```

### التراجع عن تحديث:

```bash
# في حالة فشل التحديث
cd ~/forex-ml-trading
git log --oneline -10  # عرض آخر 10 commits
git checkout <previous-commit-hash>

# استرجاع النسخة الاحتياطية
cp ~/backups/deploy_*/data/* ~/forex-ml-trading/data/
cp ~/backups/deploy_*/.env ~/forex-ml-trading/

# إعادة تشغيل
sudo systemctl restart forex-trading
```

---

## 6. حل المشاكل الشائعة

### مشكلة: "MT5 connection failed"
```bash
# تحقق من البيانات
cat .env | grep MT5

# اختبر الاتصال
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

# تحقق من أن MT5 يسمح بالاتصالات من VPS IP
```

### مشكلة: "No data available"
```bash
# تحقق من قاعدة البيانات
sqlite3 data/trading.db "SELECT COUNT(*) FROM price_data;"

# أعد جمع البيانات
python main.py collect
```

### مشكلة: "Service keeps restarting"
```bash
# فحص السجلات
journalctl -u forex-trading -n 100

# فحص أخطاء Python
python main.py trade  # تشغيل مباشر لرؤية الخطأ
```

### مشكلة: "High memory usage"
```bash
# إضافة swap إذا لزم
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### مشكلة: "Can't access dashboard"
```bash
# تحقق من الـ firewall
sudo ufw status
sudo ufw allow 8501

# تحقق من أن Streamlit يعمل
curl http://localhost:8501
```

---

## 7. الأمان والنسخ الاحتياطي

### إجراءات الأمان:

#### 1. تأمين SSH:
```bash
# تغيير منفذ SSH
sudo nano /etc/ssh/sshd_config
# غير: Port 22 إلى Port 2222

# منع root login
PermitRootLogin no

# استخدام SSH keys فقط
PasswordAuthentication no

sudo systemctl restart sshd
```

#### 2. تشفير البيانات الحساسة:
```bash
# تشفير .env
openssl enc -aes-256-cbc -salt -in .env -out .env.enc -k YOUR_PASSWORD

# فك التشفير
openssl enc -aes-256-cbc -d -in .env.enc -out .env -k YOUR_PASSWORD
```

#### 3. مراقبة الأمان:
```bash
# تثبيت fail2ban
sudo apt install fail2ban

# مراقبة محاولات الدخول
sudo tail -f /var/log/auth.log
```

### النسخ الاحتياطي التلقائي:

#### 1. نسخ احتياطي محلي:
```bash
# يومي
0 3 * * * /home/trader/backup.sh
```

#### 2. نسخ احتياطي خارجي (Dropbox):
```bash
# تثبيت rclone
curl https://rclone.org/install.sh | sudo bash

# إعداد Dropbox
rclone config

# سكريبت النسخ للسحابة
nano ~/cloud_backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR=~/backups
REMOTE_NAME=dropbox
REMOTE_DIR=/forex-bot-backups

# رفع النسخ الاحتياطية
rclone sync $BACKUP_DIR $REMOTE_NAME:$REMOTE_DIR

echo "Cloud backup completed"
```

#### 3. استرجاع من النسخة الاحتياطية:
```bash
# من نسخة محلية
cp ~/backups/trading_20240115.db ~/forex-ml-trading/data/trading.db

# من السحابة
rclone copy dropbox:/forex-bot-backups/trading_20240115.db ~/forex-ml-trading/data/
```

---

## 📞 معلومات الاتصال للدعم

في حالة وجود مشاكل:
1. راجع السجلات أولاً: `logs/*.log`
2. ابحث في الوثائق: `docs/`
3. GitHub Issues: [YOUR_REPO_URL]/issues

---

## 🎯 ملخص سريع للمطور

```bash
# الأوامر الأساسية
ssh trader@VPS_IP                    # الاتصال
sudo systemctl status forex-trading  # الحالة
tail -f logs/trader.log             # السجلات
python run_daily_analysis.py        # تحليل يومي
./deploy.sh                         # نشر تحديث
./backup.sh                         # نسخ احتياطي

# مسارات مهمة
~/forex-ml-trading/          # المشروع
~/forex-ml-trading/logs/     # السجلات
~/forex-ml-trading/data/     # البيانات
~/backups/                   # النسخ الاحتياطية

# الخدمات
forex-trading.service        # البوت الرئيسي
forex-monitor.service        # المراقبة
forex-dashboard.service      # لوحة التحكم
```

---

تم إعداد هذا الدليل بتاريخ: 2024-01-15
الإصدار: 1.0.0