# دليل واجهة الويب ونظام 24/7
# Web Dashboard & 24/7 System Guide

## 🚀 نظرة عامة
تم إنشاء نظام متكامل يتضمن:
- ✅ واجهة ويب احترافية للمراقبة والتحكم
- ✅ نظام auto-restart تلقائي
- ✅ نظام backup يومي
- ✅ خدمات systemd للعمل 24/7
- ✅ نظام تنبيهات

## 📁 الملفات المُنشأة

### 1. واجهة الويب
```
web_dashboard/
├── app.py                    # التطبيق الرئيسي Flask
├── auto_restart.py           # نظام إعادة التشغيل التلقائي
├── templates/
│   ├── base.html            # القالب الأساسي
│   ├── dashboard.html       # لوحة التحكم الرئيسية
│   └── login.html          # صفحة تسجيل الدخول
└── system_state.json        # حالة النظام
```

### 2. أنظمة الصيانة
```
daily_maintenance.sh         # صيانة يومية تلقائية
systemd/
├── forex-ml-server.service     # خدمة السيرفر الرئيسي
├── forex-ml-dashboard.service  # خدمة الواجهة
├── forex-ml-monitor.service    # خدمة المراقبة
└── install_services.sh         # تثبيت الخدمات
```

## 🔧 التثبيت على VPS

### 1. نسخ الملفات
```bash
# نسخ مجلد web_dashboard
scp -r web_dashboard root@69.62.121.53:/home/forex-ml-trading/

# نسخ ملفات الصيانة
scp daily_maintenance.sh root@69.62.121.53:/home/forex-ml-trading/
scp -r systemd root@69.62.121.53:/home/forex-ml-trading/
```

### 2. تثبيت المتطلبات
```bash
ssh root@69.62.121.53
cd /home/forex-ml-trading
source venv_pro/bin/activate

# تثبيت المكتبات المطلوبة
pip install flask flask-login flask-socketio werkzeug psutil
```

### 3. إعداد الصلاحيات
```bash
chmod +x daily_maintenance.sh
chmod +x systemd/install_services.sh
chmod +x web_dashboard/auto_restart.py
```

## 🚀 التشغيل

### الطريقة 1: استخدام systemd (مُوصى به)
```bash
cd /home/forex-ml-trading/systemd
sudo ./install_services.sh
```

### الطريقة 2: التشغيل اليدوي
```bash
# الواجهة
screen -dmS dashboard python web_dashboard/app.py

# المراقبة التلقائية
screen -dmS monitor python web_dashboard/auto_restart.py

# الصيانة اليومية (cron)
crontab -e
# أضف: 0 3 * * * /home/forex-ml-trading/daily_maintenance.sh
```

## 🌐 الوصول للواجهة

### URL الواجهة
```
http://YOUR_VPS_IP:8080
```

### بيانات الدخول الافتراضية
- **Admin**: username: `admin`, password: `admin123`
- **Viewer**: username: `viewer`, password: `viewer123`

⚠️ **مهم**: غيّر كلمات المرور فوراً في `web_dashboard/app.py`

## 📊 مميزات الواجهة

### 1. لوحة التحكم الرئيسية
- حالة جميع الأنظمة (live)
- استخدام CPU/RAM/Disk
- إحصائيات التداول
- آخر الإشارات
- رسم بياني للأداء

### 2. التحكم السريع
- ✅ إعادة تشغيل الكل
- ✅ نسخ احتياطي فوري
- ✅ مسح السجلات القديمة
- ✅ اختبار الإشارة
- ✅ إيقاف طوارئ

### 3. المراقبة المستمرة
- فحص كل 5 ثواني للحالة
- فحص كل 5 دقائق للعمليات
- إعادة تشغيل تلقائية عند التوقف
- تنبيهات عند المشاكل

## 🔄 النسخ الاحتياطي التلقائي

### يومياً (3:00 AM)
- نسخ احتياطي للنماذج
- نسخ احتياطي لقاعدة البيانات
- نسخ احتياطي للإعدادات

### أسبوعياً (الأحد)
- نسخة كاملة للنظام
- حفظ لمدة 60 يوم

### التنظيف التلقائي
- سجلات > 30 يوم: حذف
- نسخ يومية > 7 أيام: حذف
- نسخ نماذج > 30 يوم: حذف

## 🚨 نظام التنبيهات

### التنبيهات التلقائية عند:
- توقف أي نظام
- استخدام CPU > 90%
- استخدام RAM > 90%
- مساحة القرص < 10%
- فشل إعادة التشغيل المتكرر

### ملف التنبيهات
```bash
tail -f /home/forex-ml-trading/logs/alerts.log
```

## 🛠️ إدارة النظام

### عرض حالة الخدمات
```bash
systemctl status forex-ml-server
systemctl status forex-ml-dashboard
systemctl status forex-ml-monitor
```

### إعادة تشغيل خدمة
```bash
systemctl restart forex-ml-server
systemctl restart forex-ml-dashboard
```

### عرض السجلات
```bash
journalctl -u forex-ml-server -f
journalctl -u forex-ml-dashboard -f
```

### إيقاف/تشغيل الخدمات
```bash
systemctl stop forex-ml-server
systemctl start forex-ml-server
```

## 📋 الصيانة الدورية

### يومياً (تلقائي)
- ✅ نسخ احتياطي
- ✅ تنظيف السجلات
- ✅ فحص المساحة
- ✅ تقرير يومي

### أسبوعياً (يدوي)
- تحديث النظام: `apt update && apt upgrade`
- فحص الأمان: `fail2ban-client status`
- مراجعة التنبيهات

### شهرياً (يدوي)
- مراجعة أداء النماذج
- تحديث كلمات المرور
- فحص استهلاك الموارد

## 🔒 الأمان

### 1. تغيير كلمات المرور
في `web_dashboard/app.py`:
```python
USERS = {
    'admin': {
        'password': generate_password_hash('YOUR_NEW_PASSWORD'),
        'role': 'admin'
    }
}
```

### 2. تقييد الوصول (اختياري)
```bash
# استخدام firewall
ufw allow from YOUR_IP to any port 8080
```

### 3. HTTPS (موصى به)
```bash
# استخدام nginx كـ reverse proxy
apt install nginx
# إعداد SSL مع Let's Encrypt
```

## 🐛 حل المشاكل

### الواجهة لا تعمل
```bash
# تحقق من المنفذ
netstat -tuln | grep 8080

# تحقق من السجلات
tail -f logs/dashboard.log
```

### Auto-restart لا يعمل
```bash
# تحقق من العملية
ps aux | grep auto_restart

# تشغيل يدوي
python web_dashboard/auto_restart.py
```

### مشاكل systemd
```bash
# إعادة تحميل
systemctl daemon-reload

# فحص الأخطاء
journalctl -xe
```

## 📊 مؤشرات النجاح

✅ النظام يعمل 24/7 إذا:
- جميع المؤشرات خضراء في الواجهة
- لا توجد تنبيهات في alerts.log
- uptime > 7 أيام
- معدل إعادة التشغيل < 5/يوم
- استخدام الموارد < 80%

## 🎯 نصائح مهمة

1. **لا تعدل الملفات الأساسية** أثناء عمل النظام
2. **احتفظ بـ screen** كخيار احتياطي مع systemd
3. **راقب alerts.log** يومياً
4. **خذ نسخة احتياطية** قبل أي تحديث
5. **اختبر على بيئة تجريبية** أولاً

## 📞 للمساعدة

في حالة المشاكل:
1. تحقق من `/home/forex-ml-trading/logs/`
2. استخدم الواجهة للتشخيص
3. راجع systemd status
4. تحقق من استهلاك الموارد

---

🎉 **مبروك! النظام الآن يعمل 24/7 مع واجهة احترافية للمراقبة والتحكم**