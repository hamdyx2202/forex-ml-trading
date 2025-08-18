# 📋 تعليمات تشغيل السيرفر الكامل

## 🚀 الخطوات على السيرفر Linux (69.62.121.53):

### 1. تسجيل الدخول للسيرفر
```bash
ssh root@69.62.121.53
```

### 2. الذهاب لمجلد المشروع
```bash
cd /home/forex-ml-trading
# أو
cd /root/forex-ml-trading
```

### 3. تثبيت المتطلبات (مرة واحدة فقط)
```bash
# إنشاء بيئة افتراضية جديدة
python3 -m venv venv_forex

# تفعيل البيئة الافتراضية
source venv_forex/bin/activate

# تثبيت المتطلبات
pip install --upgrade pip
pip install pandas numpy scikit-learn
pip install lightgbm xgboost
pip install flask flask-cors
pip install joblib scipy
```

### 4. تشغيل السيرفر
```bash
# تأكد من تفعيل البيئة الافتراضية
source venv_forex/bin/activate

# تشغيل السيرفر
python3 run_forex_ml_server.py
```

## 🔍 للتحقق من عمل السيرفر:

### من أي جهاز:
```bash
curl http://69.62.121.53:5000/status
```

### يجب أن ترى:
```json
{
    "status": "running",
    "version": "3.0-complete",
    "server": "69.62.121.53:5000",
    "models_loaded": 0,
    "total_requests": 0
}
```

## 📊 في MT5:

### إعدادات الإكسبيرت:
- ServerURL: `http://69.62.121.53:5000`
- UseRemoteServer: True ✓
- MinConfidence: 0.65

### السماح بـ WebRequest:
Tools → Options → Expert Advisors:
- ✅ Allow automated trading
- ✅ Allow WebRequest for listed URL
- أضف: `http://69.62.121.53:5000`

## 🛠️ حل المشاكل:

### إذا ظهر خطأ "No module named 'flask'":
```bash
# تأكد من تفعيل البيئة الافتراضية
source venv_forex/bin/activate

# أعد تثبيت flask
pip install flask
```

### إذا كان المنفذ 5000 مستخدم:
```bash
# ابحث عن العملية
lsof -i :5000

# أوقف العملية القديمة
kill -9 [PID]
```

### للتشغيل في الخلفية:
```bash
nohup python3 run_forex_ml_server.py > server.log 2>&1 &
```

### لمراقبة السجلات:
```bash
tail -f forex_ml_server.log
```

## ✅ النظام الكامل يشمل:

- **7.8 مليون سجل** للتدريب
- **6 نماذج ML** (LightGBM, XGBoost, Random Forest, etc)
- **200+ ميزة تقنية**
- **10 فرضيات تداول**
- **تعلم مستمر**
- **SL/TP ديناميكي**

## 🚀 السيرفر جاهز!

بمجرد تشغيل السيرفر، سيكون جاهزاً لاستقبال الطلبات من MT5 وإرسال الإشارات.

**لا يوجد أي تبسيط - هذا هو النظام الكامل!**