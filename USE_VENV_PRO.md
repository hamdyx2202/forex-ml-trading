# 🚀 استخدام venv_pro الموجود

## إذا كان venv_pro موجود وفيه كل المكتبات:

### 1. ابحث عن مكان venv_pro
```bash
# جرب هذه المسارات
ls -la /home/forex-ml-trading/venv_pro
ls -la /root/forex-ml-trading/venv_pro
ls -la ~/venv_pro
```

### 2. انتقل للمجلد الصحيح
```bash
# مثال
cd /home/forex-ml-trading
# أو
cd /root/forex-ml-trading
```

### 3. فعّل venv_pro
```bash
source venv_pro/bin/activate
```

### 4. تحقق من المكتبات
```bash
# يجب أن يظهر (venv_pro) في بداية السطر
python3 -m pip list | grep -E "flask|pandas|numpy|sklearn"
```

### 5. شغّل السيرفر
```bash
# جرب أحد هذه الملفات
python3 run_forex_ml_server.py
# أو
python3 run_complete_system.py
# أو
python3 forex_ml_server_standalone.py
```

## 🔍 إذا لم تجد venv_pro:

### ابحث في كل النظام
```bash
find / -type d -name "venv_pro" 2>/dev/null
```

## 📝 ملاحظات مهمة:

1. **venv_pro** يحتوي على كل المكتبات المطلوبة
2. تأكد من تفعيله قبل تشغيل السيرفر
3. يجب أن ترى `(venv_pro)` في بداية سطر الأوامر

## 🚀 بديل سريع:

إذا لم تستطع إيجاد venv_pro، استخدم:
```bash
python3 forex_ml_server_standalone.py
```

هذا الملف يعمل مع أي بيئة Python ويتكيف مع المكتبات المتاحة!