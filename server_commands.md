# 🎛️ أوامر إدارة النظام

## 📋 الأوامر السريعة

### 1. إيقاف جميع العمليات:
```bash
python forex_ml_control.py stop
```
أو
```bash
./simple_commands.sh stop
```

### 2. بدء النظام الكامل:
```bash
python forex_ml_control.py start
```
أو
```bash
./simple_commands.sh start
```

### 3. بدء السيرفر فقط:
```bash
python forex_ml_control.py server
```
أو
```bash
./simple_commands.sh server
```

### 4. بدء التدريب:
```bash
python forex_ml_control.py train
```

### 5. عرض حالة النظام:
```bash
python forex_ml_control.py status
```
أو
```bash
./simple_commands.sh status
```

## 🚀 أمر واحد لبدء كل شيء:
```bash
# جعل الملف قابل للتنفيذ (مرة واحدة فقط)
chmod +x simple_commands.sh

# بدء النظام
./simple_commands.sh start
```

## 📊 القائمة التفاعلية:
```bash
python forex_ml_control.py
```

ستظهر قائمة بالخيارات:
1. إيقاف جميع العمليات
2. بدء السيرفر
3. بدء التدريب الأساسي
4. بدء التعلم المستمر
5. بدء النظام الآلي الكامل
6. تدريب زوج محدد
7. عرض حالة النظام
8. بدء سريع (recommended)

## 🔧 أوامر متقدمة:

### تدريب زوج محدد:
```bash
python forex_ml_control.py train-pair EURUSD H1
```

### بدء نوع تدريب محدد:
```bash
python forex_ml_control.py train auto     # تدريب آلي
python forex_ml_control.py train continuous # تعلم مستمر
python forex_ml_control.py train integrated # تدريب متكامل
```

### عرض السجلات:
```bash
./simple_commands.sh logs
```

### تنظيف الملفات المؤقتة:
```bash
./simple_commands.sh clean
```

## 📝 ملاحظات مهمة:

1. **قبل البدء**: تأكد من تفعيل البيئة الافتراضية:
   ```bash
   source venv_pro/bin/activate
   ```

2. **إنشاء مجلد السجلات**:
   ```bash
   mkdir -p logs
   ```

3. **إصلاح مشكلة performance_tracker**:
   الملف تم إنشاؤه، فقط انسخه للسيرفر:
   ```bash
   scp performance_tracker.py root@69.62.121.53:/home/forex-ml-trading/
   scp forex_ml_control.py root@69.62.121.53:/home/forex-ml-trading/
   scp simple_commands.sh root@69.62.121.53:/home/forex-ml-trading/
   ```

## 🎯 الأمر الأسهل والأسرع:

بعد نسخ الملفات للسيرفر:
```bash
# في السيرفر
cd /home/forex-ml-trading
chmod +x simple_commands.sh

# إيقاف كل شيء
./simple_commands.sh stop

# بدء كل شيء
./simple_commands.sh start

# فحص الحالة
./simple_commands.sh status
```

## ✅ هذا كل شيء!
الآن يمكنك إدارة النظام بأوامر بسيطة جداً.