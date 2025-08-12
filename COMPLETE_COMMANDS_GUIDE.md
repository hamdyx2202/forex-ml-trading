# 📋 دليل الأوامر الكامل - خطوة بخطوة

## 🔵 الجزء الأول: الاتصال والإعداد

### 1. افتح برنامج Terminal أو CMD
```bash
# في Windows: اضغط Windows+R واكتب cmd
# في Mac/Linux: افتح Terminal
```

### 2. اتصل بالخادم
```bash
ssh root@69.62.121.53
# أدخل كلمة المرور عند الطلب
```

### 3. انتقل لمجلد المشروع
```bash
cd /home/forex-ml-trading
```

### 4. تفعيل البيئة الافتراضية المتقدمة
```bash
source venv_pro/bin/activate
```

## 🟢 الجزء الثاني: التدريب (مرة واحدة فقط)

### 5. تدريب النماذج المتقدمة
```bash
# هذا يستغرق 3-5 ساعات - شغله واتركه
python train_advanced_95_percent.py
```

**⏸️ انتظر حتى ينتهي التدريب تماماً**

## 🟡 الجزء الثالث: تشغيل النظام

### 6. تشغيل الخادم الرئيسي
```bash
# افتح نافذة screen جديدة
screen -S server

# شغل الخادم
python src/mt5_bridge_server_linux.py

# اخرج من screen بالضغط على: Ctrl+A ثم D
```

### 7. تشغيل نظام التعلم المتقدم
```bash
# افتح نافذة screen أخرى
screen -S advanced

# شغل التعلم المتقدم
python src/advanced_learner_simple.py

# اخرج من screen: Ctrl+A ثم D
```

### 8. تشغيل نظام التعلم المستمر
```bash
# افتح نافذة screen ثالثة
screen -S continuous

# شغل التعلم المستمر
python src/continuous_learner_simple.py

# اخرج من screen: Ctrl+A ثم D
```

## 🔴 الجزء الرابع: في MetaTrader 5

### 9. إعداد MT5
```
1. افتح MetaTrader 5
2. اذهب إلى: Tools > Options > Expert Advisors
3. ضع علامة ✅ على: Allow WebRequest
4. أضف في القائمة: http://69.62.121.53:5000
5. اضغط OK
```

### 10. تحميل Expert Advisor
```
1. افتح Navigator (Ctrl+N)
2. اذهب إلى Expert Advisors
3. اسحب ForexMLBot.mq5 إلى أي رسم بياني
4. في النافذة المنبثقة:
   - Server URL: http://69.62.121.53:5000
   - Risk Per Trade: 0.01 (1%)
   - Magic Number: 123456
5. اضغط OK
```

## 🟣 الجزء الخامس: المراقبة والمتابعة

### 11. مشاهدة السجلات
```bash
# لمشاهدة سجل الخادم
screen -r server

# لمشاهدة سجل التعلم المتقدم
screen -r advanced

# لمشاهدة سجل التعلم المستمر
screen -r continuous

# للخروج من أي screen: Ctrl+A ثم D
```

### 12. مشاهدة الأداء
```bash
# شاهد أداء النماذج
python src/advanced_predictor_95.py

# شاهد السجلات
tail -f logs/*.log
```

## ⚫ أوامر مفيدة إضافية

### إيقاف كل شيء
```bash
# إيقاف جميع العمليات
pkill -f python
```

### إعادة التشغيل السريع
```bash
# إيقاف كل شيء
pkill -f python

# تفعيل البيئة
source venv_pro/bin/activate

# تشغيل الخادم فقط (للاختبار السريع)
python src/mt5_bridge_server_linux.py
```

### التحقق من العمليات النشطة
```bash
# عرض جميع screens
screen -ls

# عرض العمليات
ps aux | grep python
```

### مشاهدة استخدام الموارد
```bash
# مشاهدة استخدام CPU والذاكرة
htop
# أو
top
```

## 🆘 حل المشاكل الشائعة

### إذا توقف الخادم
```bash
screen -r server
# إذا ظهر خطأ، اضغط Ctrl+C
python src/mt5_bridge_server_linux.py
```

### إذا لم تظهر إشارات في MT5
```
1. تأكد من تشغيل الخادم
2. تأكد من إعدادات WebRequest
3. تحقق من logs في MT5: View > Experts
```

### لإعادة التدريب (شهرياً)
```bash
cd /home/forex-ml-trading
source venv_pro/bin/activate
python train_advanced_95_percent.py
```

## 📊 ملخص سريع - الأوامر الأساسية يومياً

```bash
# 1. الاتصال
ssh root@69.62.121.53

# 2. الانتقال والتفعيل
cd /home/forex-ml-trading
source venv_pro/bin/activate

# 3. التحقق من العمليات
screen -ls

# 4. إذا كانت متوقفة، شغلها:
screen -S server -d -m python src/mt5_bridge_server_linux.py
screen -S advanced -d -m python src/advanced_learner_simple.py
screen -S continuous -d -m python src/continuous_learner_simple.py
```

## 💡 نصائح مهمة

1. **لا تغلق Terminal** أثناء التدريب
2. **استخدم screen دائماً** للعمليات الطويلة
3. **ابدأ بحساب تجريبي** في MT5
4. **راقب الأداء يومياً** في الأسبوع الأول
5. **لا تخاطر بأكثر من 1%** في كل صفقة

---

✅ **النظام الآن جاهز للعمل بدقة عالية!**