# 🔧 دليل حل مشاكل نظام Forex ML Trading

## 📋 المشاكل الشائعة وحلولها

### 1. ❌ الإكسبيرت لا يرسل البيانات

#### الأعراض:
- لا توجد رسائل في Journal
- السيرفر لا يستقبل بيانات

#### التشخيص:
1. في MT5، تحقق من Journal للإكسبيرت
2. ابحث عن رسائل مثل:
   - "فشل الاتصال! رمز الخطأ: 4060"
   - "WebRequest not allowed"

#### الحل:
```
في MT5:
1. Tools → Options → Expert Advisors
2. ✅ Allow WebRequest for listed URL
3. أضف: http://69.62.121.53:5000
4. ✅ Allow DLL imports (إذا مطلوب)
5. أعد تشغيل MT5
```

### 2. ❌ السيرفر لا يعمل

#### التشخيص في السيرفر:
```bash
# فحص سريع
./quick_debug.sh

# فحص مفصل
./server_diagnostics.sh check-server
```

#### الحل:
```bash
# إيقاف أي عمليات قديمة
pkill -f mt5_bridge_server

# بدء السيرفر
source venv_pro/bin/activate
nohup python src/mt5_bridge_server_advanced.py > logs/server.log 2>&1 &

# التحقق
tail -f logs/server.log
```

### 3. ❌ خطأ ModuleNotFoundError

#### مثال:
```
ModuleNotFoundError: No module named 'performance_tracker'
```

#### الحل:
```bash
# نسخ الملفات المفقودة
scp performance_tracker.py root@69.62.121.53:/home/forex-ml-trading/
scp forex_ml_control.py root@69.62.121.53:/home/forex-ml-trading/

# أو تثبيت الحزم المفقودة
pip install pandas numpy scikit-learn xgboost joblib loguru flask
```

### 4. ❌ خطأ في الاتصال (Error 5201)

#### الأعراض:
```
❌ فشل الاتصال! رمز الخطأ: 5201
   التفاصيل: Failed to connect to specified URL
```

#### التشخيص:
```bash
# من السيرفر
./server_diagnostics.sh check-network

# اختبار المنفذ
telnet localhost 5000
```

#### الحل:
```bash
# فحص جدار الحماية
iptables -L -n | grep 5000

# فتح المنفذ إذا لزم
iptables -A INPUT -p tcp --dport 5000 -j ACCEPT

# أو استخدام ufw
ufw allow 5000
```

### 5. ❌ النماذج لا تتدرب

#### التشخيص:
```bash
# فحص التدريب
./server_diagnostics.sh check-training

# فحص الأخطاء
grep -i "error" logs/*.log | tail -20
```

#### الحل:
```bash
# تدريب يدوي
source venv_pro/bin/activate
python integrated_training_sltp.py

# أو استخدام مركز التحكم
python forex_ml_control.py
# اختر 3 للتدريب الأساسي
```

### 6. ❌ قاعدة البيانات فارغة

#### التشخيص:
```bash
# فحص البيانات
sqlite3 trading_data.db "SELECT COUNT(*) FROM ohlcv_data;"
```

#### الحل:
1. تأكد من أن الإكسبيرت يرسل البيانات
2. تحقق من السجلات:
```bash
grep "historical_data" logs/server.log | tail -20
```

## 🔍 أوامر التشخيص السريع

### للسيرفر:
```bash
# فحص سريع شامل
./quick_debug.sh

# فحص مفصل
./server_diagnostics.sh check-all

# مراقبة مباشرة
./server_diagnostics.sh monitor

# اختبار الاتصال
./server_diagnostics.sh test-connection
```

### للإكسبيرت (MT5):
1. استخدم `ForexMLBatchDataSender_AllPairs_Debug.mq5`
2. تفعيل Debug Mode = true
3. راقب Journal للتفاصيل

## 📊 سير العمل الصحيح

### 1. بدء النظام:
```bash
# في السيرفر
cd /home/forex-ml-trading
./simple_commands.sh start
```

### 2. في MT5:
- أضف الإكسبيرت على أي chart
- تأكد من السماح بـ WebRequest
- راقب Journal

### 3. التحقق من العمل:
```bash
# في السيرفر
./simple_commands.sh status
```

## 🚨 حل سريع للمشاكل الشائعة

### مشكلة: "السيرفر لا يستجيب"
```bash
# إعادة تشغيل كاملة
./simple_commands.sh stop
sleep 5
./simple_commands.sh start
```

### مشكلة: "لا توجد نماذج"
```bash
# تدريب سريع
source venv_pro/bin/activate
python forex_ml_control.py train-pair EURUSD H1
```

### مشكلة: "أخطاء في السجلات"
```bash
# تنظيف وإعادة تشغيل
./server_diagnostics.sh clean-logs
./simple_commands.sh stop
./simple_commands.sh start
```

## 📝 نصائح مهمة

1. **دائماً تحقق من Journal في MT5** - معظم المشاكل تظهر هناك
2. **استخدم النسخة Debug من الإكسبيرت** للحصول على تفاصيل أكثر
3. **راقب السجلات في السيرفر** بـ `tail -f logs/*.log`
4. **احتفظ بنسخ احتياطية** قبل أي تغييرات كبيرة

## 🆘 إذا فشل كل شيء

```bash
# إعادة تعيين كاملة (احذر!)
./simple_commands.sh stop
mv models models_backup_$(date +%Y%m%d)
mv trading_data.db trading_data_backup_$(date +%Y%m%d).db
./simple_commands.sh clean
./simple_commands.sh start
```

## ✅ الخلاصة

معظم المشاكل تحل بـ:
1. التحقق من إعدادات WebRequest في MT5
2. إعادة تشغيل السيرفر
3. استخدام أوامر التشخيص المتوفرة