# 🤖 تثبيت Expert Advisor على MetaTrader 5

## المتطلبات:
1. MetaTrader 5 على Windows
2. خادم Linux VPS يعمل عليه نظام ML
3. فتح المنفذ 5000 على VPS

## خطوات التثبيت:

### 1. نسخ EA إلى MT5:

```
1. افتح مجلد MT5:
   - في MT5: File → Open Data Folder
   - أو: C:\Users\[اسمك]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\

2. انسخ ForexMLBot.mq5 إلى:
   MQL5\Experts\ForexMLBot.mq5

3. أعد تشغيل MT5
```

### 2. تجميع EA:

```
1. افتح MetaEditor (F4 في MT5)
2. افتح ForexMLBot.mq5
3. اضغط Compile (F7)
4. تأكد من عدم وجود أخطاء
```

### 3. إعداد MT5 للسماح بـ WebRequests:

```
1. Tools → Options → Expert Advisors
2. ✅ Allow automated trading
3. ✅ Allow WebRequest for listed URL
4. أضف عنوان VPS:
   http://69.62.121.53:5000
5. OK
```

### 4. تشغيل EA على الرسم البياني:

```
1. افتح أي رسم بياني (مثل EURUSD H1)
2. من Navigator → Expert Advisors
3. اسحب ForexMLBot إلى الرسم البياني
4. ستظهر نافذة الإعدادات
```

### 5. إعدادات EA:

```
إعدادات الاتصال:
- PythonServerURL: http://69.62.121.53:5000
- SignalCheckInterval: 60 (ثانية)

إعدادات التداول:
- RiskPerTrade: 0.01 (1%)
- DefaultLotSize: 0.01
- MagicNumber: 123456
- MaxPositions: 3
- SendTradeReports: true
```

### 6. تشغيل خادم Python على VPS:

```bash
# SSH إلى VPS
ssh ubuntu@69.62.121.53

# انتقل للمشروع
cd ~/forex-ml-trading

# تفعيل البيئة الافتراضية
source venv/bin/activate

# تثبيت Flask
pip install flask flask-cors

# تشغيل الخادم
python start_bridge_server.py

# أو كخدمة دائمة:
sudo cp scripts/mt5-bridge.service /etc/systemd/system/
sudo systemctl enable mt5-bridge
sudo systemctl start mt5-bridge
```

### 7. التحقق من الاتصال:

```bash
# من VPS - تحقق من الخادم
curl http://localhost:5000/health

# من جهازك - تحقق من الوصول
curl http://69.62.121.53:5000/health
```

### 8. مراقبة السجلات:

```bash
# سجلات الخادم
tail -f ~/forex-ml-trading/logs/mt5_bridge.log

# سجلات الخدمة
sudo journalctl -u mt5-bridge -f
```

## 🔧 حل المشاكل:

### مشكلة: EA لا يتصل بالخادم
```
1. تحقق من فتح المنفذ 5000:
   sudo ufw allow 5000

2. تحقق من تشغيل الخادم:
   sudo systemctl status mt5-bridge

3. تحقق من إعدادات WebRequest في MT5
```

### مشكلة: لا توجد إشارات
```
1. تحقق من وجود نماذج مدربة:
   ls ~/forex-ml-trading/models/

2. تحقق من البيانات:
   python main.py test

3. راجع السجلات للأخطاء
```

### مشكلة: الصفقات لا تُنفذ
```
1. تحقق من رصيد الحساب
2. تحقق من إعدادات الرافعة المالية
3. تأكد من فتح السوق
4. راجع سجل EA في MT5
```

## 📊 مراقبة الأداء:

### من MT5:
- شاهد الصفقات في Terminal → Trade
- راجع السجل في Terminal → Experts
- تابع الأرباح في Terminal → History

### من VPS:
```bash
# حالة النظام
curl http://localhost:5000/status

# واجهة المراقبة
streamlit run dashboard.py
```

## 🔄 التحديثات:

لتحديث النظام:
```bash
# على VPS
cd ~/forex-ml-trading
git pull
sudo systemctl restart mt5-bridge

# في MT5
- أعد تجميع EA إذا تغير
```

## ⚠️ تحذيرات مهمة:

1. **ابدأ بحساب تجريبي** قبل الحقيقي
2. **راقب الصفقات** في البداية
3. **ضع حدود للخسارة** في إعدادات الحساب
4. **احتفظ بنسخ احتياطية** من النماذج

## 🚀 البدء:

1. تأكد من تدريب النماذج أولاً
2. ابدأ بزوج واحد فقط
3. استخدم حجم صغير (0.01)
4. راقب لمدة أسبوع
5. زد الأزواج والأحجام تدريجياً