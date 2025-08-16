# دليل إعداد Expert Advisor المتقدم

## 📋 المتطلبات:

1. **MetaTrader 5** محدث لآخر إصدار
2. **Python Server** يعمل على المنفذ 5000
3. **النماذج المدربة** في مجلد `models/`

## 🔧 خطوات التثبيت:

### 1. نسخ EA إلى MT5:
```
1. افتح مجلد بيانات MT5: File > Open Data Folder
2. انتقل إلى: MQL5/Experts/
3. انسخ ForexMLBot_Advanced_V2.mq5
4. في MT5: أعد تشغيل أو اضغط F5 في Navigator
```

### 2. تجميع EA:
```
1. في MetaEditor: افتح ForexMLBot_Advanced_V2.mq5
2. اضغط F7 للتجميع
3. تأكد من عدم وجود أخطاء
```

### 3. السماح بـ WebRequest:
```
في MT5:
1. Tools > Options > Expert Advisors
2. ✅ Allow WebRequest for listed URL
3. أضف: http://localhost:5000
```

### 4. تشغيل Python Server:
```bash
# Terminal 1 - تشغيل الخادم
cd /home/forex-ml-trading
source venv_pro/bin/activate
python mt5_prediction_server.py
```

### 5. إضافة EA للرسم البياني:
```
1. افتح أي رسم بياني (مثل EURUSD H1)
2. من Navigator: اسحب ForexMLBot_Advanced_V2
3. في الإعدادات:
   - Magic Number: 12345
   - Risk Percent: 1.0
   - Min Confidence: 0.75
   - Candles History: 200
   - اختر الاستراتيجيات المطلوبة
4. اضغط OK
```

## ⚙️ إعدادات EA:

### الإعدادات الأساسية:
- **Server URL**: http://localhost:5000/api/predict_advanced
- **Magic Number**: رقم فريد لتمييز صفقات EA
- **Risk Percent**: نسبة المخاطرة من الرصيد (1-2%)
- **Max Trades**: أقصى عدد صفقات مفتوحة
- **Min Confidence**: الحد الأدنى للثقة (0.75-0.85)

### الاستراتيجيات:
- **Ultra Short**: للسكالبينج السريع (30 دقيقة)
- **Scalping**: صفقات قصيرة (1 ساعة)
- **Short Term**: صفقات قصيرة المدى (2-4 ساعات)
- **Medium Term**: صفقات متوسطة (4-8 ساعات)
- **Long Term**: صفقات طويلة (24 ساعة+)

### إدارة الصفقات:
- **Use Trailing Stop**: تفعيل Trailing Stop التلقائي
- **Use Move to Breakeven**: نقل SL للتعادل عند الربح

## 📊 مراقبة الأداء:

### في MT5 Journal:
```
🚀 بدء تشغيل ForexMLBot Advanced V2
📊 السيرفر: http://localhost:5000/api/predict_advanced
✅ عدد الأزواج المتاحة: 19 من 19
🎯 إشارة جديدة! EURUSD M5
   📊 الاستراتيجية: scalping
   🎯 الإشارة: شراء
   📈 الثقة: 85.00%
   🛑 SL: 1.08450
   🎯 TP1: 1.08650, TP2: 1.08750, TP3: 1.08850
✅ تم فتح صفقة #12345678
```

### عبر API:
```bash
# فحص صحة الخادم
curl http://localhost:5000/api/health

# الحصول على الأداء
curl http://localhost:5000/api/get_performance
```

## 🚨 حل المشاكل الشائعة:

### 1. "WebRequest failed":
- تأكد من إضافة URL في إعدادات MT5
- تأكد من تشغيل Python server
- جرب: http://127.0.0.1:5000 بدلاً من localhost

### 2. "No signal received":
- تحقق من وجود نماذج مدربة
- تأكد من توفر بيانات كافية (200 شمعة)
- راجع سجلات Python server

### 3. "Invalid stops":
- تحقق من Spread الحالي
- تأكد من أن SL/TP يحترم Stop Level

## 📈 تحسين الأداء:

1. **اختر الأزواج المناسبة**:
   - ابدأ بالأزواج الرئيسية (EURUSD, GBPUSD)
   - أضف المعادن (XAUUSD) بحذر

2. **اضبط الإعدادات**:
   - ابدأ بـ Min Confidence عالي (0.80+)
   - استخدم Risk Percent منخفض (0.5-1%)
   - فعّل استراتيجية واحدة أو اثنتين فقط

3. **راقب النتائج**:
   - تابع Win Rate لكل استراتيجية
   - اضبط الإعدادات حسب الأداء
   - أعد التدريب شهرياً

## 🔄 التحديث المستمر:

EA يرسل نتائج الصفقات للخادم للتعلم المستمر:
- نتائج الصفقات المغلقة
- أداء كل استراتيجية
- بيانات السوق الجديدة

هذا يساعد النظام على التحسن باستمرار!

## 📞 الدعم:

في حالة وجود مشاكل:
1. راجع Journal في MT5
2. راجع logs/mt5_server.log
3. تأكد من تحديث جميع المكونات