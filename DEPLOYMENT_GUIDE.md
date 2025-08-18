# 🚀 دليل النشر الكامل - Forex ML Trading System

## 📋 المتطلبات
- **Linux Server**: 69.62.121.53 (للنظام الرئيسي)
- **Windows RDP**: لتشغيل MT5 والإكسبيرت
- **RAM**: 4GB minimum
- **Python**: 3.8+
- **Port**: 5000 (مفتوح)

## 🔧 خطوات التثبيت على السيرفر Linux

### 1. رفع الملفات للسيرفر
```bash
# رفع جميع الملفات إلى /root/forex-ml-trading/
scp -r * root@69.62.121.53:/root/forex-ml-trading/
```

### 2. الاتصال بالسيرفر
```bash
ssh root@69.62.121.53
cd /root/forex-ml-trading
```

### 3. تشغيل سكريبت التثبيت
```bash
chmod +x install_server_complete.sh
./install_server_complete.sh
```

### 4. تفعيل البيئة الافتراضية
```bash
source venv_forex/bin/activate
```

### 5. تشغيل السيرفر
```bash
python3 start_forex_server.py
```

## 📱 إعداد MT5 على Windows RDP

### 1. تثبيت الإكسبيرت
- انسخ `ForexMLBot_Advanced_V3_Unified.mq5` إلى:
  `C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts\`

### 2. تصريح الاتصال بالسيرفر
في MT5:
- Tools → Options → Expert Advisors
- ✅ Allow automated trading
- ✅ Allow WebRequest for listed URL
- أضف: `http://69.62.121.53:5000`

### 3. إعدادات الإكسبيرت
```
Server Settings:
- ServerURL: http://69.62.121.53:5000
- ServerTimeout: 5000
- UseRemoteServer: True ✓

Trading Settings:
- LotSize: 0.01
- MinConfidence: 0.65
- CandlesToSend: 200

Risk Management:
- UseServerSLTP: True ✓
- MoveToBreakeven: True ✓
- BreakevenPips: 30
```

### 4. إلحاق الإكسبيرت
- ألحق الإكسبيرت بالأزواج: EURUSD, GBPUSD, USDJPY, AUDUSD
- الأطر الزمنية: M15 و H1
- فعّل AutoTrading

## 🔍 التحقق من عمل النظام

### 1. فحص السيرفر
```bash
# من أي جهاز
curl http://69.62.121.53:5000/status
```

### 2. مراقبة السجلات
```bash
# على السيرفر
tail -f forex_server.log
tail -f unified_server.log
```

### 3. في MT5
- راقب لوحة الإكسبيرت
- تحقق من Journal للرسائل
- راقب الصفقات المفتوحة

## 🛠️ الصيانة

### إعادة تدريب النماذج
```bash
curl -X POST http://69.62.121.53:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSDm","timeframe":"M15"}'
```

### تشغيل كخدمة دائمة
```bash
# إنشاء ملف الخدمة
sudo nano /etc/systemd/system/forex-ml.service
```

محتوى الملف:
```ini
[Unit]
Description=Forex ML Trading Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/forex-ml-trading
Environment="PATH=/root/forex-ml-trading/venv_forex/bin"
ExecStart=/root/forex-ml-trading/venv_forex/bin/python start_forex_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

تفعيل الخدمة:
```bash
sudo systemctl enable forex-ml
sudo systemctl start forex-ml
sudo systemctl status forex-ml
```

## 📊 مراقبة الأداء

### إحصائيات النظام
```bash
# استخدام الذاكرة
free -h

# استخدام المعالج
top

# حجم قاعدة البيانات
du -h data/forex_ml.db
```

### إحصائيات التداول
- عدد النماذج المدربة
- معدل دقة التنبؤات
- عدد الصفقات المنفذة
- معدل الفوز

## ⚠️ حل المشاكل

### السيرفر لا يستجيب
1. تحقق من الـ firewall:
   ```bash
   sudo ufw allow 5000/tcp
   ```

2. تحقق من العملية:
   ```bash
   ps aux | grep python
   ```

### لا توجد إشارات
1. تحقق من النماذج المدربة
2. راجع السجلات
3. تأكد من إرسال 200 شمعة

### أخطاء في MT5
1. تحقق من إعدادات WebRequest
2. زيادة Timeout
3. التأكد من اتصال الإنترنت

## 🔐 الأمان

1. **استخدم HTTPS** (اختياري):
   ```python
   app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
   ```

2. **قيّد الوصول بـ API Key**

3. **احم قاعدة البيانات**

## 📈 التحسينات المستقبلية

1. إضافة المزيد من الأزواج
2. تحسين خوارزميات التعلم
3. إضافة WebSocket للتحديثات الحية
4. لوحة معلومات ويب

---
**النظام جاهز للعمل! 🚀**