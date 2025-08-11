# 🐧 تحديث للعمل على Linux VPS

## المشكلة:
MetaTrader5 لا يعمل على Linux VPS لأنه يحتاج Windows.

## الحل النهائي:

### 1. خادم مبسط على Linux:
تم إنشاء `mt5_bridge_server_linux.py` الذي:
- لا يحتاج MT5
- لا يحتاج talib
- يعمل كـ API فقط
- يستقبل الأسعار من EA
- يحلل ويرجع إشارات

### 2. التوافق التلقائي:
`start_bridge_server.py` يكتشف نظام التشغيل:
- Linux → يستخدم النسخة المبسطة
- Windows → يستخدم النسخة الكاملة

## التثبيت على Linux VPS:

```bash
# 1. تحديث النظام
sudo apt update && sudo apt upgrade -y

# 2. تثبيت Python
sudo apt install python3 python3-pip python3-venv -y

# 3. استنساخ المشروع
git clone https://github.com/your-repo/forex-ml-trading.git
cd forex-ml-trading

# 4. إنشاء البيئة الافتراضية
python3 -m venv venv
source venv/bin/activate

# 5. تثبيت المتطلبات الأساسية فقط
pip install flask flask-cors pandas numpy loguru

# 6. تشغيل الخادم
python start_bridge_server.py
```

## كيف يعمل النظام الآن؟

### على Linux VPS:
1. **الخادم المبسط** يعمل بدون MT5
2. **يستقبل** بيانات الأسعار من EA
3. **يحلل** باستخدام استراتيجية بسيطة
4. **يرجع** إشارات التداول

### على Windows (MT5):
1. **EA** يرسل الأسعار للخادم
2. **يستقبل** الإشارات
3. **ينفذ** الصفقات الحقيقية

## الميزات:

### ✅ ما يعمل:
- استقبال الأسعار من EA
- تحليل بسيط (SMA + RSI)
- إرجاع إشارات BUY/SELL
- حفظ نتائج الصفقات
- حساب معدل النجاح

### ⚠️ القيود:
- لا يستخدم النماذج المعقدة (حالياً)
- استراتيجية بسيطة فقط
- لا يجمع بيانات تاريخية

## تشغيل الخادم كخدمة:

```bash
# 1. إنشاء ملف الخدمة
sudo nano /etc/systemd/system/mt5-bridge.service

# 2. محتوى الملف:
[Unit]
Description=MT5 Bridge Server (Linux)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/forex-ml-trading
Environment="PATH=/home/ubuntu/forex-ml-trading/venv/bin"
ExecStart=/home/ubuntu/forex-ml-trading/venv/bin/python start_bridge_server.py
Restart=always

[Install]
WantedBy=multi-user.target

# 3. تفعيل الخدمة
sudo systemctl enable mt5-bridge
sudo systemctl start mt5-bridge

# 4. التحقق من الحالة
sudo systemctl status mt5-bridge
```

## اختبار الخادم:

```bash
# 1. اختبار الصحة
curl http://localhost:5000/health

# 2. اختبار إشارة
curl -X POST http://localhost:5000/get_signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","price":1.0850}'

# 3. حالة النظام
curl http://localhost:5000/status
```

## مراقبة السجلات:

```bash
# سجلات الخادم
tail -f logs/mt5_bridge_linux.log

# سجلات النظام
sudo journalctl -u mt5-bridge -f
```

## التحسينات المستقبلية:

1. **إضافة المزيد من المؤشرات**
2. **تحميل النماذج المدربة**
3. **حفظ البيانات في قاعدة بيانات**
4. **تحسين الاستراتيجية**

## الخلاصة:

النظام الآن يعمل على Linux VPS بدون:
- MetaTrader5
- talib
- أي مكتبات معقدة

فقط Flask + pandas + numpy!