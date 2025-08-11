# 🚀 دليل البدء السريع على Linux VPS

## الأوامر السريعة:

### 1. إعداد النظام (مرة واحدة فقط):
```bash
# بعد تفعيل البيئة الافتراضية
source venv/bin/activate

# إعداد قاعدة البيانات
python main_linux.py setup

# اختبار النظام
python main_linux.py test
```

### 2. تشغيل الخادم:
```bash
# تشغيل مباشر
python main_linux.py server

# أو
python start_bridge_server.py

# أو كخدمة دائمة
sudo systemctl start mt5-bridge
```

### 3. التحقق من العمل:
```bash
# من terminal آخر
curl http://localhost:5000/health

# مراقبة السجلات
tail -f logs/mt5_bridge_linux.log
```

## 🔧 حل المشاكل الشائعة:

### مشكلة: No module named 'MetaTrader5'
```bash
# هذا طبيعي على Linux! استخدم:
python main_linux.py  # بدلاً من python main.py
```

### مشكلة: Permission denied
```bash
chmod +x main_linux.py
chmod +x start_bridge_server.py
```

### مشكلة: Port 5000 already in use
```bash
# إيقاف العملية القديمة
sudo lsof -i :5000
sudo kill -9 <PID>
```

## ✅ التحقق من أن كل شيء يعمل:

عندما يعمل الخادم بنجاح ستري:
```
2025-08-11 22:00:00 | INFO     | Starting Linux Bridge Server on 0.0.0.0:5000
2025-08-11 22:00:00 | INFO     | This is a simplified version that works without MT5
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

وعندما يتصل EA ستري:
```
2025-08-11 22:01:00 | INFO     | Raw data received: {"symbol":"EURUSDm","price":1.16185}
2025-08-11 22:01:00 | INFO     | Parsed JSON data: {'symbol': 'EURUSDm', 'price': 1.16185}
2025-08-11 22:01:00 | INFO     | Processing signal for EURUSDm at 1.16185
```

## 📌 ملاحظات مهمة:

1. **لا تحتاج MT5 على Linux** - الخادم يعمل بدونه
2. **EA على Windows** يرسل البيانات للخادم
3. **الخادم يحلل** ويرجع الإشارات
4. **كل شيء يحفظ** في قاعدة البيانات

## 🎯 الخلاصة:

```bash
# كل ما تحتاجه:
source venv/bin/activate
python main_linux.py setup    # مرة واحدة
python main_linux.py server   # للتشغيل
```

**النظام جاهز للعمل! 🚀**