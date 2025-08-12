# 🚀 حل سريع - استخدم EA الذي يعمل بالفعل!

## المشكلة:
- الخادم يعمل ✅ (نرى اتصالات من EA آخر)
- لكن `/api/test` لم يُحدث بعد على الخادم

## الحل السريع:

### استخدم EA التداول الأصلي الذي يعمل:

### 1. **ForexMLBot.mq5** (النسخة الأصلية)
هذا EA يعمل بالفعل كما نرى في السجلات!

يستخدم `/get_signal` endpoint الذي يعمل.

### 2. **أو عدّل ForexMLDataSyncFixed.mq5:**

غيّر هذا السطر:
```mql5
if(SendDataToServer("/api/test", json))
```

إلى:
```mql5
if(SendDataToServer("/get_signal", json))
```

### 3. **أو استخدم سكريبت بسيط للإرسال:**

```mql5
//+------------------------------------------------------------------+
//|                                              SimpleDataSender.mq5 |
//|                                      إرسال بسيط للبيانات        |
//+------------------------------------------------------------------+
#property copyright "Simple Data Sender"
#property version   "1.00"

input string ServerURL = "http://69.62.121.53:5000";
input string Symbol1 = "EURUSD";
input string Symbol2 = "GBPUSD";

int OnInit()
{
    // اختبار الاتصال
    if(TestConnection())
    {
        Print("✅ Connection successful!");
        
        // إرسال بيانات بسيطة
        SendSymbolData(Symbol1 + "m");
        SendSymbolData(Symbol2 + "m");
    }
    else
    {
        Print("❌ Connection failed!");
    }
    
    return(INIT_SUCCEEDED);
}

bool TestConnection()
{
    string url = ServerURL + "/health";
    char post[], result[];
    string headers = "";
    
    int res = WebRequest("GET", url, headers, 5000, post, result, headers);
    return (res == 200);
}

void SendSymbolData(string symbol)
{
    MqlRates rates[];
    int copied = CopyRates(symbol, PERIOD_H1, 0, 100, rates);
    
    if(copied > 0)
    {
        // إرسال كطلب إشارة
        for(int i = 0; i < MathMin(5, copied); i++)
        {
            string json = StringFormat("{\"symbol\":\"%s\",\"price\":%f}", 
                                     symbol, rates[i].close);
            
            SendToServer("/get_signal", json);
        }
        
        Print("✅ Sent data for ", symbol);
    }
}

void SendToServer(string endpoint, string json)
{
    string url = ServerURL + endpoint;
    string headers = "Content-Type: application/json\r\n";
    char post[], result[];
    
    StringToCharArray(json, post);
    WebRequest("POST", url, headers, 5000, post, result, headers);
}
```

## 🎯 الحل الأسرع:

**استخدم ForexMLBot.mq5 الأصلي!**

هو يعمل بالفعل كما نرى في سجلات الخادم:
```
Processing signal for EURJPYm at 172.178
Processing signal for GBPJPYm at 199.11
```

## 📝 ملاحظة:
الخادم الحالي يستخدم `/get_signal` وليس `/api/test`.

يمكنك:
1. استخدام EA الأصلي الذي يعمل
2. أو انتظر حتى أعيد تشغيل الخادم بالتحديثات الجديدة
3. أو عدّل EA ليستخدم endpoints الموجودة

**الخادم يعمل - فقط استخدم الطريقة الصحيحة!** ✅