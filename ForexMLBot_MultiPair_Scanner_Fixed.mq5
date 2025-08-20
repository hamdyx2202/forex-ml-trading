//+------------------------------------------------------------------+
//|                  ForexMLBot_MultiPair_Scanner_Fixed.mq5           |
//|                     نظام متعدد الأزواج والفريمات                |
//|                     مع دعم أسماء الأزواج المختلفة               |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property version   "4.1"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

// إعدادات الإدخال
input string   ServerURL = "http://69.62.121.53:5000";     // رابط السيرفر
input bool     UseRemoteServer = true;                     // استخدام السيرفر البعيد
input double   MinConfidence = 0.65;                       // الحد الأدنى للثقة
input int      CandlesToSend = 200;                        // عدد الشموع المرسلة
input double   RiskPercent = 1.0;                          // نسبة المخاطرة لكل صفقة
input int      MaxTradesPerPair = 1;                       // أقصى عدد صفقات لكل زوج
input int      CheckIntervalSeconds = 60;                  // فترة الفحص بالثواني
input bool     AutoDetectPairs = true;                     // اكتشاف الأزواج تلقائياً
input string   ManualPairs = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,EURJPY,GBPJPY,EURNZD"; // الأزواج اليدوية
input string   Timeframes = "M5,M15,M30,H1,H4";           // الفريمات
input bool     EnableLogging = true;                       // تفعيل السجلات
input string   SymbolSuffix = "m";                         // لاحقة الرموز (m, .ecn, إلخ)
input int      OrderTimeout = 30000;                       // Timeout للأوامر بالميلي ثانية
input int      MaxRetries = 3;                             // عدد محاولات إعادة فتح الصفقة
input int      RetryDelay = 2000;                          // التأخير بين المحاولات بالميلي ثانية

// متغيرات عامة
CTrade trade;
CPositionInfo position;
string pairs[];
ENUM_TIMEFRAMES timeframes[];
datetime lastCheckTime[];
int totalSignals = 0;
int totalTrades = 0;
string detectedSuffix = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // تهيئة إعدادات التداول
    trade.SetExpertMagicNumber(123456);
    trade.SetDeviationInPoints(20);  // زيادة الانحراف المسموح
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    trade.SetAsyncMode(false);  // التداول المتزامن
    
    // اكتشاف لاحقة الرموز
    detectedSuffix = DetectSymbolSuffix();
    Print("🔍 تم اكتشاف لاحقة الرموز: '", detectedSuffix, "'");
    
    // تحليل الأزواج
    if(AutoDetectPairs)
    {
        DetectAvailablePairs();
    }
    else
    {
        // استخدام الأزواج اليدوية مع اللاحقة
        string manual_pairs[];
        StringSplit(ManualPairs, ',', manual_pairs);
        ArrayResize(pairs, ArraySize(manual_pairs));
        
        for(int i = 0; i < ArraySize(manual_pairs); i++)
        {
            pairs[i] = manual_pairs[i] + detectedSuffix;
        }
    }
    
    // تحليل الفريمات
    string tf_array[];
    StringSplit(Timeframes, ',', tf_array);
    ArrayResize(timeframes, ArraySize(tf_array));
    
    for(int i = 0; i < ArraySize(tf_array); i++)
    {
        timeframes[i] = StringToTimeframe(tf_array[i]);
    }
    
    // تهيئة مصفوفة آخر فحص
    int totalCombinations = ArraySize(pairs) * ArraySize(timeframes);
    ArrayResize(lastCheckTime, totalCombinations);
    ArrayInitialize(lastCheckTime, 0);
    
    // عرض الإعدادات
    Print("====================================");
    Print("🚀 Forex ML Bot - Multi Pair Scanner v4.1");
    Print("📊 Server: ", ServerURL);
    Print("🔍 Symbol Suffix: '", detectedSuffix, "'");
    Print("🎯 Active Pairs (", ArraySize(pairs), "):");
    
    // عرض الأزواج المتاحة
    string pairsList = "";
    for(int i = 0; i < ArraySize(pairs) && i < 10; i++)
    {
        if(SymbolSelect(pairs[i], true))
        {
            pairsList += pairs[i] + " ";
        }
    }
    Print("   ", pairsList);
    
    Print("⏰ Timeframes: ", ArraySize(timeframes), " - ", Timeframes);
    Print("💪 Min Confidence: ", MinConfidence);
    Print("💰 Risk per trade: ", RiskPercent, "%");
    Print("====================================");
    
    // التحقق من الاتصال
    if(UseRemoteServer && !CheckServerConnection())
    {
        Print("❌ فشل الاتصال بالسيرفر!");
        return INIT_FAILED;
    }
    
    // تعيين Timer
    EventSetTimer(CheckIntervalSeconds);
    
    Print("✅ تم التهيئة بنجاح!");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| اكتشاف لاحقة الرموز                                             |
//+------------------------------------------------------------------+
string DetectSymbolSuffix()
{
    // محاولة أسماء شائعة مع لواحق مختلفة
    string testPairs[] = {"EURUSD", "GBPUSD", "USDJPY"};
    string suffixes[] = {"", "m", ".ecn", ".pro", "_ecn", "_pro"};
    
    for(int s = 0; s < ArraySize(suffixes); s++)
    {
        int found = 0;
        for(int p = 0; p < ArraySize(testPairs); p++)
        {
            string symbol = testPairs[p] + suffixes[s];
            if(SymbolSelect(symbol, true))
            {
                found++;
            }
        }
        
        if(found >= 2) // إذا وجدنا على الأقل 2 من 3
        {
            return suffixes[s];
        }
    }
    
    // إذا لم نجد، نستخدم رمز الرسم البياني الحالي
    string currentSymbol = Symbol();
    
    // البحث عن اللاحقة في الرمز الحالي
    for(int i = 0; i < ArraySize(testPairs); i++)
    {
        int pos = StringFind(currentSymbol, testPairs[i]);
        if(pos == 0)
        {
            return StringSubstr(currentSymbol, StringLen(testPairs[i]));
        }
    }
    
    return SymbolSuffix; // استخدام القيمة المدخلة
}

//+------------------------------------------------------------------+
//| اكتشاف الأزواج المتاحة تلقائياً                                 |
//+------------------------------------------------------------------+
void DetectAvailablePairs()
{
    string majorPairs[] = {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", 
        "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "EURNZD",
        "EURAUD", "EURCAD", "EURGBP", "GBPAUD", "GBPCAD",
        "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "AUDNZD"
    };
    
    ArrayResize(pairs, 0);
    
    for(int i = 0; i < ArraySize(majorPairs); i++)
    {
        string symbol = majorPairs[i] + detectedSuffix;
        
        if(SymbolSelect(symbol, true))
        {
            // التحقق من أن الرمز قابل للتداول
            if(SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
            {
                ArrayResize(pairs, ArraySize(pairs) + 1);
                pairs[ArraySize(pairs) - 1] = symbol;
            }
        }
    }
    
    Print("✅ تم اكتشاف ", ArraySize(pairs), " زوج عملات للتداول");
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    Print("====================================");
    Print("📊 إحصائيات الجلسة:");
    Print("   إشارات: ", totalSignals);
    Print("   صفقات: ", totalTrades);
    Print("   أزواج نشطة: ", ArraySize(pairs));
    Print("====================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // نستخدم Timer بدلاً من OnTick لتجنب الضغط الزائد
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    ScanAllPairs();
}

//+------------------------------------------------------------------+
//| فحص جميع الأزواج                                               |
//+------------------------------------------------------------------+
void ScanAllPairs()
{
    datetime currentTime = TimeCurrent();
    
    for(int p = 0; p < ArraySize(pairs); p++)
    {
        string symbol = pairs[p];
        
        // التحقق من وجود الرمز وأنه قابل للتداول
        if(!SymbolSelect(symbol, true))
        {
            continue;
        }
        
        for(int t = 0; t < ArraySize(timeframes); t++)
        {
            ENUM_TIMEFRAMES tf = timeframes[t];
            int index = p * ArraySize(timeframes) + t;
            
            // التحقق من آخر فحص
            if(currentTime - lastCheckTime[index] < CheckIntervalSeconds)
                continue;
            
            lastCheckTime[index] = currentTime;
            
            // فحص الزوج والفريم
            CheckPairTimeframe(symbol, tf);
        }
    }
}

//+------------------------------------------------------------------+
//| فحص زوج وفريم محدد                                              |
//+------------------------------------------------------------------+
void CheckPairTimeframe(string symbol, ENUM_TIMEFRAMES timeframe)
{
    // التحقق من عدد الصفقات المفتوحة
    int openTrades = CountOpenTrades(symbol);
    if(openTrades >= MaxTradesPerPair)
    {
        return;
    }
    
    // جلب البيانات
    MqlRates rates[];
    int copied = CopyRates(symbol, timeframe, 0, CandlesToSend, rates);
    
    if(copied < CandlesToSend)
    {
        return;
    }
    
    // إعداد طلب JSON
    string jsonRequest = PrepareRequest(symbol, timeframe, rates);
    
    // إرسال للسيرفر
    string response = SendToServer(jsonRequest);
    
    if(response != "")
    {
        ProcessServerResponse(symbol, timeframe, response);
    }
}

//+------------------------------------------------------------------+
//| إعداد طلب JSON                                                  |
//+------------------------------------------------------------------+
string PrepareRequest(string symbol, ENUM_TIMEFRAMES timeframe, MqlRates &rates[])
{
    // إزالة اللاحقة من اسم الرمز للسيرفر
    string cleanSymbol = symbol;
    if(detectedSuffix != "")
    {
        StringReplace(cleanSymbol, detectedSuffix, "");
    }
    
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";  // نرسل الرمز الكامل
    json += "\"clean_symbol\":\"" + cleanSymbol + "\",";  // والرمز النظيف
    json += "\"timeframe\":\"" + TimeframeToString(timeframe) + "\",";
    json += "\"candles\":[";
    
    for(int i = 0; i < ArraySize(rates); i++)
    {
        if(i > 0) json += ",";
        json += "{";
        json += "\"time\":\"" + TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES) + "\",";
        json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
        json += "}";
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| إرسال للسيرفر                                                   |
//+------------------------------------------------------------------+
string SendToServer(string jsonData)
{
    if(!UseRemoteServer) return "";
    
    char post[], result[];
    string headers = "Content-Type: application/json\r\n";
    
    StringToCharArray(jsonData, post);
    ArrayResize(post, ArraySize(post) - 1);
    
    ResetLastError();
    int res = WebRequest("POST", ServerURL + "/predict", headers, 5000, post, result, headers);
    
    if(res == -1)
    {
        int error = GetLastError();
        if(error != 5203 && EnableLogging) // تجاهل أخطاء الاتصال المتكررة
        {
            Print("❌ خطأ WebRequest: ", error);
        }
        return "";
    }
    
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| معالجة استجابة السيرفر                                          |
//+------------------------------------------------------------------+
void ProcessServerResponse(string symbol, ENUM_TIMEFRAMES timeframe, string response)
{
    // تحليل JSON
    string action = GetJsonValue(response, "action");
    double confidence = StringToDouble(GetJsonValue(response, "confidence"));
    double sl_price = StringToDouble(GetJsonValue(response, "sl_price"));
    double tp1_price = StringToDouble(GetJsonValue(response, "tp1_price"));
    double tp2_price = StringToDouble(GetJsonValue(response, "tp2_price"));
    
    if(action != "" && action != "NONE")
    {
        totalSignals++;
        
        if(EnableLogging)
        {
            Print("📊 ", symbol, " ", TimeframeToString(timeframe), 
                  " - Signal: ", action, " (", DoubleToString(confidence*100, 1), "%)");
        }
        
        // تنفيذ الصفقة
        if(confidence >= MinConfidence)
        {
            ExecuteTrade(symbol, action, sl_price, tp1_price, tp2_price);
        }
    }
}

//+------------------------------------------------------------------+
//| تنفيذ الصفقة                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(string symbol, string action, double sl, double tp1, double tp2)
{
    // التحقق من الصفقات المفتوحة
    if(CountOpenTrades(symbol) >= MaxTradesPerPair)
    {
        return;
    }
    
    // فحص ظروف السوق
    if(!IsGoodTimeToTrade(symbol))
    {
        Print("⚠️ ظروف السوق غير مناسبة للتداول: ", symbol);
        return;
    }
    
    // حساب حجم الصفقة
    double lotSize = CalculateLotSize(symbol, sl);
    if(lotSize <= 0) return;
    
    // استخدام الدالة المحسنة مع إعادة المحاولة
    bool result = false;
    if(action == "BUY")
    {
        result = OpenTradeWithRetry(symbol, ORDER_TYPE_BUY, lotSize, sl, tp1, "ML Signal");
    }
    else if(action == "SELL")
    {
        result = OpenTradeWithRetry(symbol, ORDER_TYPE_SELL, lotSize, sl, tp1, "ML Signal");
    }
    
    if(result)
    {
        totalTrades++;
    }
}

//+------------------------------------------------------------------+
//| حساب حجم الصفقة                                                  |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double sl_price)
{
    double price = SymbolInfoDouble(symbol, SYMBOL_BID);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double sl_points = MathAbs(price - sl_price) / point;
    
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * RiskPercent / 100;
    
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = riskAmount / (sl_points * tickValue);
    
    // تطبيق الحدود
    double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| عد الصفقات المفتوحة                                             |
//+------------------------------------------------------------------+
int CountOpenTrades(string symbol)
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(position.SelectByIndex(i))
        {
            if(position.Symbol() == symbol && position.Magic() == 123456)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| تحويل النص لفريم زمني                                          |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
    if(tf == "M1") return PERIOD_M1;
    if(tf == "M5") return PERIOD_M5;
    if(tf == "M15") return PERIOD_M15;
    if(tf == "M30") return PERIOD_M30;
    if(tf == "H1") return PERIOD_H1;
    if(tf == "H4") return PERIOD_H4;
    if(tf == "D1") return PERIOD_D1;
    if(tf == "W1") return PERIOD_W1;
    if(tf == "MN1") return PERIOD_MN1;
    return PERIOD_M15;
}

//+------------------------------------------------------------------+
//| تحويل الفريم لنص                                                |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
{
    switch(tf)
    {
        case PERIOD_M1: return "M1";
        case PERIOD_M5: return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1: return "H1";
        case PERIOD_H4: return "H4";
        case PERIOD_D1: return "D1";
        case PERIOD_W1: return "W1";
        case PERIOD_MN1: return "MN1";
    }
    return "M15";
}

//+------------------------------------------------------------------+
//| استخراج قيمة من JSON                                           |
//+------------------------------------------------------------------+
string GetJsonValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\":");
    if(keyPos == -1) return "";
    
    int start = keyPos + StringLen(key) + 3;
    int end = StringFind(json, ",", start);
    if(end == -1) end = StringFind(json, "}", start);
    
    string value = StringSubstr(json, start, end - start);
    StringReplace(value, "\"", "");
    StringTrimLeft(value);
    StringTrimRight(value);
    
    return value;
}

//+------------------------------------------------------------------+
//| فحص الاتصال بالسيرفر                                            |
//+------------------------------------------------------------------+
bool CheckServerConnection()
{
    char post[], result[];
    string headers = "Content-Type: application/json\r\n";
    
    ResetLastError();
    int res = WebRequest("GET", ServerURL + "/status", headers, 5000, post, result, headers);
    
    if(res == -1)
    {
        Print("❌ فشل الاتصال بالسيرفر: ", GetLastError());
        return false;
    }
    
    string response = CharArrayToString(result);
    if(StringFind(response, "running") >= 0)
    {
        Print("✅ السيرفر متصل ويعمل");
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| فتح صفقة مع إعادة المحاولة                                       |
//+------------------------------------------------------------------+
bool OpenTradeWithRetry(string symbol, ENUM_ORDER_TYPE orderType, double lotSize, 
                       double sl, double tp, string comment)
{
    for(int attempt = 1; attempt <= MaxRetries; attempt++)
    {
        // فحص الاتصال قبل المحاولة
        if(!CheckTerminalConnection())
        {
            Print("⚠️ Terminal not connected, waiting...");
            Sleep(RetryDelay);
            continue;
        }
        
        // تحديث الأسعار
        MqlTick tick;
        if(!SymbolInfoTick(symbol, tick))
        {
            Print("❌ Failed to get tick for ", symbol);
            return false;
        }
        
        // استخدام الأسعار الحالية
        double price = (orderType == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
        
        // إعادة حساب SL/TP بناءً على السعر الحالي
        double slDistance = MathAbs(sl - price);
        double tpDistance = MathAbs(tp - price);
        
        if(orderType == ORDER_TYPE_BUY)
        {
            sl = price - slDistance;
            tp = price + tpDistance;
        }
        else
        {
            sl = price + slDistance;
            tp = price - tpDistance;
        }
        
        // فتح الصفقة
        bool result = false;
        ResetLastError();
        
        if(orderType == ORDER_TYPE_BUY)
        {
            result = trade.Buy(lotSize, symbol, price, sl, tp, comment);
        }
        else
        {
            result = trade.Sell(lotSize, symbol, price, sl, tp, comment);
        }
        
        if(result)
        {
            string orderTypeStr = (orderType == ORDER_TYPE_BUY) ? "شراء" : "بيع";
            Print("✅ فتح صفقة ", orderTypeStr, " بنجاح: ", symbol, 
                  " المحاولة: ", attempt,
                  " Lot: ", DoubleToString(lotSize, 2),
                  " Price: ", DoubleToString(price, _Digits),
                  " SL: ", DoubleToString(sl, _Digits),
                  " TP: ", DoubleToString(tp, _Digits));
            return true;
        }
        else
        {
            int error = GetLastError();
            string retcode = trade.ResultRetcodeDescription();
            Print("❌ فشل فتح الصفقة - المحاولة ", attempt, "/", MaxRetries);
            Print("   Error: ", error, " - ", retcode);
            Print("   Retcode: ", trade.ResultRetcode());
            
            // معالجة أخطاء محددة
            if(error == TRADE_RETCODE_TIMEOUT || 
               error == TRADE_RETCODE_NO_REPLY ||
               trade.ResultRetcode() == 10004)  // Requote
            {
                if(attempt < MaxRetries)
                {
                    Print("⏳ إعادة المحاولة بعد ", RetryDelay/1000, " ثانية...");
                    Sleep(RetryDelay);
                    continue;
                }
            }
            else if(error == TRADE_RETCODE_MARKET_CLOSED)
            {
                Print("❌ السوق مغلق");
                return false;
            }
            else if(error == TRADE_RETCODE_NO_MONEY)
            {
                Print("❌ رصيد غير كافي");
                return false;
            }
            else
            {
                // خطأ آخر - لا نعيد المحاولة
                break;
            }
        }
    }
    
    Print("❌ فشل فتح الصفقة بعد ", MaxRetries, " محاولات");
    return false;
}

//+------------------------------------------------------------------+
//| فحص اتصال المنصة                                                |
//+------------------------------------------------------------------+
bool CheckTerminalConnection()
{
    if(!TerminalInfoInteger(TERMINAL_CONNECTED))
    {
        return false;
    }
    
    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
    {
        Print("❌ Trading not allowed on this account!");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| فحص ظروف السوق قبل التداول                                     |
//+------------------------------------------------------------------+
bool IsGoodTimeToTrade(string symbol)
{
    // فحص السبريد
    double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
    double avgSpread = 20; // متوسط السبريد المقبول بالنقاط
    
    if(spread > avgSpread * 3)
    {
        Print("⚠️ Spread too high for ", symbol, ": ", spread, " points");
        return false;
    }
    
    // فحص السوق مفتوح
    ENUM_SYMBOL_TRADE_MODE tradeMode = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
    if(tradeMode == SYMBOL_TRADE_MODE_DISABLED || tradeMode == SYMBOL_TRADE_MODE_CLOSEONLY)
    {
        Print("⚠️ Trading disabled for ", symbol);
        return false;
    }
    
    return true;
}
//+------------------------------------------------------------------+