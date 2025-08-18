//+------------------------------------------------------------------+
//|                     ForexMLBot_MultiPair_Scanner.mq5              |
//|                     نظام متعدد الأزواج والفريمات                |
//|                     يفحص جميع الفرص المتاحة                      |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property version   "4.0"
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
input string   TradingPairs = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,EURJPY,GBPJPY,EURNZD"; // الأزواج
input string   Timeframes = "M5,M15,M30,H1,H4";           // الفريمات
input bool     EnableLogging = true;                       // تفعيل السجلات

// متغيرات عامة
CTrade trade;
CPositionInfo position;
string pairs[];
ENUM_TIMEFRAMES timeframes[];
datetime lastCheckTime[];
int totalSignals = 0;
int totalTrades = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // تحليل الأزواج
    StringSplit(TradingPairs, ',', pairs);
    
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
    Print("🚀 Forex ML Bot - Multi Pair Scanner");
    Print("📊 Server: ", ServerURL);
    Print("🎯 Pairs: ", ArraySize(pairs), " - ", TradingPairs);
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
    
    Print("✅ تم التهيئة بنجاح!");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("====================================");
    Print("📊 إحصائيات الجلسة:");
    Print("   إشارات: ", totalSignals);
    Print("   صفقات: ", totalTrades);
    Print("====================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // فحص جميع الأزواج والفريمات
    ScanAllPairs();
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
        
        // التحقق من وجود الرمز
        if(!SymbolSelect(symbol, true))
        {
            if(EnableLogging) Print("⚠️ الرمز غير متاح: ", symbol);
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
        if(EnableLogging) Print("⚠️ بيانات غير كافية لـ ", symbol, " ", TimeframeToString(timeframe));
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
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + TimeframeToString(timeframe) + "\",";
    json += "\"candles\":[";
    
    for(int i = 0; i < ArraySize(rates); i++)
    {
        if(i > 0) json += ",";
        json += "{";
        json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
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
        if(EnableLogging) Print("❌ خطأ WebRequest: ", error);
        return "";
    }
    
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| معالجة استجابة السيرفر                                          |
//+------------------------------------------------------------------+
void ProcessServerResponse(string symbol, ENUM_TIMEFRAMES timeframe, string response)
{
    // تحليل JSON (مبسط)
    string action = GetJsonValue(response, "action");
    double confidence = StringToDouble(GetJsonValue(response, "confidence"));
    double sl_price = StringToDouble(GetJsonValue(response, "sl_price"));
    double tp1_price = StringToDouble(GetJsonValue(response, "tp1_price"));
    double tp2_price = StringToDouble(GetJsonValue(response, "tp2_price"));
    
    totalSignals++;
    
    if(EnableLogging)
    {
        Print("📊 ", symbol, " ", TimeframeToString(timeframe), 
              " - Signal: ", action, " (", DoubleToString(confidence*100, 1), "%)");
    }
    
    // تنفيذ الصفقة
    if(confidence >= MinConfidence && action != "NONE")
    {
        ExecuteTrade(symbol, action, sl_price, tp1_price, tp2_price);
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
        if(EnableLogging) Print("⚠️ وصلت للحد الأقصى من الصفقات لـ ", symbol);
        return;
    }
    
    // حساب حجم الصفقة
    double lotSize = CalculateLotSize(symbol, sl);
    if(lotSize <= 0) return;
    
    // فتح الصفقة
    trade.SetExpertMagicNumber(123456);
    
    if(action == "BUY")
    {
        if(trade.Buy(lotSize, symbol, 0, sl, tp1))
        {
            totalTrades++;
            Print("✅ فتح صفقة شراء: ", symbol, " Lot: ", lotSize);
        }
    }
    else if(action == "SELL")
    {
        if(trade.Sell(lotSize, symbol, 0, sl, tp1))
        {
            totalTrades++;
            Print("✅ فتح صفقة بيع: ", symbol, " Lot: ", lotSize);
        }
    }
}

//+------------------------------------------------------------------+
//| حساب حجم الصفقة                                                  |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double sl_price)
{
    double price = SymbolInfoDouble(symbol, SYMBOL_BID);
    double sl_pips = MathAbs(price - sl_price) / SymbolInfoDouble(symbol, SYMBOL_POINT) / 10;
    
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * RiskPercent / 100;
    
    double pipValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = riskAmount / (sl_pips * pipValue * 10);
    
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