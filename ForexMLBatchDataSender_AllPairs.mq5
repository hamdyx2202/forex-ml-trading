//+------------------------------------------------------------------+
//|                              ForexMLBatchDataSender_AllPairs.mq5 |
//|                   Comprehensive Data Collection for All Pairs    |
//|                              Version 2.0                         |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property version   "2.0"
#property strict

// ============== إعدادات الخادم ==============
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpCollectionHours = 8;     // ساعات جمع البيانات
input int      InpBarsToSend = 5000;       // عدد الشموع لكل زوج
input int      InpBatchSize = 1000;        // حجم الدفعة

// ============== إعدادات الأزواج ==============
input bool     InpUseAllAvailableSymbols = true;  // استخدام جميع الرموز المتاحة
input string   InpCustomSymbols = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,NZDUSD,USDCAD,EURJPY,GBPJPY,EURGBP,EURAUD,GBPAUD,AUDCAD,NZDCAD,XAUUSD,XAGUSD,USOIL,UKOIL,US30,NAS100,SP500,DAX,BTCUSD,ETHUSD"; // رموز مخصصة

// ============== إعدادات الأطر الزمنية ==============
input bool     InpAllTimeframes = true;    // جمع جميع الأطر الزمنية
input bool     InpM5 = true;               // M5
input bool     InpM15 = true;              // M15
input bool     InpH1 = true;               // H1
input bool     InpH4 = true;               // H4
input bool     InpD1 = false;              // D1

// ============== إعدادات متقدمة ==============
input bool     InpAutoDetectSuffix = true; // اكتشاف اللاحقة تلقائياً
input bool     InpFilterBySpread = true;   // تصفية حسب السبريد
input double   InpMaxSpreadPoints = 50;    // أقصى سبريد مسموح
input bool     InpSkipExoticPairs = false; // تخطي الأزواج الغريبة

// متغيرات عامة
string g_symbols[];
int g_totalSymbols = 0;
datetime g_lastSendTime = 0;
int g_currentSymbolIndex = 0;
ENUM_TIMEFRAMES g_timeframes[];
int g_totalTimeframes = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("===== ForexML Batch Data Sender v2.0 - All Pairs =====");
    
    // جمع الرموز
    if(!CollectSymbols()) {
        Alert("❌ فشل في جمع الرموز!");
        return INIT_FAILED;
    }
    
    // إعداد الأطر الزمنية
    SetupTimeframes();
    
    // عرض المعلومات
    Print("✅ تم العثور على ", g_totalSymbols, " رمز");
    Print("✅ سيتم جمع ", g_totalTimeframes, " إطار زمني لكل رمز");
    Print("✅ إجمالي البيانات: ", g_totalSymbols * g_totalTimeframes * InpBarsToSend, " شمعة");
    
    // عرض الرموز
    Print("\n📊 الرموز التي سيتم جمعها:");
    for(int i = 0; i < MathMin(g_totalSymbols, 20); i++) {
        Print("   ", i+1, ". ", g_symbols[i]);
    }
    if(g_totalSymbols > 20) {
        Print("   ... و ", g_totalSymbols - 20, " رمز آخر");
    }
    
    EventSetTimer(InpCollectionHours * 3600);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Collect all available symbols                                    |
//+------------------------------------------------------------------+
bool CollectSymbols()
{
    ArrayResize(g_symbols, 0);
    g_totalSymbols = 0;
    
    if(InpUseAllAvailableSymbols) {
        // جمع جميع الرموز المتاحة
        int totalInMarketWatch = SymbolsTotal(true);
        
        for(int i = 0; i < totalInMarketWatch; i++) {
            string symbol = SymbolName(i, true);
            
            // تصفية الرموز
            if(IsValidSymbol(symbol)) {
                AddSymbolToList(symbol);
            }
        }
        
        // إضافة رموز إضافية من القائمة الكاملة
        int totalSymbols = SymbolsTotal(false);
        string importantSymbols[] = {
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
            "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "GBPAUD", "AUDCAD", "NZDCAD",
            "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
            "USOIL", "UKOIL", "NGAS",
            "US30", "NAS100", "SP500", "DAX", "FTSE100", "NIKKEI",
            "BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD"
        };
        
        for(int i = 0; i < ArraySize(importantSymbols); i++) {
            string symbol = FindSymbolWithSuffix(importantSymbols[i]);
            if(symbol != "" && !IsSymbolInList(symbol)) {
                if(SymbolSelect(symbol, true)) {
                    AddSymbolToList(symbol);
                }
            }
        }
    }
    else {
        // استخدام الرموز المخصصة
        string customSymbols[];
        int count = StringSplit(InpCustomSymbols, ',', customSymbols);
        
        for(int i = 0; i < count; i++) {
            string symbol = customSymbols[i];
            // إزالة المسافات
            StringTrimRight(symbol);
            StringTrimLeft(symbol);
            
            // محاولة إيجاد الرمز مع اللاحقة
            string actualSymbol = FindSymbolWithSuffix(symbol);
            if(actualSymbol != "") {
                if(SymbolSelect(actualSymbol, true)) {
                    AddSymbolToList(actualSymbol);
                }
            }
        }
    }
    
    g_totalSymbols = ArraySize(g_symbols);
    return g_totalSymbols > 0;
}

//+------------------------------------------------------------------+
//| Check if symbol is valid for trading                            |
//+------------------------------------------------------------------+
bool IsValidSymbol(string symbol)
{
    // تخطي الرموز غير الصالحة
    if(StringLen(symbol) < 6) return false;
    if(StringFind(symbol, "#") >= 0) return false;  // تخطي الأسهم المؤقتة
    if(StringFind(symbol, ".") == 0) return false;  // تخطي الرموز التي تبدأ بنقطة
    
    // فحص السبريد
    if(InpFilterBySpread) {
        double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
        if(spread > InpMaxSpreadPoints) return false;
    }
    
    // تخطي الأزواج الغريبة إذا طُلب
    if(InpSkipExoticPairs) {
        string majors[] = {"EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"};
        bool isExotic = true;
        
        for(int i = 0; i < ArraySize(majors); i++) {
            if(StringFind(symbol, majors[i]) >= 0) {
                isExotic = false;
                break;
            }
        }
        
        // السماح بالمعادن والمؤشرات والطاقة
        if(StringFind(symbol, "XAU") >= 0 || StringFind(symbol, "XAG") >= 0 ||
           StringFind(symbol, "OIL") >= 0 || StringFind(symbol, "GAS") >= 0 ||
           StringFind(symbol, "US30") >= 0 || StringFind(symbol, "NAS") >= 0 ||
           StringFind(symbol, "DAX") >= 0 || StringFind(symbol, "BTC") >= 0 ||
           StringFind(symbol, "ETH") >= 0) {
            isExotic = false;
        }
        
        if(isExotic) return false;
    }
    
    // التحقق من إمكانية التداول
    if(!SymbolInfoInteger(symbol, SYMBOL_SELECT)) return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Find symbol with broker suffix                                  |
//+------------------------------------------------------------------+
string FindSymbolWithSuffix(string baseSymbol)
{
    // محاولة الرمز كما هو
    if(SymbolInfoInteger(baseSymbol, SYMBOL_EXIST)) {
        return baseSymbol;
    }
    
    if(InpAutoDetectSuffix) {
        // قائمة اللواحق الشائعة
        string suffixes[] = {"", "m", ".m", "pro", ".pro", ".ecn", "ecn", ".fx", "fx", ".r", ".i", ".a", ".c"};
        
        for(int i = 0; i < ArraySize(suffixes); i++) {
            string testSymbol = baseSymbol + suffixes[i];
            if(SymbolInfoInteger(testSymbol, SYMBOL_EXIST)) {
                return testSymbol;
            }
        }
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| Add symbol to list                                              |
//+------------------------------------------------------------------+
void AddSymbolToList(string symbol)
{
    int size = ArraySize(g_symbols);
    ArrayResize(g_symbols, size + 1);
    g_symbols[size] = symbol;
}

//+------------------------------------------------------------------+
//| Check if symbol already in list                                 |
//+------------------------------------------------------------------+
bool IsSymbolInList(string symbol)
{
    for(int i = 0; i < ArraySize(g_symbols); i++) {
        if(g_symbols[i] == symbol) return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Setup timeframes array                                          |
//+------------------------------------------------------------------+
void SetupTimeframes()
{
    ArrayResize(g_timeframes, 0);
    g_totalTimeframes = 0;
    
    if(InpAllTimeframes || InpM5) AddTimeframe(PERIOD_M5);
    if(InpAllTimeframes || InpM15) AddTimeframe(PERIOD_M15);
    if(InpAllTimeframes || InpH1) AddTimeframe(PERIOD_H1);
    if(InpAllTimeframes || InpH4) AddTimeframe(PERIOD_H4);
    if(InpAllTimeframes || InpD1) AddTimeframe(PERIOD_D1);
    
    g_totalTimeframes = ArraySize(g_timeframes);
}

//+------------------------------------------------------------------+
//| Add timeframe to array                                          |
//+------------------------------------------------------------------+
void AddTimeframe(ENUM_TIMEFRAMES tf)
{
    int size = ArraySize(g_timeframes);
    ArrayResize(g_timeframes, size + 1);
    g_timeframes[size] = tf;
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    SendAllData();
}

//+------------------------------------------------------------------+
//| Send all collected data                                         |
//+------------------------------------------------------------------+
void SendAllData()
{
    Print("\n========== بدء إرسال البيانات ==========");
    Print("⏰ الوقت: ", TimeToString(TimeCurrent()));
    
    int totalSent = 0;
    int totalFailed = 0;
    
    // إرسال البيانات لكل رمز
    for(int s = 0; s < g_totalSymbols; s++) {
        string symbol = g_symbols[s];
        
        // إرسال لكل إطار زمني
        for(int t = 0; t < g_totalTimeframes; t++) {
            bool success = SendSymbolData(symbol, g_timeframes[t]);
            
            if(success) {
                totalSent++;
            } else {
                totalFailed++;
            }
            
            // تأخير صغير بين الطلبات
            Sleep(100);
        }
        
        // عرض التقدم
        if((s + 1) % 10 == 0 || s == g_totalSymbols - 1) {
            Print("📊 التقدم: ", s + 1, "/", g_totalSymbols, " رمز");
        }
    }
    
    Print("\n✅ اكتمل الإرسال!");
    Print("   نجح: ", totalSent, " طلب");
    Print("   فشل: ", totalFailed, " طلب");
    Print("========================================\n");
}

//+------------------------------------------------------------------+
//| Send data for specific symbol and timeframe                     |
//+------------------------------------------------------------------+
bool SendSymbolData(string symbol, ENUM_TIMEFRAMES timeframe)
{
    // التحقق من توفر البيانات
    int bars = iBars(symbol, timeframe);
    if(bars < 100) {
        Print("⚠️ ", symbol, " ", TimeframeToString(timeframe), " - بيانات غير كافية");
        return false;
    }
    
    // جمع البيانات
    string jsonData = PrepareJSONData(symbol, timeframe);
    if(jsonData == "") return false;
    
    // إرسال البيانات
    char postData[], resultData[];
    string resultHeaders;
    
    ArrayResize(postData, StringToCharArray(jsonData, postData, 0, WHOLE_ARRAY, CP_UTF8) - 1);
    
    string headers = "Content-Type: application/json\r\n";
    headers += "Accept: application/json\r\n";
    
    ResetLastError();
    int res = WebRequest("POST", InpServerURL, headers, 10000, postData, resultData, resultHeaders);
    
    if(res == -1) {
        int error = GetLastError();
        Print("❌ ", symbol, " ", TimeframeToString(timeframe), " - خطأ: ", error);
        return false;
    }
    
    string result = CharArrayToString(resultData, 0, WHOLE_ARRAY, CP_UTF8);
    
    if(res == 200 || res == 201) {
        Print("✅ ", symbol, " ", TimeframeToString(timeframe), " - تم الإرسال");
        return true;
    } else {
        Print("❌ ", symbol, " ", TimeframeToString(timeframe), " - HTTP ", res);
        return false;
    }
}

//+------------------------------------------------------------------+
//| Prepare JSON data for sending                                   |
//+------------------------------------------------------------------+
string PrepareJSONData(string symbol, ENUM_TIMEFRAMES timeframe)
{
    int totalBars = MathMin(iBars(symbol, timeframe), InpBarsToSend);
    
    string json = "{";
    json += "\"symbol\": \"" + symbol + "\",";
    json += "\"timeframe\": \"" + TimeframeToString(timeframe) + "\",";
    json += "\"timestamp\": \"" + TimeToString(TimeCurrent()) + "\",";
    json += "\"bars_count\": " + IntegerToString(totalBars) + ",";
    json += "\"broker\": \"" + AccountInfoString(ACCOUNT_COMPANY) + "\",";
    json += "\"features_version\": 75,";  // للنظام الجديد
    json += "\"data\": [";
    
    // إرسال البيانات على دفعات
    int batchCount = 0;
    
    for(int i = totalBars - 1; i >= 0; i--) {
        if(batchCount > 0) json += ",";
        
        json += "{";
        json += "\"time\": \"" + TimeToString(iTime(symbol, timeframe, i)) + "\",";
        json += "\"open\": " + DoubleToString(iOpen(symbol, timeframe, i), 5) + ",";
        json += "\"high\": " + DoubleToString(iHigh(symbol, timeframe, i), 5) + ",";
        json += "\"low\": " + DoubleToString(iLow(symbol, timeframe, i), 5) + ",";
        json += "\"close\": " + DoubleToString(iClose(symbol, timeframe, i), 5) + ",";
        json += "\"volume\": " + IntegerToString(iVolume(symbol, timeframe, i));
        json += "}";
        
        batchCount++;
        
        // إرسال الدفعة إذا وصلت للحد الأقصى
        if(batchCount >= InpBatchSize && i > 0) {
            json += "],\"partial\": true}";
            
            if(!SendBatch(json)) return "";
            
            // بدء دفعة جديدة
            json = "{";
            json += "\"symbol\": \"" + symbol + "\",";
            json += "\"timeframe\": \"" + TimeframeToString(timeframe) + "\",";
            json += "\"timestamp\": \"" + TimeToString(TimeCurrent()) + "\",";
            json += "\"continuation\": true,";
            json += "\"data\": [";
            batchCount = 0;
        }
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Send batch of data                                              |
//+------------------------------------------------------------------+
bool SendBatch(string jsonData)
{
    char postData[], resultData[];
    string resultHeaders;
    
    ArrayResize(postData, StringToCharArray(jsonData, postData, 0, WHOLE_ARRAY, CP_UTF8) - 1);
    
    string headers = "Content-Type: application/json\r\n";
    
    int res = WebRequest("POST", InpServerURL, headers, 10000, postData, resultData, resultHeaders);
    
    return (res == 200 || res == 201);
}

//+------------------------------------------------------------------+
//| Convert timeframe to string                                     |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
{
    switch(tf) {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        case PERIOD_W1:  return "W1";
        case PERIOD_MN1: return "MN1";
        default:         return "M5";
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("👋 Batch Data Sender stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    // في وضع الاختبار، إرسال البيانات مرة واحدة
    static bool sent = false;
    if(!sent && g_totalSymbols > 0) {
        SendAllData();
        sent = true;
    }
}