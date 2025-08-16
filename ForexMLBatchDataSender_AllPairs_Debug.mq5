//+------------------------------------------------------------------+
//|                   ForexMLBatchDataSender_AllPairs_Debug.mq5      |
//|                   نسخة تشخيصية مع سجل مفصل                      |
//|                              Version 2.1                         |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property version   "2.1"
#property strict

// ============== إعدادات الخادم ==============
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpCollectionHours = 8;     // ساعات جمع البيانات
input int      InpBarsToSend = 500;        // عدد الشموع لكل زوج (مخفض للاختبار)
input int      InpBatchSize = 100;         // حجم الدفعة (مخفض للاختبار)
input bool     InpDebugMode = true;        // وضع التشخيص المفصل
input int      InpTestPairs = 3;           // عدد الأزواج للاختبار

// ============== إعدادات الأزواج ==============
input bool     InpUseAllAvailableSymbols = false;  // استخدام جميع الرموز المتاحة
input string   InpCustomSymbols = "EURUSD,GBPUSD,XAUUSD"; // رموز مخصصة للاختبار

// ============== إعدادات الأطر الزمنية ==============
input bool     InpAllTimeframes = false;   // جمع جميع الأطر الزمنية
input bool     InpH1Only = true;           // H1 فقط للاختبار

// متغيرات عامة
string g_symbols[];
int g_totalSymbols = 0;
datetime g_lastSendTime = 0;
int g_successCount = 0;
int g_failCount = 0;
ENUM_TIMEFRAMES g_timeframes[];
int g_totalTimeframes = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    PrintDebug("========== بدء تهيئة ForexML Batch Data Sender v2.1 Debug ==========");
    PrintDebug("📅 التاريخ والوقت: " + TimeToString(TimeCurrent()));
    PrintDebug("🌐 عنوان السيرفر: " + InpServerURL);
    PrintDebug("🔧 وضع التشخيص: " + (InpDebugMode ? "مفعل" : "معطل"));
    
    // التحقق من إعدادات WebRequest
    CheckWebRequestSettings();
    
    // جمع الرموز
    PrintDebug("\n📊 بدء جمع الرموز...");
    if(!CollectSymbols()) {
        PrintDebug("❌ فشل في جمع الرموز!");
        Alert("❌ فشل في جمع الرموز! تحقق من السجل");
        return INIT_FAILED;
    }
    
    // إعداد الأطر الزمنية
    PrintDebug("\n⏰ إعداد الأطر الزمنية...");
    SetupTimeframes();
    
    // عرض المعلومات
    PrintDebug("\n✅ ملخص التهيئة:");
    PrintDebug("   - عدد الرموز: " + IntegerToString(g_totalSymbols));
    PrintDebug("   - عدد الأطر الزمنية: " + IntegerToString(g_totalTimeframes));
    PrintDebug("   - إجمالي البيانات المتوقعة: " + IntegerToString(g_totalSymbols * g_totalTimeframes * InpBarsToSend) + " شمعة");
    
    // عرض الرموز المحددة
    PrintDebug("\n📋 الرموز التي سيتم إرسالها:");
    for(int i = 0; i < g_totalSymbols; i++) {
        PrintDebug("   " + IntegerToString(i+1) + ". " + g_symbols[i]);
    }
    
    // اختبار الاتصال
    PrintDebug("\n🔗 اختبار الاتصال بالسيرفر...");
    TestServerConnection();
    
    // تعيين المؤقت
    EventSetTimer(60); // كل دقيقة للاختبار
    
    PrintDebug("\n✅ تم إكمال التهيئة بنجاح!");
    PrintDebug("========================================\n");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Check WebRequest settings                                        |
//+------------------------------------------------------------------+
void CheckWebRequestSettings()
{
    PrintDebug("\n🔍 فحص إعدادات WebRequest:");
    
    // التحقق من أن URL مسموح
    string allowedURLs = TerminalInfoString(TERMINAL_DATA_PATH);
    PrintDebug("   - مسار البيانات: " + allowedURLs);
    
    // تحذير مهم
    PrintDebug("⚠️ تأكد من إضافة عنوان السيرفر في:");
    PrintDebug("   Tools -> Options -> Expert Advisors -> Allow WebRequest for listed URL:");
    PrintDebug("   " + InpServerURL);
}

//+------------------------------------------------------------------+
//| Test server connection                                           |
//+------------------------------------------------------------------+
void TestServerConnection()
{
    PrintDebug("🔗 اختبار الاتصال بـ: " + InpServerURL);
    
    string testData = "{\"test\": true, \"timestamp\": \"" + TimeToString(TimeCurrent()) + "\"}";
    char postData[], resultData[];
    string resultHeaders;
    
    ArrayResize(postData, StringToCharArray(testData, postData, 0, WHOLE_ARRAY, CP_UTF8) - 1);
    
    string headers = "Content-Type: application/json\r\n";
    headers += "User-Agent: ForexMLBot/2.1\r\n";
    
    PrintDebug("📤 إرسال طلب اختبار...");
    
    ResetLastError();
    int startTime = GetTickCount();
    int res = WebRequest("POST", InpServerURL, headers, 5000, postData, resultData, resultHeaders);
    int responseTime = GetTickCount() - startTime;
    
    PrintDebug("⏱️ وقت الاستجابة: " + IntegerToString(responseTime) + " ms");
    
    if(res == -1) {
        int error = GetLastError();
        PrintDebug("❌ فشل الاتصال! رمز الخطأ: " + IntegerToString(error));
        PrintDebug("   التفاصيل: " + GetErrorDescription(error));
    } else {
        PrintDebug("✅ نجح الاتصال! كود الاستجابة: " + IntegerToString(res));
        string response = CharArrayToString(resultData, 0, WHOLE_ARRAY, CP_UTF8);
        PrintDebug("📥 الاستجابة: " + StringSubstr(response, 0, 200));
    }
}

//+------------------------------------------------------------------+
//| Collect symbols function                                         |
//+------------------------------------------------------------------+
bool CollectSymbols()
{
    ArrayResize(g_symbols, 0);
    g_totalSymbols = 0;
    
    if(InpUseAllAvailableSymbols) {
        PrintDebug("🔍 جمع جميع الرموز المتاحة...");
        int totalInMarketWatch = SymbolsTotal(true);
        PrintDebug("   عدد الرموز في Market Watch: " + IntegerToString(totalInMarketWatch));
        
        for(int i = 0; i < totalInMarketWatch && g_totalSymbols < InpTestPairs; i++) {
            string symbol = SymbolName(i, true);
            if(IsValidSymbol(symbol)) {
                AddSymbolToList(symbol);
                PrintDebug("   ✅ تمت إضافة: " + symbol);
            }
        }
    } else {
        PrintDebug("🔍 استخدام الرموز المخصصة...");
        string customSymbols[];
        int count = StringSplit(InpCustomSymbols, ',', customSymbols);
        PrintDebug("   عدد الرموز المحددة: " + IntegerToString(count));
        
        for(int i = 0; i < count && g_totalSymbols < InpTestPairs; i++) {
            string symbol = customSymbols[i];
            StringTrimRight(symbol);
            StringTrimLeft(symbol);
            
            PrintDebug("   🔍 البحث عن: " + symbol);
            
            string actualSymbol = FindSymbolWithSuffix(symbol);
            if(actualSymbol != "") {
                if(SymbolSelect(actualSymbol, true)) {
                    AddSymbolToList(actualSymbol);
                    PrintDebug("   ✅ تمت إضافة: " + actualSymbol);
                } else {
                    PrintDebug("   ❌ فشل تحديد: " + actualSymbol);
                }
            } else {
                PrintDebug("   ❌ لم يتم العثور على: " + symbol);
            }
        }
    }
    
    g_totalSymbols = ArraySize(g_symbols);
    PrintDebug("📊 إجمالي الرموز المجموعة: " + IntegerToString(g_totalSymbols));
    
    return g_totalSymbols > 0;
}

//+------------------------------------------------------------------+
//| Check if symbol is valid                                        |
//+------------------------------------------------------------------+
bool IsValidSymbol(string symbol)
{
    // فحوصات أساسية
    if(StringLen(symbol) < 6) return false;
    if(!SymbolInfoInteger(symbol, SYMBOL_SELECT)) return false;
    
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    if(bid <= 0) return false;
    
    return true;
}

//+------------------------------------------------------------------+
//| Find symbol with suffix                                         |
//+------------------------------------------------------------------+
string FindSymbolWithSuffix(string baseSymbol)
{
    PrintDebug("      🔎 البحث عن متغيرات " + baseSymbol);
    
    // محاولة الرمز كما هو
    if(SymbolInfoInteger(baseSymbol, SYMBOL_EXIST)) {
        PrintDebug("      ✅ موجود كما هو: " + baseSymbol);
        return baseSymbol;
    }
    
    // قائمة اللواحق الشائعة
    string suffixes[] = {"", "m", ".m", "pro", ".pro", ".ecn", "ecn", ".fx", "fx"};
    
    for(int i = 0; i < ArraySize(suffixes); i++) {
        string testSymbol = baseSymbol + suffixes[i];
        if(SymbolInfoInteger(testSymbol, SYMBOL_EXIST)) {
            PrintDebug("      ✅ موجود مع لاحقة: " + testSymbol);
            return testSymbol;
        }
    }
    
    PrintDebug("      ❌ لم يتم العثور على أي متغير");
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
//| Setup timeframes                                                |
//+------------------------------------------------------------------+
void SetupTimeframes()
{
    ArrayResize(g_timeframes, 0);
    g_totalTimeframes = 0;
    
    if(InpH1Only) {
        AddTimeframe(PERIOD_H1);
        PrintDebug("   تم تحديد H1 فقط للاختبار");
    } else if(InpAllTimeframes) {
        AddTimeframe(PERIOD_M5);
        AddTimeframe(PERIOD_M15);
        AddTimeframe(PERIOD_H1);
        AddTimeframe(PERIOD_H4);
        PrintDebug("   تم تحديد أطر زمنية متعددة");
    }
    
    g_totalTimeframes = ArraySize(g_timeframes);
}

//+------------------------------------------------------------------+
//| Add timeframe                                                    |
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
    PrintDebug("\n========== بدء دورة الإرسال ==========");
    PrintDebug("⏰ الوقت: " + TimeToString(TimeCurrent()));
    
    SendAllData();
    
    PrintDebug("📊 النتائج: نجح=" + IntegerToString(g_successCount) + 
               ", فشل=" + IntegerToString(g_failCount));
    PrintDebug("========================================\n");
}

//+------------------------------------------------------------------+
//| Send all data                                                    |
//+------------------------------------------------------------------+
void SendAllData()
{
    g_successCount = 0;
    g_failCount = 0;
    
    for(int s = 0; s < g_totalSymbols; s++) {
        string symbol = g_symbols[s];
        PrintDebug("\n📈 معالجة " + symbol + "...");
        
        for(int t = 0; t < g_totalTimeframes; t++) {
            bool success = SendSymbolData(symbol, g_timeframes[t]);
            
            if(success) {
                g_successCount++;
            } else {
                g_failCount++;
            }
            
            Sleep(500); // تأخير نصف ثانية
        }
    }
}

//+------------------------------------------------------------------+
//| Send symbol data                                                |
//+------------------------------------------------------------------+
bool SendSymbolData(string symbol, ENUM_TIMEFRAMES timeframe)
{
    string tfStr = TimeframeToString(timeframe);
    PrintDebug("\n   🔄 " + symbol + " " + tfStr);
    
    // التحقق من توفر البيانات
    int bars = iBars(symbol, timeframe);
    PrintDebug("   📊 عدد الشموع المتاحة: " + IntegerToString(bars));
    
    if(bars < 100) {
        PrintDebug("   ❌ بيانات غير كافية!");
        return false;
    }
    
    // جمع البيانات
    PrintDebug("   📦 جمع البيانات...");
    string jsonData = PrepareJSONData(symbol, timeframe);
    
    if(jsonData == "") {
        PrintDebug("   ❌ فشل إعداد البيانات!");
        return false;
    }
    
    PrintDebug("   📏 حجم البيانات: " + IntegerToString(StringLen(jsonData)) + " حرف");
    
    // إرسال البيانات
    PrintDebug("   📤 إرسال البيانات...");
    
    char postData[], resultData[];
    string resultHeaders;
    
    ArrayResize(postData, StringToCharArray(jsonData, postData, 0, WHOLE_ARRAY, CP_UTF8) - 1);
    
    string headers = "Content-Type: application/json\r\n";
    headers += "Accept: application/json\r\n";
    headers += "User-Agent: ForexMLBot/2.1\r\n";
    
    ResetLastError();
    int startTime = GetTickCount();
    int res = WebRequest("POST", InpServerURL, headers, 10000, postData, resultData, resultHeaders);
    int responseTime = GetTickCount() - startTime;
    
    if(res == -1) {
        int error = GetLastError();
        PrintDebug("   ❌ فشل الإرسال!");
        PrintDebug("      رمز الخطأ: " + IntegerToString(error));
        PrintDebug("      الوصف: " + GetErrorDescription(error));
        return false;
    }
    
    string result = CharArrayToString(resultData, 0, WHOLE_ARRAY, CP_UTF8);
    
    PrintDebug("   📥 كود الاستجابة: " + IntegerToString(res));
    PrintDebug("   ⏱️ وقت الاستجابة: " + IntegerToString(responseTime) + " ms");
    
    if(res == 200 || res == 201) {
        PrintDebug("   ✅ نجح الإرسال!");
        if(StringLen(result) > 0) {
            PrintDebug("   📥 الرد: " + StringSubstr(result, 0, 100));
        }
        return true;
    } else {
        PrintDebug("   ❌ فشل - HTTP " + IntegerToString(res));
        if(StringLen(result) > 0) {
            PrintDebug("   📥 رسالة الخطأ: " + result);
        }
        return false;
    }
}

//+------------------------------------------------------------------+
//| Prepare JSON data                                               |
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
    json += "\"account\": " + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) + ",";
    json += "\"data\": [";
    
    for(int i = totalBars - 1; i >= 0; i--) {
        if(i < totalBars - 1) json += ",";
        
        json += "{";
        json += "\"time\": \"" + TimeToString(iTime(symbol, timeframe, i)) + "\",";
        json += "\"open\": " + DoubleToString(iOpen(symbol, timeframe, i), 5) + ",";
        json += "\"high\": " + DoubleToString(iHigh(symbol, timeframe, i), 5) + ",";
        json += "\"low\": " + DoubleToString(iLow(symbol, timeframe, i), 5) + ",";
        json += "\"close\": " + DoubleToString(iClose(symbol, timeframe, i), 5) + ",";
        json += "\"volume\": " + IntegerToString(iVolume(symbol, timeframe, i));
        json += "}";
    }
    
    json += "]}";
    
    return json;
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
        default:         return "H1";
    }
}

//+------------------------------------------------------------------+
//| Get error description                                           |
//+------------------------------------------------------------------+
string GetErrorDescription(int error)
{
    switch(error) {
        case 4060: return "Function is not allowed for call";
        case 4014: return "System function is not allowed for call";
        case 5200: return "Invalid URL";
        case 5201: return "Failed to connect to specified URL";
        case 5202: return "Timeout exceeded";
        case 5203: return "HTTP request failed";
        default:   return "Unknown error";
    }
}

//+------------------------------------------------------------------+
//| Debug print function                                            |
//+------------------------------------------------------------------+
void PrintDebug(string message)
{
    if(InpDebugMode) {
        Print(message);
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    PrintDebug("\n========== إيقاف EA ==========");
    PrintDebug("📊 إحصائيات الجلسة:");
    PrintDebug("   - عمليات ناجحة: " + IntegerToString(g_successCount));
    PrintDebug("   - عمليات فاشلة: " + IntegerToString(g_failCount));
    PrintDebug("👋 السبب: " + IntegerToString(reason));
    PrintDebug("=============================\n");
}

//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    // لا نحتاج OnTick - نستخدم Timer
}