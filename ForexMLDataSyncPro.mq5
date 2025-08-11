//+------------------------------------------------------------------+
//|                                         ForexMLDataSyncPro.mq5   |
//|                نسخة احترافية - تدعم جميع الأزواج تلقائياً      |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System Pro"
#property version   "3.00"
#property description "يرسل جميع أزواج الفوركس والمعادن من MT5 إلى خادم Linux"

// إعدادات الخادم
input string   ServerURL = "http://YOUR_VPS_IP:5000";     // عنوان خادم Linux
input string   APIKey = "your_secure_api_key";            // مفتاح API للأمان
input int      UpdateIntervalSeconds = 300;                // فترة التحديث (ثواني)
input int      HistoryDays = 1095;                        // أيام البيانات التاريخية (3 سنوات)
input bool     SendHistoricalData = true;                 // إرسال البيانات التاريخية
input bool     SendLiveData = true;                       // إرسال البيانات الحية
input bool     AutoStart = true;                          // البدء التلقائي

// إعدادات الفلترة
input bool     IncludeMajors = true;                      // تضمين الأزواج الرئيسية
input bool     IncludeCrosses = true;                     // تضمين الأزواج المتقاطعة
input bool     IncludeMetals = true;                      // تضمين المعادن
input bool     IncludeExotics = true;                     // تضمين الأزواج الغريبة
input bool     AutoDetectSuffix = true;                   // اكتشاف النهايات تلقائياً
input string   CustomSuffix = "";                          // نهاية مخصصة (اتركها فارغة للتلقائي)

// إعدادات متقدمة
input int      BatchSize = 1000;                          // حجم دفعة الإرسال
input int      MaxSymbols = 100;                          // الحد الأقصى للرموز
input bool     ShowProgress = true;                       // عرض شريط التقدم

// قائمة الرموز المكتشفة
string DiscoveredSymbols[];
int totalSymbolsFound = 0;
int currentSymbolIndex = 0;

// الأطر الزمنية
ENUM_TIMEFRAMES Timeframes[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H4, PERIOD_D1};

// متغيرات عامة
datetime lastUpdateTime = 0;
bool isRunning = false;
int totalSentBars = 0;
int failedRequests = 0;
string detectedSuffix = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // اكتشاف جميع الرموز المتاحة
    DiscoverAllSymbols();
    
    // إنشاء لوحة التحكم المتقدمة
    CreateAdvancedPanel();
    
    // البدء التلقائي
    if(AutoStart && totalSymbolsFound > 0)
    {
        StartDataSync();
    }
    
    // تعيين Timer للتحديث الدوري
    EventSetTimer(UpdateIntervalSeconds);
    
    Print("✅ ForexML Data Sync Pro initialized");
    Print("📡 Server URL: ", ServerURL);
    Print("📊 Total symbols discovered: ", totalSymbolsFound);
    Print("🔍 Detected suffix: ", detectedSuffix == "" ? "None" : detectedSuffix);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| اكتشاف جميع الرموز المتاحة بذكاء                                |
//+------------------------------------------------------------------+
void DiscoverAllSymbols()
{
    ArrayResize(DiscoveredSymbols, 0);
    totalSymbolsFound = 0;
    
    // اكتشاف النهاية المستخدمة تلقائياً
    if(AutoDetectSuffix && CustomSuffix == "")
    {
        detectedSuffix = DetectSymbolSuffix();
    }
    else if(CustomSuffix != "")
    {
        detectedSuffix = CustomSuffix;
    }
    
    // البحث في جميع الرموز
    int totalInPlatform = SymbolsTotal(true);
    
    for(int i = 0; i < totalInPlatform && totalSymbolsFound < MaxSymbols; i++)
    {
        string symbol = SymbolName(i, true);
        
        if(IsValidForexOrMetalSymbol(symbol))
        {
            // تفعيل الرمز في Market Watch
            if(SymbolSelect(symbol, true))
            {
                ArrayResize(DiscoveredSymbols, totalSymbolsFound + 1);
                DiscoveredSymbols[totalSymbolsFound] = symbol;
                totalSymbolsFound++;
            }
        }
    }
    
    // ترتيب الرموز أبجدياً
    SortSymbols();
    
    Print("🔍 Symbol discovery completed:");
    Print("   • Total symbols in platform: ", totalInPlatform);
    Print("   • Valid Forex/Metal symbols: ", totalSymbolsFound);
    
    // عرض بعض الأمثلة
    int showCount = MathMin(5, totalSymbolsFound);
    for(int i = 0; i < showCount; i++)
    {
        Print("   • ", DiscoveredSymbols[i]);
    }
    if(totalSymbolsFound > showCount)
    {
        Print("   • ... and ", totalSymbolsFound - showCount, " more");
    }
}

//+------------------------------------------------------------------+
//| اكتشاف النهاية المستخدمة في المنصة                             |
//+------------------------------------------------------------------+
string DetectSymbolSuffix()
{
    // البحث عن EURUSD مع نهايات مختلفة
    string testSymbol = "EURUSD";
    string suffixes[] = {"", ".", "..", "m", "_m", "pro", ".pro", "ecn", ".ecn", "-5", ".r", "cash", ".cash", ".a", ".i"};
    
    for(int i = 0; i < ArraySize(suffixes); i++)
    {
        string checkSymbol = testSymbol + suffixes[i];
        
        // التحقق من وجود الرمز
        for(int j = 0; j < SymbolsTotal(true); j++)
        {
            if(SymbolName(j, true) == checkSymbol)
            {
                Print("🔍 Detected symbol suffix: '", suffixes[i], "'");
                return suffixes[i];
            }
        }
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| التحقق من صحة رمز الفوركس أو المعدن                           |
//+------------------------------------------------------------------+
bool IsValidForexOrMetalSymbol(string symbol)
{
    // التحقق من طول الرمز
    int len = StringLen(symbol);
    if(len < 6 || len > 15) return false;
    
    // التحقق من نوع الرمز
    ENUM_SYMBOL_CALC_MODE calcMode = (ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_CALC_MODE);
    
    bool isForexOrCFD = (calcMode == SYMBOL_CALC_MODE_FOREX || 
                         calcMode == SYMBOL_CALC_MODE_CFD ||
                         calcMode == SYMBOL_CALC_MODE_CFDINDEX ||
                         calcMode == SYMBOL_CALC_MODE_CFDLEVERAGE);
    
    if(!isForexOrCFD) return false;
    
    // التحقق من أن الرمز قابل للتداول
    if(SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
        return false;
    
    // تحويل إلى أحرف كبيرة للمقارنة
    string upperSymbol = symbol;
    StringToUpper(upperSymbol);
    
    // قائمة العملات المعروفة
    string currencies[] = {"EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", 
                          "CNH", "CNY", "HKD", "SGD", "MXN", "NOK", "SEK", "DKK",
                          "PLN", "CZK", "HUF", "TRY", "ZAR", "RUB", "INR", "BRL"};
    
    // التحقق من أزواج العملات
    bool isCurrencyPair = false;
    for(int i = 0; i < ArraySize(currencies) && !isCurrencyPair; i++)
    {
        for(int j = 0; j < ArraySize(currencies); j++)
        {
            if(i != j && StringFind(upperSymbol, currencies[i] + currencies[j]) == 0)
            {
                isCurrencyPair = true;
                
                // تطبيق الفلاتر
                if(!IncludeMajors && IsMajorPair(currencies[i], currencies[j]))
                    return false;
                if(!IncludeCrosses && IsCrossPair(currencies[i], currencies[j]))
                    return false;
                if(!IncludeExotics && IsExoticPair(currencies[i], currencies[j]))
                    return false;
                    
                break;
            }
        }
    }
    
    // التحقق من المعادن
    bool isMetal = false;
    string metals[] = {"XAU", "XAG", "GOLD", "SILVER", "XPT", "XPD", "PLATINUM", "PALLADIUM"};
    
    for(int i = 0; i < ArraySize(metals); i++)
    {
        if(StringFind(upperSymbol, metals[i]) >= 0)
        {
            isMetal = true;
            if(!IncludeMetals) return false;
            break;
        }
    }
    
    return isCurrencyPair || isMetal;
}

//+------------------------------------------------------------------+
//| التحقق من الأزواج الرئيسية                                     |
//+------------------------------------------------------------------+
bool IsMajorPair(string curr1, string curr2)
{
    string pair = curr1 + curr2;
    string majors[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"};
    
    for(int i = 0; i < ArraySize(majors); i++)
    {
        if(pair == majors[i] || curr2 + curr1 == majors[i])
            return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| التحقق من الأزواج المتقاطعة                                     |
//+------------------------------------------------------------------+
bool IsCrossPair(string curr1, string curr2)
{
    // زوج متقاطع = لا يحتوي على USD
    return (curr1 != "USD" && curr2 != "USD");
}

//+------------------------------------------------------------------+
//| التحقق من الأزواج الغريبة                                      |
//+------------------------------------------------------------------+
bool IsExoticPair(string curr1, string curr2)
{
    string exoticCurrencies[] = {"TRY", "ZAR", "MXN", "HKD", "SGD", "NOK", "SEK", "DKK", 
                                 "PLN", "CZK", "HUF", "RUB", "INR", "BRL", "CNH", "CNY"};
    
    for(int i = 0; i < ArraySize(exoticCurrencies); i++)
    {
        if(curr1 == exoticCurrencies[i] || curr2 == exoticCurrencies[i])
            return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| ترتيب الرموز أبجدياً                                           |
//+------------------------------------------------------------------+
void SortSymbols()
{
    for(int i = 0; i < totalSymbolsFound - 1; i++)
    {
        for(int j = i + 1; j < totalSymbolsFound; j++)
        {
            if(DiscoveredSymbols[i] > DiscoveredSymbols[j])
            {
                string temp = DiscoveredSymbols[i];
                DiscoveredSymbols[i] = DiscoveredSymbols[j];
                DiscoveredSymbols[j] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| إنشاء لوحة تحكم متقدمة                                          |
//+------------------------------------------------------------------+
void CreateAdvancedPanel()
{
    int x = 10, y = 30;
    
    // العنوان
    CreateLabel("FXML_Title", "ForexML Data Sync Pro", x, y, clrGold, 14);
    
    // معلومات الرموز
    y += 25;
    CreateLabel("FXML_SymbolInfo", "Symbols: 0/" + IntegerToString(totalSymbolsFound), x, y, clrWhite, 10);
    
    // حالة الاتصال
    y += 20;
    CreateLabel("FXML_Status", "Status: Ready", x, y, clrWhite, 10);
    
    // عداد البيانات
    y += 20;
    CreateLabel("FXML_Counter", "Sent: 0 bars", x, y, clrWhite, 10);
    
    // شريط التقدم
    y += 20;
    CreateProgressBar("FXML_Progress", x, y, 200, 20);
    
    // الأزرار
    y += 30;
    CreateButton("FXML_Start", "Start Sync", x, y, 95, 25, clrGreen);
    CreateButton("FXML_Stop", "Stop Sync", x + 105, y, 95, 25, clrRed);
    
    y += 30;
    CreateButton("FXML_SendHistory", "Send All History", x, y, 95, 25, clrBlue);
    CreateButton("FXML_TestConn", "Test Connection", x + 105, y, 95, 25, clrOrange);
    
    y += 30;
    CreateButton("FXML_Refresh", "Refresh Symbols", x, y, 200, 25, clrPurple);
    
    // قائمة الرموز الحالية
    y += 35;
    CreateLabel("FXML_CurrentSymbol", "Current: None", x, y, clrYellow, 10);
}

//+------------------------------------------------------------------+
//| إنشاء شريط تقدم                                                 |
//+------------------------------------------------------------------+
void CreateProgressBar(string name, int x, int y, int width, int height)
{
    // الخلفية
    string bgName = name + "_BG";
    ObjectCreate(0, bgName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
    ObjectSetInteger(0, bgName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, bgName, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, bgName, OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, bgName, OBJPROP_XSIZE, width);
    ObjectSetInteger(0, bgName, OBJPROP_YSIZE, height);
    ObjectSetInteger(0, bgName, OBJPROP_BGCOLOR, clrDarkGray);
    ObjectSetInteger(0, bgName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    
    // شريط التقدم
    ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x + 2);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y + 2);
    ObjectSetInteger(0, name, OBJPROP_XSIZE, 0);
    ObjectSetInteger(0, name, OBJPROP_YSIZE, height - 4);
    ObjectSetInteger(0, name, OBJPROP_BGCOLOR, clrLime);
    ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
}

//+------------------------------------------------------------------+
//| تحديث شريط التقدم                                               |
//+------------------------------------------------------------------+
void UpdateProgressBar(double percent)
{
    int maxWidth = 196; // 200 - 4 for borders
    int currentWidth = (int)(maxWidth * percent / 100.0);
    ObjectSetInteger(0, "FXML_Progress", OBJPROP_XSIZE, currentWidth);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    ObjectsDeleteAll(0, "FXML_");
    Print("ForexML Data Sync Pro stopped. Total bars sent: ", totalSentBars);
}

//+------------------------------------------------------------------+
//| Timer function                                                    |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(isRunning && SendLiveData)
    {
        SendRecentData();
    }
}

//+------------------------------------------------------------------+
//| Chart event function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
    if(id == CHARTEVENT_OBJECT_CLICK)
    {
        if(sparam == "FXML_Start")
        {
            StartDataSync();
        }
        else if(sparam == "FXML_Stop")
        {
            StopDataSync();
        }
        else if(sparam == "FXML_SendHistory")
        {
            SendAllHistoricalData();
        }
        else if(sparam == "FXML_TestConn")
        {
            TestConnection();
        }
        else if(sparam == "FXML_Refresh")
        {
            RefreshSymbols();
        }
    }
}

//+------------------------------------------------------------------+
//| Start data synchronization                                        |
//+------------------------------------------------------------------+
void StartDataSync()
{
    if(totalSymbolsFound == 0)
    {
        Alert("No symbols found! Click 'Refresh Symbols' first.");
        return;
    }
    
    isRunning = true;
    UpdateStatus("Running", clrLime);
    
    if(SendHistoricalData)
    {
        SendAllHistoricalData();
    }
    
    Print("✅ Data sync started for ", totalSymbolsFound, " symbols");
}

//+------------------------------------------------------------------+
//| Stop data synchronization                                         |
//+------------------------------------------------------------------+
void StopDataSync()
{
    isRunning = false;
    UpdateStatus("Stopped", clrRed);
    Print("⏹ Data sync stopped");
}

//+------------------------------------------------------------------+
//| Send all historical data                                          |
//+------------------------------------------------------------------+
void SendAllHistoricalData()
{
    if(totalSymbolsFound == 0)
    {
        Alert("No symbols found!");
        return;
    }
    
    UpdateStatus("Sending history...", clrYellow);
    currentSymbolIndex = 0;
    
    datetime endTime = TimeCurrent();
    datetime startTime = endTime - HistoryDays * 24 * 3600;
    
    // Send data for all discovered symbols
    for(int i = 0; i < totalSymbolsFound; i++)
    {
        string symbol = DiscoveredSymbols[i];
        currentSymbolIndex = i;
        
        // Update current symbol display
        ObjectSetString(0, "FXML_CurrentSymbol", OBJPROP_TEXT, "Current: " + symbol);
        
        // Update symbol progress
        UpdateSymbolProgress(i, totalSymbolsFound);
        
        // Check if symbol is still valid
        if(!SymbolSelect(symbol, true))
        {
            Print("⚠️ Cannot select symbol: ", symbol);
            continue;
        }
        
        for(int j = 0; j < ArraySize(Timeframes); j++)
        {
            ENUM_TIMEFRAMES tf = Timeframes[j];
            string timeframe = TimeframeToString(tf);
            
            // Get historical data
            MqlRates rates[];
            int copied = CopyRates(symbol, tf, startTime, endTime, rates);
            
            if(copied <= 0)
            {
                Print("❌ No data for ", symbol, " ", timeframe);
                continue;
            }
            
            // Send data in batches
            int batches = (copied + BatchSize - 1) / BatchSize;
            
            for(int batch = 0; batch < batches; batch++)
            {
                int start = batch * BatchSize;
                int end = MathMin(start + BatchSize, copied);
                
                // Create JSON for batch
                string jsonData = CreateBatchJSON(symbol, timeframe, rates, start, end);
                
                // Send data
                if(SendDataToServer("/api/historical_data", jsonData))
                {
                    totalSentBars += (end - start);
                    UpdateCounter();
                }
                else
                {
                    failedRequests++;
                    Print("❌ Failed to send ", symbol, " ", timeframe, " batch ", batch+1);
                }
                
                // Small delay to avoid overloading
                Sleep(100);
            }
        }
        
        // Update overall progress
        if(ShowProgress)
        {
            double progress = (double)(i + 1) / totalSymbolsFound * 100;
            UpdateProgressBar(progress);
            UpdateStatus(StringFormat("Sending: %.1f%%", progress), clrYellow);
        }
    }
    
    UpdateStatus("History sent", clrLime);
    UpdateProgressBar(100);
    ObjectSetString(0, "FXML_CurrentSymbol", OBJPROP_TEXT, "Current: Completed");
    
    Print("✅ Historical data sync completed. Total bars: ", totalSentBars);
}

//+------------------------------------------------------------------+
//| Send recent data                                                  |
//+------------------------------------------------------------------+
void SendRecentData()
{
    UpdateStatus("Updating...", clrYellow);
    
    // Send data for all discovered symbols
    for(int i = 0; i < totalSymbolsFound; i++)
    {
        string symbol = DiscoveredSymbols[i];
        
        for(int j = 0; j < ArraySize(Timeframes); j++)
        {
            ENUM_TIMEFRAMES tf = Timeframes[j];
            string timeframe = TimeframeToString(tf);
            
            // Get last 100 bars
            MqlRates rates[];
            int copied = CopyRates(symbol, tf, 0, 100, rates);
            
            if(copied > 0)
            {
                string jsonData = CreateBatchJSON(symbol, timeframe, rates, 0, copied);
                
                if(SendDataToServer("/api/live_data", jsonData))
                {
                    totalSentBars += copied;
                    UpdateCounter();
                }
            }
        }
    }
    
    lastUpdateTime = TimeCurrent();
    UpdateStatus("Running", clrLime);
}

//+------------------------------------------------------------------+
//| Create batch JSON                                                 |
//+------------------------------------------------------------------+
string CreateBatchJSON(string symbol, string timeframe, MqlRates &rates[], int start, int end)
{
    string json = "{";
    json += "\"api_key\":\"" + APIKey + "\",";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + timeframe + "\",";
    json += "\"data\":[";
    
    for(int i = start; i < end; i++)
    {
        if(i > start) json += ",";
        
        json += "{";
        json += "\"time\":" + IntegerToString(rates[i].time) + ",";
        json += "\"open\":" + DoubleToString(rates[i].open, 5) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, 5) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, 5) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, 5) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume) + ",";
        json += "\"spread\":" + IntegerToString(rates[i].spread);
        json += "}";
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Send data to server                                               |
//+------------------------------------------------------------------+
bool SendDataToServer(string endpoint, string jsonData)
{
    string headers = "Content-Type: application/json\r\n";
    char post[], result[];
    
    StringToCharArray(jsonData, post);
    
    string url = ServerURL + endpoint;
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, 5000, post, result, headers);
    
    if(res == 200)
    {
        return true;
    }
    else
    {
        int error = GetLastError();
        if(error == 4060)
        {
            Print("❌ URL not allowed. Add to MT5: Tools > Options > Expert Advisors > Allow WebRequest");
            Print("Add URL: ", ServerURL);
        }
        else
        {
            Print("❌ Server error: ", res, ", Error: ", error);
        }
        return false;
    }
}

//+------------------------------------------------------------------+
//| Test connection                                                   |
//+------------------------------------------------------------------+
void TestConnection()
{
    UpdateStatus("Testing...", clrYellow);
    
    string json = "{\"api_key\":\"" + APIKey + "\",\"test\":true}";
    
    if(SendDataToServer("/api/test", json))
    {
        UpdateStatus("Connected", clrLime);
        Alert("✅ Connection successful!");
    }
    else
    {
        UpdateStatus("Failed", clrRed);
        Alert("❌ Connection failed!");
    }
}

//+------------------------------------------------------------------+
//| Refresh symbols                                                   |
//+------------------------------------------------------------------+
void RefreshSymbols()
{
    UpdateStatus("Refreshing...", clrYellow);
    
    // Re-discover all symbols
    DiscoverAllSymbols();
    
    // Update display
    ObjectSetString(0, "FXML_SymbolInfo", OBJPROP_TEXT, "Symbols: 0/" + IntegerToString(totalSymbolsFound));
    UpdateProgressBar(0);
    
    UpdateStatus("Ready", clrWhite);
    Alert("✅ Found " + IntegerToString(totalSymbolsFound) + " symbols");
}

//+------------------------------------------------------------------+
//| Update status                                                     |
//+------------------------------------------------------------------+
void UpdateStatus(string status, color clr)
{
    ObjectSetString(0, "FXML_Status", OBJPROP_TEXT, "Status: " + status);
    ObjectSetInteger(0, "FXML_Status", OBJPROP_COLOR, clr);
}

//+------------------------------------------------------------------+
//| Update counter                                                    |
//+------------------------------------------------------------------+
void UpdateCounter()
{
    ObjectSetString(0, "FXML_Counter", OBJPROP_TEXT, "Sent: " + IntegerToString(totalSentBars) + " bars");
}

//+------------------------------------------------------------------+
//| Update symbol progress                                            |
//+------------------------------------------------------------------+
void UpdateSymbolProgress(int current, int total)
{
    ObjectSetString(0, "FXML_SymbolInfo", OBJPROP_TEXT, 
                    "Symbols: " + IntegerToString(current+1) + "/" + IntegerToString(total));
}

//+------------------------------------------------------------------+
//| Convert timeframe to string                                      |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
{
    switch(tf)
    {
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
//| إنشاء تسمية                                                     |
//+------------------------------------------------------------------+
void CreateLabel(string name, string text, int x, int y, color clr, int size)
{
    ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
}

//+------------------------------------------------------------------+
//| إنشاء زر                                                        |
//+------------------------------------------------------------------+
void CreateButton(string name, string text, int x, int y, int width, int height, color clr)
{
    ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, name, OBJPROP_XSIZE, width);
    ObjectSetInteger(0, name, OBJPROP_YSIZE, height);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
    ObjectSetInteger(0, name, OBJPROP_BGCOLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_BORDER_COLOR, clrGray);
}