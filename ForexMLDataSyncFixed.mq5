//+------------------------------------------------------------------+
//|                                       ForexMLDataSyncFixed.mq5   |
//|                   نسخة محسنة - تحل مشاكل الاتصال والبيانات    |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System Fixed"
#property version   "4.00"
#property description "نسخة محسنة مع معالجة أفضل للأخطاء"

// إعدادات الخادم
input string   ServerURL = "http://YOUR_VPS_IP:5000";     // عنوان خادم Linux
input string   APIKey = "your_secure_api_key";            // مفتاح API للأمان
input int      UpdateIntervalSeconds = 300;                // فترة التحديث (ثواني)
input int      HistoryDays = 365;                         // أيام البيانات التاريخية (سنة افتراضياً)
input bool     SendHistoricalData = true;                 // إرسال البيانات التاريخية
input bool     SendLiveData = true;                       // إرسال البيانات الحية
input bool     AutoStart = false;                         // البدء التلقائي

// إعدادات الأزواج
input string   SymbolsToSync = "EURUSD,GBPUSD,USDJPY,XAUUSD,AUDUSD,USDCAD";  // الأزواج للمزامنة (فاصلة)
input bool     AutoDetectSuffix = true;                   // اكتشاف النهاية تلقائياً
input string   ManualSuffix = "";                         // النهاية اليدوية (اتركها فارغة)

// إعدادات متقدمة
input int      BatchSize = 500;                           // حجم دفعة الإرسال (قلل للاتصال البطيء)
input int      RequestTimeout = 10000;                     // مهلة الطلب بالملي ثانية
input int      MaxRetries = 3;                            // عدد المحاولات عند الفشل
input bool     SkipMissingData = true;                    // تجاهل الأزواج بدون بيانات
input bool     VerboseLogging = false;                    // سجلات تفصيلية

// الرموز النشطة
string ActiveSymbols[];
int totalActiveSymbols = 0;
string detectedSuffix = "";

// الأطر الزمنية المحددة
ENUM_TIMEFRAMES ActiveTimeframes[] = {PERIOD_H1, PERIOD_H4, PERIOD_D1};  // أطر زمنية أقل لتسريع النقل

// متغيرات عامة
datetime lastUpdateTime = 0;
bool isRunning = false;
int totalSentBars = 0;
int failedRequests = 0;
int successfulRequests = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("🚀 ForexML Data Sync Fixed v4.0");
    Print("========================================");
    
    // اكتشاف النهاية
    if(AutoDetectSuffix && ManualSuffix == "")
    {
        detectedSuffix = DetectSymbolSuffix();
    }
    else if(ManualSuffix != "")
    {
        detectedSuffix = ManualSuffix;
    }
    
    // تحضير قائمة الرموز
    PrepareSymbolsList();
    
    // إنشاء لوحة التحكم
    CreateControlPanel();
    
    // اختبار الاتصال أولاً
    if(AutoStart)
    {
        Print("⏳ Testing connection first...");
        if(TestConnectionQuiet())
        {
            StartDataSync();
        }
        else
        {
            Print("❌ Connection test failed. Please check:");
            Print("   1. Server URL: ", ServerURL);
            Print("   2. Server is running on Linux");
            Print("   3. WebRequest is allowed in MT5");
            UpdateStatus("Connection Failed", clrRed);
        }
    }
    
    // تعيين Timer
    EventSetTimer(UpdateIntervalSeconds);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| اكتشاف النهاية المستخدمة                                        |
//+------------------------------------------------------------------+
string DetectSymbolSuffix()
{
    // البحث عن EURUSD مع نهايات مختلفة
    string testPairs[] = {"EURUSD", "GBPUSD", "USDJPY"};
    string suffixes[] = {"", "m", ".m", "_m", "pro", ".pro", "ecn", ".ecn", "-5", ".r"};
    
    for(int p = 0; p < ArraySize(testPairs); p++)
    {
        for(int s = 0; s < ArraySize(suffixes); s++)
        {
            string checkSymbol = testPairs[p] + suffixes[s];
            
            // التحقق في Market Watch
            for(int i = 0; i < SymbolsTotal(false); i++)
            {
                if(SymbolName(i, false) == checkSymbol)
                {
                    Print("✅ Detected suffix: '", suffixes[s], "' (found ", checkSymbol, ")");
                    return suffixes[s];
                }
            }
            
            // التحقق في جميع الرموز
            for(int i = 0; i < SymbolsTotal(true); i++)
            {
                if(SymbolName(i, true) == checkSymbol)
                {
                    SymbolSelect(checkSymbol, true);
                    Print("✅ Detected suffix: '", suffixes[s], "' (found ", checkSymbol, ")");
                    return suffixes[s];
                }
            }
        }
    }
    
    Print("⚠️ No suffix detected, using default");
    return "";
}

//+------------------------------------------------------------------+
//| تحضير قائمة الرموز النشطة                                       |
//+------------------------------------------------------------------+
void PrepareSymbolsList()
{
    ArrayResize(ActiveSymbols, 0);
    totalActiveSymbols = 0;
    
    // تقسيم الرموز المدخلة
    string symbols[];
    int count = StringSplit(SymbolsToSync, ',', symbols);
    
    Print("📋 Preparing symbols list...");
    
    for(int i = 0; i < count; i++)
    {
        string baseSymbol = symbols[i];
        StringTrimLeft(baseSymbol);
        StringTrimRight(baseSymbol);
        
        // إضافة النهاية
        string fullSymbol = baseSymbol + detectedSuffix;
        
        // التحقق من وجود الرمز
        bool symbolExists = false;
        
        // البحث في Market Watch أولاً
        for(int j = 0; j < SymbolsTotal(false); j++)
        {
            if(SymbolName(j, false) == fullSymbol)
            {
                symbolExists = true;
                break;
            }
        }
        
        // البحث في جميع الرموز
        if(!symbolExists)
        {
            for(int j = 0; j < SymbolsTotal(true); j++)
            {
                if(SymbolName(j, true) == fullSymbol)
                {
                    symbolExists = true;
                    SymbolSelect(fullSymbol, true);
                    break;
                }
            }
        }
        
        if(symbolExists)
        {
            // التحقق من أن الرمز يحتوي على بيانات
            MqlRates testRates[];
            int testCopied = CopyRates(fullSymbol, PERIOD_H1, 0, 1, testRates);
            
            if(testCopied > 0)
            {
                ArrayResize(ActiveSymbols, totalActiveSymbols + 1);
                ActiveSymbols[totalActiveSymbols] = fullSymbol;
                totalActiveSymbols++;
                Print("✅ Added: ", fullSymbol);
            }
            else if(!SkipMissingData)
            {
                Print("⚠️ No data for: ", fullSymbol);
            }
        }
        else
        {
            Print("❌ Not found: ", fullSymbol);
            
            // محاولة بدون نهاية
            if(detectedSuffix != "")
            {
                for(int j = 0; j < SymbolsTotal(true); j++)
                {
                    if(SymbolName(j, true) == baseSymbol)
                    {
                        SymbolSelect(baseSymbol, true);
                        ArrayResize(ActiveSymbols, totalActiveSymbols + 1);
                        ActiveSymbols[totalActiveSymbols] = baseSymbol;
                        totalActiveSymbols++;
                        Print("✅ Added without suffix: ", baseSymbol);
                        break;
                    }
                }
            }
        }
    }
    
    Print("📊 Total active symbols: ", totalActiveSymbols);
}

//+------------------------------------------------------------------+
//| إنشاء لوحة التحكم البسيطة                                       |
//+------------------------------------------------------------------+
void CreateControlPanel()
{
    int x = 10, y = 30;
    
    // العنوان
    CreateLabel("FX_Title", "ForexML Sync Fixed", x, y, clrGold, 12);
    
    // معلومات الرموز
    y += 25;
    CreateLabel("FX_Symbols", "Symbols: " + IntegerToString(totalActiveSymbols), x, y, clrWhite, 10);
    
    // الحالة
    y += 20;
    CreateLabel("FX_Status", "Status: Ready", x, y, clrWhite, 10);
    
    // العدادات
    y += 20;
    CreateLabel("FX_Success", "Success: 0", x, y, clrLime, 10);
    
    y += 20;
    CreateLabel("FX_Failed", "Failed: 0", x, y, clrRed, 10);
    
    // الأزرار
    y += 30;
    CreateButton("FX_Start", "Start", x, y, 80, 25, clrGreen);
    CreateButton("FX_Stop", "Stop", x + 90, y, 80, 25, clrRed);
    
    y += 30;
    CreateButton("FX_Test", "Test Connection", x, y, 170, 25, clrBlue);
}

//+------------------------------------------------------------------+
//| معالج الأحداث                                                   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
    if(id == CHARTEVENT_OBJECT_CLICK)
    {
        if(sparam == "FX_Start")
        {
            StartDataSync();
        }
        else if(sparam == "FX_Stop")
        {
            StopDataSync();
        }
        else if(sparam == "FX_Test")
        {
            TestConnection();
        }
    }
}

//+------------------------------------------------------------------+
//| بدء المزامنة                                                     |
//+------------------------------------------------------------------+
void StartDataSync()
{
    if(totalActiveSymbols == 0)
    {
        Alert("No active symbols found!");
        return;
    }
    
    isRunning = true;
    UpdateStatus("Starting...", clrYellow);
    
    // اختبار الاتصال أولاً
    if(!TestConnectionQuiet())
    {
        Alert("Connection failed! Check server.");
        isRunning = false;
        UpdateStatus("Failed", clrRed);
        return;
    }
    
    UpdateStatus("Running", clrLime);
    
    if(SendHistoricalData)
    {
        SendAllHistoricalData();
    }
}

//+------------------------------------------------------------------+
//| إيقاف المزامنة                                                   |
//+------------------------------------------------------------------+
void StopDataSync()
{
    isRunning = false;
    UpdateStatus("Stopped", clrRed);
}

//+------------------------------------------------------------------+
//| إرسال البيانات التاريخية                                        |
//+------------------------------------------------------------------+
void SendAllHistoricalData()
{
    UpdateStatus("Sending...", clrYellow);
    
    datetime endTime = TimeCurrent();
    datetime startTime = endTime - HistoryDays * 24 * 3600;
    
    int totalBatches = totalActiveSymbols * ArraySize(ActiveTimeframes);
    int currentBatch = 0;
    
    for(int i = 0; i < totalActiveSymbols; i++)
    {
        string symbol = ActiveSymbols[i];
        
        for(int j = 0; j < ArraySize(ActiveTimeframes); j++)
        {
            ENUM_TIMEFRAMES tf = ActiveTimeframes[j];
            string timeframe = TimeframeToString(tf);
            
            currentBatch++;
            
            // تحديث الحالة
            string progress = StringFormat("Sending %s %s (%d/%d)", 
                                         symbol, timeframe, currentBatch, totalBatches);
            UpdateStatus(progress, clrYellow);
            
            // جلب البيانات
            MqlRates rates[];
            ResetLastError();
            int copied = CopyRates(symbol, tf, startTime, endTime, rates);
            
            if(copied <= 0)
            {
                if(VerboseLogging)
                {
                    Print("⚠️ No data for ", symbol, " ", timeframe, " Error: ", GetLastError());
                }
                continue;
            }
            
            // إرسال على دفعات
            int batches = (copied + BatchSize - 1) / BatchSize;
            bool batchSuccess = true;
            
            for(int batch = 0; batch < batches && batchSuccess; batch++)
            {
                int start = batch * BatchSize;
                int end = MathMin(start + BatchSize, copied);
                
                // إنشاء JSON
                string jsonData = CreateBatchJSON(symbol, timeframe, rates, start, end);
                
                // إرسال مع إعادة المحاولة
                bool sent = false;
                for(int retry = 0; retry < MaxRetries && !sent; retry++)
                {
                    if(SendDataToServer("/api/historical_data", jsonData))
                    {
                        totalSentBars += (end - start);
                        successfulRequests++;
                        UpdateCounters();
                        sent = true;
                        
                        if(VerboseLogging)
                        {
                            Print("✅ Sent ", symbol, " ", timeframe, 
                                  " batch ", batch+1, "/", batches);
                        }
                    }
                    else
                    {
                        if(retry < MaxRetries - 1)
                        {
                            Sleep(1000); // انتظار ثانية قبل إعادة المحاولة
                        }
                    }
                }
                
                if(!sent)
                {
                    failedRequests++;
                    UpdateCounters();
                    batchSuccess = false;
                    
                    if(VerboseLogging)
                    {
                        Print("❌ Failed ", symbol, " ", timeframe, 
                              " batch ", batch+1, " after ", MaxRetries, " retries");
                    }
                }
                
                // تأخير بين الدفعات
                Sleep(100);
            }
        }
    }
    
    UpdateStatus("Completed", clrLime);
    Print("✅ Sync completed. Sent: ", totalSentBars, " bars");
    Print("   Success: ", successfulRequests, " Failed: ", failedRequests);
}

//+------------------------------------------------------------------+
//| إرسال البيانات الحديثة                                          |
//+------------------------------------------------------------------+
void SendRecentData()
{
    if(!isRunning || totalActiveSymbols == 0) return;
    
    UpdateStatus("Updating...", clrYellow);
    
    for(int i = 0; i < totalActiveSymbols; i++)
    {
        string symbol = ActiveSymbols[i];
        
        for(int j = 0; j < ArraySize(ActiveTimeframes); j++)
        {
            ENUM_TIMEFRAMES tf = ActiveTimeframes[j];
            string timeframe = TimeframeToString(tf);
            
            MqlRates rates[];
            int copied = CopyRates(symbol, tf, 0, 100, rates);
            
            if(copied > 0)
            {
                string jsonData = CreateBatchJSON(symbol, timeframe, rates, 0, copied);
                
                if(SendDataToServer("/api/live_data", jsonData))
                {
                    totalSentBars += copied;
                    successfulRequests++;
                }
                else
                {
                    failedRequests++;
                }
                UpdateCounters();
            }
        }
    }
    
    UpdateStatus("Running", clrLime);
}

//+------------------------------------------------------------------+
//| إنشاء JSON للدفعة                                                |
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
//| إرسال البيانات للخادم                                           |
//+------------------------------------------------------------------+
bool SendDataToServer(string endpoint, string jsonData)
{
    string headers = "Content-Type: application/json\r\n";
    char post[], result[];
    
    StringToCharArray(jsonData, post);
    
    string url = ServerURL + endpoint;
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, RequestTimeout, post, result, headers);
    
    if(res == 200 || res == 201)
    {
        return true;
    }
    else
    {
        int error = GetLastError();
        
        if(error == 4060)
        {
            Print("❌ URL not allowed. Add to MT5:");
            Print("   Tools → Options → Expert Advisors → Allow WebRequest");
            Print("   Add URL: ", ServerURL);
            Alert("Please allow WebRequest for: " + ServerURL);
        }
        else if(error == 5203)
        {
            if(VerboseLogging)
            {
                Print("⚠️ Server timeout. Check if server is running.");
            }
        }
        else
        {
            if(VerboseLogging)
            {
                Print("❌ HTTP: ", res, ", Error: ", error);
            }
        }
        
        return false;
    }
}

//+------------------------------------------------------------------+
//| اختبار الاتصال                                                  |
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
        Alert("❌ Connection failed! Check server and WebRequest settings.");
    }
}

//+------------------------------------------------------------------+
//| اختبار اتصال صامت                                              |
//+------------------------------------------------------------------+
bool TestConnectionQuiet()
{
    string json = "{\"api_key\":\"" + APIKey + "\",\"test\":true}";
    return SendDataToServer("/api/test", json);
}

//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(isRunning && SendLiveData)
    {
        SendRecentData();
    }
}

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    ObjectsDeleteAll(0, "FX_");
}

//+------------------------------------------------------------------+
//| Helper Functions                                                  |
//+------------------------------------------------------------------+

void UpdateStatus(string status, color clr)
{
    ObjectSetString(0, "FX_Status", OBJPROP_TEXT, "Status: " + status);
    ObjectSetInteger(0, "FX_Status", OBJPROP_COLOR, clr);
}

void UpdateCounters()
{
    ObjectSetString(0, "FX_Success", OBJPROP_TEXT, "Success: " + IntegerToString(successfulRequests));
    ObjectSetString(0, "FX_Failed", OBJPROP_TEXT, "Failed: " + IntegerToString(failedRequests));
}

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