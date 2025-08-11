//+------------------------------------------------------------------+
//|                                            ForexMLDataSync.mq5   |
//|                     نظام مزامنة البيانات مع Linux VPS         |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property version   "2.00"
#property description "يرسل البيانات التاريخية والحية من MT5 إلى خادم Linux"

// إعدادات الخادم
input string   ServerURL = "http://YOUR_VPS_IP:5000";     // عنوان خادم Linux
input string   APIKey = "your_secure_api_key";            // مفتاح API للأمان
input int      UpdateIntervalSeconds = 300;                // فترة التحديث (ثواني)
input int      HistoryDays = 1095;                        // أيام البيانات التاريخية (3 سنوات)
input bool     SendHistoricalData = true;                 // إرسال البيانات التاريخية
input bool     SendLiveData = true;                       // إرسال البيانات الحية
input bool     AutoStart = true;                          // البدء التلقائي

// الأزواج الأساسية - سيتم البحث عن جميع النهايات المحتملة
string BaseSymbols[] = {
    // Major Pairs
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    
    // Cross Pairs
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CHFJPY", "CADJPY",
    "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
    "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "AUDCAD", "AUDNZD", "AUDCHF",
    "NZDCAD", "NZDCHF", "CADCHF",
    
    // Metals
    "XAUUSD", "XAGUSD", "GOLD", "SILVER",
    
    // Additional
    "USDSGD", "USDHKD", "USDMXN", "USDNOK", "USDSEK", "USDZAR", "USDTRY",
    "EURPLN", "EURNOK", "EURSEK", "EURTRY",
    "GBPPLN", "GBPNOK", "GBPSEK"
};

// النهايات المحتملة لأسماء الرموز
string SymbolSuffixes[] = {"", ".", "..", "m", "_m", "pro", ".pro", "ecn", ".ecn", "-5", ".r", "cash", ".cash"};

// قائمة الرموز الفعلية المتاحة
string AvailableSymbols[];

ENUM_TIMEFRAMES Timeframes[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H4, PERIOD_D1};

// متغيرات عامة
datetime lastUpdateTime = 0;
bool isRunning = false;
int totalSentBars = 0;
int failedRequests = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // البحث عن جميع الرموز المتاحة
    FindAvailableSymbols();
    
    // إنشاء أزرار التحكم
    CreateControlPanel();
    
    // البدء التلقائي
    if(AutoStart)
    {
        StartDataSync();
    }
    
    // تعيين Timer للتحديث الدوري
    EventSetTimer(UpdateIntervalSeconds);
    
    Print("✅ ForexML Data Sync EA initialized");
    Print("📡 Server URL: ", ServerURL);
    Print("📊 Available symbols found: ", ArraySize(AvailableSymbols));
    
    // طباعة بعض الرموز المتاحة
    int showCount = MathMin(10, ArraySize(AvailableSymbols));
    for(int i = 0; i < showCount; i++)
    {
        Print("  • ", AvailableSymbols[i]);
    }
    if(ArraySize(AvailableSymbols) > showCount)
    {
        Print("  ... and ", ArraySize(AvailableSymbols) - showCount, " more");
    }
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    ObjectsDeleteAll(0, "FXML_");
    Print("ForexML Data Sync EA stopped. Total bars sent: ", totalSentBars);
}

//+------------------------------------------------------------------+
//| Timer function - تحديث دوري                                      |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(isRunning && SendLiveData)
    {
        SendRecentData();
    }
}

//+------------------------------------------------------------------+
//| إنشاء لوحة التحكم                                                |
//+------------------------------------------------------------------+
void CreateControlPanel()
{
    int x = 10, y = 30;
    
    // العنوان
    CreateLabel("FXML_Title", "ForexML Data Sync", x, y, clrGold, 12);
    
    // حالة الاتصال
    y += 25;
    CreateLabel("FXML_Status", "Status: Ready", x, y, clrWhite, 10);
    
    // عداد البيانات المرسلة
    y += 20;
    CreateLabel("FXML_Counter", "Sent: 0 bars", x, y, clrWhite, 10);
    
    // الأزرار
    y += 30;
    CreateButton("FXML_Start", "Start Sync", x, y, 100, 25, clrGreen);
    
    y += 30;
    CreateButton("FXML_Stop", "Stop Sync", x, y, 100, 25, clrRed);
    
    y += 30;
    CreateButton("FXML_SendHistory", "Send History", x, y, 100, 25, clrBlue);
    
    y += 30;
    CreateButton("FXML_TestConn", "Test Connection", x, y, 100, 25, clrOrange);
}

//+------------------------------------------------------------------+
//| معالج أحداث الرسم البياني                                       |
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
    }
}

//+------------------------------------------------------------------+
//| بدء مزامنة البيانات                                             |
//+------------------------------------------------------------------+
void StartDataSync()
{
    isRunning = true;
    UpdateStatus("Running", clrLime);
    
    if(SendHistoricalData)
    {
        SendAllHistoricalData();
    }
    
    Print("✅ Data sync started");
}

//+------------------------------------------------------------------+
//| إيقاف مزامنة البيانات                                           |
//+------------------------------------------------------------------+
void StopDataSync()
{
    isRunning = false;
    UpdateStatus("Stopped", clrRed);
    Print("⏹ Data sync stopped");
}

//+------------------------------------------------------------------+
//| البحث عن جميع الرموز المتاحة في المنصة                          |
//+------------------------------------------------------------------+
void FindAvailableSymbols()
{
    ArrayResize(AvailableSymbols, 0);
    
    // البحث في جميع الرموز في Market Watch
    int totalSymbols = SymbolsTotal(false);
    
    for(int i = 0; i < totalSymbols; i++)
    {
        string symbolName = SymbolName(i, false);
        
        // التحقق من أن الرمز من نوع Forex أو Metals
        ENUM_SYMBOL_CALC_MODE calcMode = (ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(symbolName, SYMBOL_TRADE_CALC_MODE);
        
        if(calcMode == SYMBOL_CALC_MODE_FOREX || 
           calcMode == SYMBOL_CALC_MODE_CFD ||
           calcMode == SYMBOL_CALC_MODE_CFDINDEX ||
           calcMode == SYMBOL_CALC_MODE_CFDLEVERAGE)
        {
            // التحقق من أن الرمز يطابق أحد الأزواج الأساسية
            for(int j = 0; j < ArraySize(BaseSymbols); j++)
            {
                string baseSymbol = BaseSymbols[j];
                
                // البحث عن التطابق مع أي نهاية محتملة
                for(int k = 0; k < ArraySize(SymbolSuffixes); k++)
                {
                    string checkSymbol = baseSymbol + SymbolSuffixes[k];
                    
                    if(symbolName == checkSymbol || 
                       (StringFind(symbolName, baseSymbol) == 0 && StringLen(symbolName) <= StringLen(baseSymbol) + 5))
                    {
                        // التأكد من أن الرمز قابل للتداول
                        if(SymbolInfoInteger(symbolName, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
                        {
                            // إضافة الرمز إذا لم يكن موجوداً
                            bool exists = false;
                            for(int m = 0; m < ArraySize(AvailableSymbols); m++)
                            {
                                if(AvailableSymbols[m] == symbolName)
                                {
                                    exists = true;
                                    break;
                                }
                            }
                            
                            if(!exists)
                            {
                                int size = ArraySize(AvailableSymbols);
                                ArrayResize(AvailableSymbols, size + 1);
                                AvailableSymbols[size] = symbolName;
                                
                                // تفعيل الرمز في Market Watch
                                SymbolSelect(symbolName, true);
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    
    // البحث أيضاً في جميع الرموز المتاحة (غير المعروضة في Market Watch)
    totalSymbols = SymbolsTotal(true);
    
    for(int i = 0; i < totalSymbols; i++)
    {
        string symbolName = SymbolName(i, true);
        
        // نفس المنطق للبحث
        for(int j = 0; j < ArraySize(BaseSymbols); j++)
        {
            string baseSymbol = BaseSymbols[j];
            
            if(StringFind(symbolName, baseSymbol) == 0)
            {
                // التحقق من أن الرمز ليس موجوداً بالفعل
                bool exists = false;
                for(int m = 0; m < ArraySize(AvailableSymbols); m++)
                {
                    if(AvailableSymbols[m] == symbolName)
                    {
                        exists = true;
                        break;
                    }
                }
                
                if(!exists && SymbolInfoInteger(symbolName, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
                {
                    int size = ArraySize(AvailableSymbols);
                    ArrayResize(AvailableSymbols, size + 1);
                    AvailableSymbols[size] = symbolName;
                    
                    // تفعيل الرمز
                    SymbolSelect(symbolName, true);
                }
                break;
            }
        }
    }
    
    Print("📊 Found ", ArraySize(AvailableSymbols), " available symbols");
}

//+------------------------------------------------------------------+
//| إرسال جميع البيانات التاريخية                                   |
//+------------------------------------------------------------------+
void SendAllHistoricalData()
{
    UpdateStatus("Sending history...", clrYellow);
    
    datetime endTime = TimeCurrent();
    datetime startTime = endTime - HistoryDays * 24 * 3600;
    
    // إرسال بيانات جميع الرموز المتاحة
    for(int i = 0; i < ArraySize(AvailableSymbols); i++)
    {
        string symbol = AvailableSymbols[i];
        
        // التحقق من توفر الرمز
        if(!SymbolSelect(symbol, true))
        {
            Print("⚠️ Cannot select symbol: ", symbol);
            continue;
        }
        
        for(int j = 0; j < ArraySize(Timeframes); j++)
        {
            ENUM_TIMEFRAMES tf = Timeframes[j];
            string timeframe = TimeframeToString(tf);
            
            // جلب البيانات التاريخية
            MqlRates rates[];
            int copied = CopyRates(symbol, tf, startTime, endTime, rates);
            
            if(copied <= 0)
            {
                Print("❌ No data for ", symbol, " ", timeframe);
                continue;
            }
            
            // إرسال البيانات على دفعات
            int batchSize = 1000;  // حجم الدفعة
            int batches = (copied + batchSize - 1) / batchSize;
            
            for(int batch = 0; batch < batches; batch++)
            {
                int start = batch * batchSize;
                int end = MathMin(start + batchSize, copied);
                
                // إنشاء JSON للدفعة
                string jsonData = CreateBatchJSON(symbol, timeframe, rates, start, end);
                
                // إرسال البيانات
                if(SendDataToServer("/api/historical_data", jsonData))
                {
                    totalSentBars += (end - start);
                    UpdateCounter();
                    Print("✅ Sent ", symbol, " ", timeframe, " batch ", batch+1, "/", batches);
                }
                else
                {
                    failedRequests++;
                    Print("❌ Failed to send ", symbol, " ", timeframe, " batch ", batch+1);
                }
                
                // تأخير صغير لتجنب إرهاق الخادم
                Sleep(100);
            }
        }
        
        // عرض التقدم
        double progress = (double)(i + 1) / ArraySize(AvailableSymbols) * 100;
        UpdateStatus(StringFormat("Sending: %.1f%%", progress), clrYellow);
    }
    
    UpdateStatus("History sent", clrLime);
    Print("✅ Historical data sync completed. Total bars: ", totalSentBars);
}

//+------------------------------------------------------------------+
//| إرسال البيانات الحديثة                                          |
//+------------------------------------------------------------------+
void SendRecentData()
{
    UpdateStatus("Updating...", clrYellow);
    
    // إرسال بيانات جميع الرموز المتاحة
    for(int i = 0; i < ArraySize(AvailableSymbols); i++)
    {
        string symbol = AvailableSymbols[i];
        
        for(int j = 0; j < ArraySize(Timeframes); j++)
        {
            ENUM_TIMEFRAMES tf = Timeframes[j];
            string timeframe = TimeframeToString(tf);
            
            // جلب آخر 100 شمعة
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
//| إنشاء JSON للبيانات                                             |
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
//| إرسال البيانات إلى الخادم                                      |
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
        Alert("❌ Connection failed!");
    }
}

//+------------------------------------------------------------------+
//| تحديث حالة الاتصال                                              |
//+------------------------------------------------------------------+
void UpdateStatus(string status, color clr)
{
    ObjectSetString(0, "FXML_Status", OBJPROP_TEXT, "Status: " + status);
    ObjectSetInteger(0, "FXML_Status", OBJPROP_COLOR, clr);
}

//+------------------------------------------------------------------+
//| تحديث عداد البيانات                                             |
//+------------------------------------------------------------------+
void UpdateCounter()
{
    ObjectSetString(0, "FXML_Counter", OBJPROP_TEXT, "Sent: " + IntegerToString(totalSentBars) + " bars");
}

//+------------------------------------------------------------------+
//| تحويل الإطار الزمني إلى نص                                     |
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