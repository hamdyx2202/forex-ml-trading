//+------------------------------------------------------------------+
//|                                       ForexMLBatchDataSender.mq5 |
//|                        إرسال البيانات على دفعات كبيرة وسريعة   |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Forex ML Batch Data Sender"
#property version   "2.00"
#property description "يرسل البيانات التاريخية بشكل سريع عبر دفعات كبيرة"

// إعدادات الخادم
input string   InpServerURL = "http://69.62.121.53:5000";    // عنوان الخادم
input int      InpHistoryDays = 365;                         // أيام البيانات التاريخية
input int      InpBatchSize = 1000;                          // حجم الدفعة (عدد الشموع)
input bool     InpAutoStart = true;                          // البدء التلقائي
input string   InpSymbolsToSend = "EURUSD,GBPUSD,USDJPY,XAUUSD,AUDUSD,USDCAD,EURJPY,GBPJPY"; // الرموز

// الأطر الزمنية
ENUM_TIMEFRAMES AllTimeframes[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};

// متغيرات عامة
string symbolSuffix = "";
int totalSymbolsSent = 0;
int totalBarsSent = 0;
bool isRunning = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // اكتشاف نهاية الرموز
    DetectSymbolSuffix();
    
    // إنشاء واجهة متقدمة
    CreateAdvancedPanel();
    
    Print("========================================");
    Print("🚀 Forex ML Batch Data Sender v2.0");
    Print("========================================");
    Print("Server: ", InpServerURL);
    Print("Batch size: ", InpBatchSize, " bars per request");
    Print("Symbol suffix: '", symbolSuffix, "'");
    
    // اختبار الاتصال
    if(TestConnection())
    {
        UpdateStatus("Connected", clrLime);
        
        if(InpAutoStart)
        {
            StartBatchSending();
        }
    }
    else
    {
        UpdateStatus("Connection Failed", clrRed);
        Alert("❌ Connection failed! Check server.");
    }
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| بدء إرسال الدفعات                                               |
//+------------------------------------------------------------------+
void StartBatchSending()
{
    isRunning = true;
    UpdateStatus("Sending batches...", clrYellow);
    
    // تحويل قائمة الرموز إلى مصفوفة
    string symbols[];
    int symbolCount = StringSplit(InpSymbolsToSend, ',', symbols);
    
    datetime endTime = TimeCurrent();
    datetime startTime = endTime - InpHistoryDays * 24 * 3600;
    
    int totalItems = symbolCount * ArraySize(AllTimeframes);
    int currentItem = 0;
    
    // إرسال كل رمز وإطار زمني
    for(int i = 0; i < symbolCount && isRunning; i++)
    {
        string baseSymbol = symbols[i];
        StringTrimLeft(baseSymbol);
        StringTrimRight(baseSymbol);
        string symbol = baseSymbol + symbolSuffix;
        
        // التحقق من وجود الرمز
        if(!SymbolSelect(symbol, true))
        {
            Print("⚠️ Symbol not found: ", symbol);
            continue;
        }
        
        for(int j = 0; j < ArraySize(AllTimeframes) && isRunning; j++)
        {
            ENUM_TIMEFRAMES tf = AllTimeframes[j];
            currentItem++;
            
            // تحديث التقدم
            double progress = (double)currentItem / totalItems * 100;
            UpdateProgress(progress, symbol + " " + EnumToString(tf));
            
            // جلب كل البيانات
            MqlRates rates[];
            int copied = CopyRates(symbol, tf, startTime, endTime, rates);
            
            if(copied <= 0)
            {
                Print("❌ No data for ", symbol, " ", EnumToString(tf));
                continue;
            }
            
            Print("📊 Sending ", symbol, " ", EnumToString(tf), " - ", copied, " bars");
            
            // إرسال على دفعات كبيرة
            bool success = SendBatchData(symbol, EnumToString(tf), rates);
            
            if(success)
            {
                totalBarsSent += copied;
                UpdateCounters();
            }
            
            // راحة قصيرة بين الرموز
            Sleep(100);
        }
        
        totalSymbolsSent++;
    }
    
    isRunning = false;
    UpdateStatus("Completed!", clrLime);
    UpdateProgress(100, "All done!");
    
    Alert("✅ Batch sending completed!\nTotal bars sent: " + IntegerToString(totalBarsSent));
    
    Print("========================================");
    Print("✅ BATCH SENDING COMPLETED");
    Print("Symbols processed: ", totalSymbolsSent);
    Print("Total bars sent: ", totalBarsSent);
    Print("========================================");
}

//+------------------------------------------------------------------+
//| إرسال دفعة كبيرة من البيانات                                   |
//+------------------------------------------------------------------+
bool SendBatchData(string symbol, string timeframe, MqlRates &rates[])
{
    int totalBars = ArraySize(rates);
    int batches = (totalBars + InpBatchSize - 1) / InpBatchSize;
    bool allSuccess = true;
    
    for(int batch = 0; batch < batches; batch++)
    {
        int start = batch * InpBatchSize;
        int end = MathMin(start + InpBatchSize, totalBars);
        int batchSize = end - start;
        
        // إنشاء JSON للدفعة
        string json = CreateBatchJSON(symbol, timeframe, rates, start, end);
        
        // إرسال الدفعة
        bool sent = SendToServer(json);
        
        if(sent)
        {
            Print("✅ Sent batch ", batch+1, "/", batches, " (", batchSize, " bars)");
        }
        else
        {
            Print("❌ Failed batch ", batch+1, "/", batches);
            allSuccess = false;
        }
        
        // تحديث العداد
        UpdateStatus(StringFormat("Sent %d/%d batches", batch+1, batches), clrYellow);
    }
    
    return allSuccess;
}

//+------------------------------------------------------------------+
//| إنشاء JSON للدفعة الكبيرة                                       |
//+------------------------------------------------------------------+
string CreateBatchJSON(string symbol, string timeframe, MqlRates &rates[], int start, int end)
{
    string json = "{";
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
//| إرسال للخادم                                                    |
//+------------------------------------------------------------------+
bool SendToServer(string jsonData)
{
    string url = InpServerURL + "/api/historical_data";
    string headers = "Content-Type: application/json\r\n";
    char post[], result[];
    
    StringToCharArray(jsonData, post);
    ArrayResize(post, ArraySize(post) - 1);
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, 10000, post, result, headers);
    
    if(res == 200 || res == 201)
    {
        return true;
    }
    else if(res == 404)
    {
        // جرب endpoint آخر
        url = InpServerURL + "/api/live_data";
        res = WebRequest("POST", url, headers, 10000, post, result, headers);
        return (res == 200 || res == 201);
    }
    else
    {
        Print("Server response: ", res, " Error: ", GetLastError());
        return false;
    }
}

//+------------------------------------------------------------------+
//| اكتشاف نهاية الرموز                                             |
//+------------------------------------------------------------------+
void DetectSymbolSuffix()
{
    string testSymbols[] = {"EURUSD", "GBPUSD"};
    string suffixes[] = {"", "m", ".m", "_m", "pro", ".pro"};
    
    for(int t = 0; t < ArraySize(testSymbols); t++)
    {
        for(int s = 0; s < ArraySize(suffixes); s++)
        {
            string checkSymbol = testSymbols[t] + suffixes[s];
            
            for(int i = 0; i < SymbolsTotal(false); i++)
            {
                if(SymbolName(i, false) == checkSymbol)
                {
                    symbolSuffix = suffixes[s];
                    return;
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| اختبار الاتصال                                                  |
//+------------------------------------------------------------------+
bool TestConnection()
{
    string url = InpServerURL + "/health";
    char post[], result[];
    string headers = "";
    
    int res = WebRequest("GET", url, headers, 5000, post, result, headers);
    return (res == 200);
}

//+------------------------------------------------------------------+
//| إنشاء واجهة متقدمة                                              |
//+------------------------------------------------------------------+
void CreateAdvancedPanel()
{
    int x = 10, y = 30;
    
    // العنوان
    CreateLabel("BD_Title", "Batch Data Sender v2", x, y, clrGold, 14);
    
    // الحالة
    y += 30;
    CreateLabel("BD_Status", "Status: Ready", x, y, clrWhite, 11);
    
    // التقدم
    y += 25;
    CreateLabel("BD_Progress", "Progress: 0%", x, y, clrLime, 10);
    
    // العدادات
    y += 20;
    CreateLabel("BD_Symbols", "Symbols: 0", x, y, clrWhite, 10);
    
    y += 20;
    CreateLabel("BD_Bars", "Bars sent: 0", x, y, clrWhite, 10);
    
    // شريط التقدم
    y += 25;
    CreateProgressBar("BD_ProgressBar", x, y, 200, 20);
    
    // الأزرار
    y += 30;
    CreateButton("BD_Start", "Start Sending", x, y, 95, 25, clrGreen);
    CreateButton("BD_Stop", "Stop", x + 105, y, 95, 25, clrRed);
    
    y += 30;
    CreateButton("BD_Test", "Test Connection", x, y, 200, 25, clrBlue);
}

//+------------------------------------------------------------------+
//| Event Handlers                                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    ObjectsDeleteAll(0, "BD_");
}

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
    if(id == CHARTEVENT_OBJECT_CLICK)
    {
        if(sparam == "BD_Start")
        {
            if(!isRunning)
            {
                StartBatchSending();
            }
        }
        else if(sparam == "BD_Stop")
        {
            isRunning = false;
            UpdateStatus("Stopped", clrRed);
        }
        else if(sparam == "BD_Test")
        {
            if(TestConnection())
                Alert("✅ Connection successful!");
            else
                Alert("❌ Connection failed!");
        }
    }
}

//+------------------------------------------------------------------+
//| Helper Functions                                                  |
//+------------------------------------------------------------------+
void UpdateStatus(string status, color clr)
{
    ObjectSetString(0, "BD_Status", OBJPROP_TEXT, "Status: " + status);
    ObjectSetInteger(0, "BD_Status", OBJPROP_COLOR, clr);
}

void UpdateProgress(double percent, string current)
{
    ObjectSetString(0, "BD_Progress", OBJPROP_TEXT, 
                    StringFormat("Progress: %.1f%% - %s", percent, current));
    
    // تحديث شريط التقدم
    int width = (int)(196 * percent / 100);
    ObjectSetInteger(0, "BD_ProgressBar", OBJPROP_XSIZE, width);
}

void UpdateCounters()
{
    ObjectSetString(0, "BD_Symbols", OBJPROP_TEXT, "Symbols: " + IntegerToString(totalSymbolsSent));
    ObjectSetString(0, "BD_Bars", OBJPROP_TEXT, "Bars sent: " + IntegerToString(totalBarsSent));
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

void CreateProgressBar(string name, int x, int y, int width, int height)
{
    // خلفية الشريط
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

void OnTick()
{
    // لا نحتاج شيء هنا
}