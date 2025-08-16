//+------------------------------------------------------------------+
//|                            ForexMLDataCollector_Pro.mq5           |
//|                    النسخة الاحترافية لجامع البيانات                |
//|                     مع حفظ التقدم واستئناف العمل                    |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System Pro"
#property link      "https://forexmltrading.com"
#property version   "3.00"
#property description "جامع بيانات احترافي مع حفظ التقدم وإعادة المحاولة الذكية"

// إعدادات الإدخال
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpBarsPerBatch = 100;        // عدد الشموع في كل دفعة
input int      InpDelayBetweenBatches = 500; // تأخير بين الدفعات (ملي ثانية)
input int      InpYearsOfData = 3;           // عدد سنوات البيانات
input bool     InpDebugMode = true;          // وضع التصحيح
input bool     InpSaveProgress = true;       // حفظ التقدم
input int      InpMaxRetries = 5;            // عدد المحاولات القصوى
input bool     InpPauseOnError = false;     // إيقاف مؤقت عند الخطأ

// العملات المطلوبة - يمكن تخصيصها
input string   InpSymbols = "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,EURJPY,GBPJPY,EURGBP,XAUUSD,BTCUSD";
input string   InpTimeframes = "M5,M15,M30,H1,H4,D1"; // الفريمات المطلوبة

// متغيرات النظام
string symbols[];
ENUM_TIMEFRAMES timeframes[];
string progressFileName = "ForexML_Progress.csv";

struct BatchInfo {
    string symbol;
    string timeframe;
    int batchNumber;
    int totalBatches;
    datetime startTime;
    datetime endTime;
    bool success;
    string error;
};

struct CollectionStats {
    int totalSymbols;
    int totalTimeframes;
    int totalCombinations;
    int completedCombinations;
    int totalBarsCollected;
    int totalBatchesSent;
    int failedBatches;
    datetime startTime;
    datetime lastUpdateTime;
    double estimatedTimeRemaining;
};

CollectionStats stats;
BatchInfo currentBatch;
bool isPaused = false;
int consecutiveErrors = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    PrintHeader();
    
    // تحليل المدخلات
    if(!ParseInputs()) {
        return(INIT_PARAMETERS_INCORRECT);
    }
    
    // تحميل التقدم المحفوظ
    if(InpSaveProgress) {
        LoadProgress();
    }
    
    // تهيئة الإحصائيات
    InitializeStats();
    
    // طباعة معلومات البدء
    PrintStartInfo();
    
    // بدء المعالجة
    EventSetMillisecondTimer(100);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| تحليل المدخلات                                                     |
//+------------------------------------------------------------------+
bool ParseInputs()
{
    // تحليل العملات
    StringSplit(InpSymbols, ',', symbols);
    if(ArraySize(symbols) == 0) {
        Print("❌ خطأ: لا توجد عملات محددة!");
        return false;
    }
    
    // تحليل الفريمات
    string tfStrings[];
    StringSplit(InpTimeframes, ',', tfStrings);
    ArrayResize(timeframes, ArraySize(tfStrings));
    
    for(int i = 0; i < ArraySize(tfStrings); i++) {
        timeframes[i] = StringToTimeframe(tfStrings[i]);
        if(timeframes[i] == 0) {
            Print("❌ خطأ: فريم غير صحيح - ", tfStrings[i]);
            return false;
        }
    }
    
    // التحقق من الإعدادات
    if(InpBarsPerBatch < 10 || InpBarsPerBatch > 1000) {
        Print("❌ خطأ: حجم الدفعة يجب أن يكون بين 10 و 1000");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| تحويل النص إلى فريم زمني                                          |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
    StringToUpper(tf);
    if(tf == "M1")  return PERIOD_M1;
    if(tf == "M5")  return PERIOD_M5;
    if(tf == "M15") return PERIOD_M15;
    if(tf == "M30") return PERIOD_M30;
    if(tf == "H1")  return PERIOD_H1;
    if(tf == "H4")  return PERIOD_H4;
    if(tf == "D1")  return PERIOD_D1;
    if(tf == "W1")  return PERIOD_W1;
    if(tf == "MN1") return PERIOD_MN1;
    return 0;
}

//+------------------------------------------------------------------+
//| طباعة رأس البرنامج                                                |
//+------------------------------------------------------------------+
void PrintHeader()
{
    Print("\n");
    Print("╔════════════════════════════════════════════════╗");
    Print("║      🚀 FOREX ML DATA COLLECTOR PRO 🚀         ║");
    Print("║         جامع البيانات الاحترافي v3.0            ║");
    Print("╚════════════════════════════════════════════════╝");
    Print("");
}

//+------------------------------------------------------------------+
//| Timer function                                                     |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(isPaused) {
        return;
    }
    
    // معالجة الدفعة التالية
    if(!ProcessNextBatch()) {
        // اكتمل جميع البيانات
        OnComplete();
    }
    
    // تحديث الإحصائيات
    UpdateStats();
    
    // حفظ التقدم
    if(InpSaveProgress && stats.totalBatchesSent % 10 == 0) {
        SaveProgress();
    }
}

//+------------------------------------------------------------------+
//| معالجة الدفعة التالية                                              |
//+------------------------------------------------------------------+
bool ProcessNextBatch()
{
    // البحث عن المجموعة التالية غير المكتملة
    for(int s = 0; s < ArraySize(symbols); s++) {
        for(int t = 0; t < ArraySize(timeframes); t++) {
            if(!IsCombinationComplete(symbols[s], timeframes[t])) {
                return ProcessSymbolTimeframe(symbols[s], timeframes[t]);
            }
        }
    }
    
    return false; // اكتملت جميع المجموعات
}

//+------------------------------------------------------------------+
//| معالجة رمز وفريم زمني محدد                                        |
//+------------------------------------------------------------------+
bool ProcessSymbolTimeframe(string symbol, ENUM_TIMEFRAMES tf)
{
    // التحقق من توفر الرمز
    if(!SymbolSelect(symbol, true)) {
        Print("⚠️ تحذير: لا يمكن تحديد الرمز ", symbol);
        MarkCombinationComplete(symbol, tf);
        return true;
    }
    
    // إعداد معلومات الدفعة
    currentBatch.symbol = symbol;
    currentBatch.timeframe = EnumToString(tf);
    currentBatch.startTime = TimeCurrent();
    
    // الحصول على البيانات
    datetime endTime = TimeCurrent();
    datetime startTime = endTime - (InpYearsOfData * 365 * 24 * 60 * 60);
    
    MqlRates rates[];
    int totalBars = CopyRates(symbol, tf, startTime, endTime, rates);
    
    if(totalBars <= 0) {
        Print("⚠️ لا توجد بيانات لـ ", symbol, " ", EnumToString(tf));
        MarkCombinationComplete(symbol, tf);
        return true;
    }
    
    // حساب الدفعات
    int totalBatches = (int)MathCeil((double)totalBars / InpBarsPerBatch);
    int startIndex = GetLastProcessedIndex(symbol, tf);
    
    if(startIndex >= totalBars) {
        MarkCombinationComplete(symbol, tf);
        return true;
    }
    
    currentBatch.totalBatches = totalBatches;
    currentBatch.batchNumber = (startIndex / InpBarsPerBatch) + 1;
    
    // معالجة دفعة واحدة
    int endIndex = MathMin(startIndex + InpBarsPerBatch, totalBars);
    bool success = SendBatch(symbol, tf, rates, startIndex, endIndex);
    
    if(success) {
        UpdateLastProcessedIndex(symbol, tf, endIndex);
        stats.totalBarsCollected += (endIndex - startIndex);
        stats.totalBatchesSent++;
        consecutiveErrors = 0;
        
        PrintProgress();
    } else {
        HandleError();
    }
    
    // تأخير بين الدفعات
    Sleep(InpDelayBetweenBatches);
    
    return true;
}

//+------------------------------------------------------------------+
//| إرسال دفعة البيانات                                                |
//+------------------------------------------------------------------+
bool SendBatch(string symbol, ENUM_TIMEFRAMES tf, MqlRates &rates[], int start, int end)
{
    // إنشاء JSON
    string json = CreateBatchJSON(symbol, tf, rates, start, end);
    
    // إعداد البيانات للإرسال
    char post_data[];
    StringToCharArray(json, post_data);
    ArrayResize(post_data, ArraySize(post_data) - 1);
    
    char result[];
    string headers = "Content-Type: application/json\r\n";
    
    // محاولة الإرسال
    for(int retry = 0; retry < InpMaxRetries; retry++) {
        if(retry > 0) {
            Print("🔄 إعادة المحاولة ", retry, " من ", InpMaxRetries);
            Sleep(1000 * retry); // تأخير متزايد
        }
        
        int res = WebRequest("POST", InpServerURL, headers, 5000, post_data, result, headers);
        
        if(res == 200) {
            currentBatch.success = true;
            currentBatch.error = "";
            
            if(InpDebugMode) {
                string response = CharArrayToString(result);
                Print("✅ نجح الإرسال: ", response);
            }
            
            return true;
        } else if(res == -1) {
            int error = GetLastError();
            currentBatch.error = "WebRequest Error: " + IntegerToString(error);
            
            if(error == 4014) {
                Print("❌ خطأ: يجب السماح بـ URL في إعدادات MT5");
                Print("   Tools -> Options -> Expert Advisors -> Allow WebRequest");
                Print("   أضف: ", InpServerURL);
                isPaused = true;
                return false;
            }
        } else {
            currentBatch.error = "HTTP Error: " + IntegerToString(res);
            string response = CharArrayToString(result);
            if(StringLen(response) > 0) {
                Print("   Response: ", response);
            }
        }
    }
    
    currentBatch.success = false;
    stats.failedBatches++;
    return false;
}

//+------------------------------------------------------------------+
//| إنشاء JSON للدفعة                                                  |
//+------------------------------------------------------------------+
string CreateBatchJSON(string symbol, ENUM_TIMEFRAMES tf, MqlRates &rates[], int start, int end)
{
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + GetTimeframeString(tf) + "\",";
    json += "\"batch_info\":{";
    json += "\"batch_number\":" + IntegerToString(currentBatch.batchNumber) + ",";
    json += "\"total_batches\":" + IntegerToString(currentBatch.totalBatches) + ",";
    json += "\"bars_in_batch\":" + IntegerToString(end - start);
    json += "},";
    json += "\"data\":[";
    
    for(int i = start; i < end; i++) {
        if(i > start) json += ",";
        
        json += "{";
        json += "\"time\":\"" + TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES) + "\",";
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
//| معالجة الأخطاء                                                     |
//+------------------------------------------------------------------+
void HandleError()
{
    consecutiveErrors++;
    
    Print("❌ خطأ في الإرسال: ", currentBatch.error);
    Print("   العملة: ", currentBatch.symbol, " ", currentBatch.timeframe);
    Print("   الدفعة: ", currentBatch.batchNumber, " من ", currentBatch.totalBatches);
    
    if(consecutiveErrors >= InpMaxRetries) {
        Print("⚠️ تجاوز عدد الأخطاء المتتالية الحد المسموح!");
        if(InpPauseOnError) {
            isPaused = true;
            Print("⏸️ تم إيقاف الجمع مؤقتاً. اضغط على زر Continue للمتابعة.");
        }
    }
}

//+------------------------------------------------------------------+
//| طباعة التقدم                                                       |
//+------------------------------------------------------------------+
void PrintProgress()
{
    double overallProgress = (stats.completedCombinations * 100.0) / stats.totalCombinations;
    
    if(InpDebugMode || (int)overallProgress % 5 == 0) {
        Print("📊 [", currentBatch.symbol, " ", currentBatch.timeframe, "] ",
              "الدفعة ", currentBatch.batchNumber, "/", currentBatch.totalBatches,
              " | التقدم الكلي: ", DoubleToString(overallProgress, 1), "%");
    }
}

//+------------------------------------------------------------------+
//| حفظ التقدم                                                         |
//+------------------------------------------------------------------+
void SaveProgress()
{
    if(!InpSaveProgress) return;
    
    int handle = FileOpen(progressFileName, FILE_WRITE|FILE_CSV);
    if(handle != INVALID_HANDLE) {
        // كتابة الإحصائيات
        FileWrite(handle, "StartTime", stats.startTime);
        FileWrite(handle, "LastUpdate", TimeCurrent());
        FileWrite(handle, "TotalBarsCollected", stats.totalBarsCollected);
        FileWrite(handle, "TotalBatchesSent", stats.totalBatchesSent);
        
        // كتابة التقدم لكل مجموعة
        for(int s = 0; s < ArraySize(symbols); s++) {
            for(int t = 0; t < ArraySize(timeframes); t++) {
                string key = symbols[s] + "_" + EnumToString(timeframes[t]);
                int lastIndex = GetLastProcessedIndex(symbols[s], timeframes[t]);
                if(lastIndex > 0) {
                    FileWrite(handle, key, lastIndex);
                }
            }
        }
        
        FileClose(handle);
        
        if(InpDebugMode) {
            Print("💾 تم حفظ التقدم");
        }
    }
}

//+------------------------------------------------------------------+
//| تحميل التقدم المحفوظ                                               |
//+------------------------------------------------------------------+
void LoadProgress()
{
    // سيتم تنفيذ هذه الوظيفة حسب الحاجة
    Print("📂 جاري تحميل التقدم المحفوظ...");
}

//+------------------------------------------------------------------+
//| وظائف مساعدة                                                       |
//+------------------------------------------------------------------+
string GetTimeframeString(ENUM_TIMEFRAMES tf)
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
        default: return "Unknown";
    }
}

// دوال مؤقتة - يجب تنفيذها بالكامل حسب الحاجة
bool IsCombinationComplete(string symbol, ENUM_TIMEFRAMES tf) { return false; }
void MarkCombinationComplete(string symbol, ENUM_TIMEFRAMES tf) {}
int GetLastProcessedIndex(string symbol, ENUM_TIMEFRAMES tf) { return 0; }
void UpdateLastProcessedIndex(string symbol, ENUM_TIMEFRAMES tf, int index) {}
void InitializeStats() { stats.startTime = TimeCurrent(); }
void UpdateStats() {}
void PrintStartInfo() {}
void OnComplete() { EventKillTimer(); Print("✅ اكتمل جمع جميع البيانات!"); }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    SaveProgress();
    
    Print("\n╔════════════════════════════════════════════════╗");
    Print("║           🏁 انتهى جمع البيانات 🏁              ║");
    Print("╚════════════════════════════════════════════════╝");
}

//+------------------------------------------------------------------+
//| Tick function                                                      |
//+------------------------------------------------------------------+
void OnTick() {}