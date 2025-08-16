//+------------------------------------------------------------------+
//|                    ForexMLDataCollector_AllTimeframes.mq5         |
//|                    جامع البيانات مع جميع الفريمات الزمنية        |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://forexmltrading.com"
#property version   "5.00"
#property description "جامع بيانات يدعم جميع الفريمات الزمنية المتاحة"

// إعدادات الإدخال
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpBarsPerBatch = 100;        // عدد الشموع في كل دفعة
input int      InpDelayBetweenBatches = 500; // تأخير بين الدفعات (ملي ثانية)
input int      InpYearsOfData = 3;           // عدد سنوات البيانات
input bool     InpDebugMode = true;          // وضع التصحيح

// إعدادات الفريمات المطلوبة
input bool     InpCollect_M1 = false;        // جمع فريم الدقيقة
input bool     InpCollect_M5 = true;         // جمع فريم 5 دقائق
input bool     InpCollect_M15 = true;        // جمع فريم 15 دقيقة
input bool     InpCollect_M30 = true;        // جمع فريم 30 دقيقة
input bool     InpCollect_H1 = true;         // جمع فريم الساعة
input bool     InpCollect_H4 = true;         // جمع فريم 4 ساعات
input bool     InpCollect_D1 = true;         // جمع فريم اليوم
input bool     InpCollect_W1 = false;        // جمع فريم الأسبوع
input bool     InpCollect_MN1 = false;       // جمع فريم الشهر

// إعدادات العملات - يمكن تخصيصها
input string   InpSymbolsList = "AUTO";      // قائمة العملات (AUTO = تلقائي)

// العملات الافتراضية
string defaultSymbols[] = {
    // Forex Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    // Forex Crosses
    "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCAD", "AUDCAD", "AUDNZD",
    // Metals
    "XAUUSD", "XAGUSD",
    // Crypto
    "BTCUSD", "ETHUSD",
    // Energy
    "WTIUSD", "XBRUSD"
};

// متغيرات عامة
ENUM_TIMEFRAMES activeTimeframes[];
string activeSymbols[];
int totalCombinations = 0;
int processedCombinations = 0;
int currentSymbolIndex = 0;
int currentTimeframeIndex = 0;
int currentBatchStart = 0;
bool isProcessing = false;
datetime startTime;

// إحصائيات لكل فريم
struct TimeframeStats {
    ENUM_TIMEFRAMES timeframe;
    string name;
    int symbolsProcessed;
    int totalBars;
    int avgBarsPerSymbol;
    double avgTimePerSymbol;
};

TimeframeStats tfStats[];

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    PrintHeader();
    
    // بناء قائمة الفريمات النشطة
    BuildActiveTimeframes();
    
    if(ArraySize(activeTimeframes) == 0) {
        Print("❌ خطأ: لم يتم اختيار أي فريم زمني!");
        return(INIT_FAILED);
    }
    
    // بناء قائمة العملات
    BuildActiveSymbols();
    
    if(ArraySize(activeSymbols) == 0) {
        Print("❌ خطأ: لا توجد عملات متاحة!");
        return(INIT_FAILED);
    }
    
    // حساب إجمالي المجموعات
    totalCombinations = ArraySize(activeSymbols) * ArraySize(activeTimeframes);
    
    // تهيئة الإحصائيات
    ArrayResize(tfStats, ArraySize(activeTimeframes));
    for(int i = 0; i < ArraySize(activeTimeframes); i++) {
        tfStats[i].timeframe = activeTimeframes[i];
        tfStats[i].name = GetTimeframeString(activeTimeframes[i]);
        tfStats[i].symbolsProcessed = 0;
        tfStats[i].totalBars = 0;
        tfStats[i].avgBarsPerSymbol = 0;
        tfStats[i].avgTimePerSymbol = 0;
    }
    
    // طباعة معلومات البدء
    PrintStartInfo();
    
    startTime = TimeCurrent();
    
    // بدء المعالجة
    EventSetMillisecondTimer(100);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| بناء قائمة الفريمات النشطة                                         |
//+------------------------------------------------------------------+
void BuildActiveTimeframes()
{
    ArrayResize(activeTimeframes, 0);
    
    // إضافة الفريمات حسب الاختيار
    if(InpCollect_M1)  AddTimeframe(PERIOD_M1);
    if(InpCollect_M5)  AddTimeframe(PERIOD_M5);
    if(InpCollect_M15) AddTimeframe(PERIOD_M15);
    if(InpCollect_M30) AddTimeframe(PERIOD_M30);
    if(InpCollect_H1)  AddTimeframe(PERIOD_H1);
    if(InpCollect_H4)  AddTimeframe(PERIOD_H4);
    if(InpCollect_D1)  AddTimeframe(PERIOD_D1);
    if(InpCollect_W1)  AddTimeframe(PERIOD_W1);
    if(InpCollect_MN1) AddTimeframe(PERIOD_MN1);
}

//+------------------------------------------------------------------+
//| إضافة فريم زمني                                                   |
//+------------------------------------------------------------------+
void AddTimeframe(ENUM_TIMEFRAMES tf)
{
    ArrayResize(activeTimeframes, ArraySize(activeTimeframes) + 1);
    activeTimeframes[ArraySize(activeTimeframes) - 1] = tf;
}

//+------------------------------------------------------------------+
//| بناء قائمة العملات النشطة                                          |
//+------------------------------------------------------------------+
void BuildActiveSymbols()
{
    if(InpSymbolsList == "AUTO") {
        // استخدام القائمة الافتراضية
        ArrayResize(activeSymbols, ArraySize(defaultSymbols));
        ArrayCopy(activeSymbols, defaultSymbols);
        
        // التحقق من توفر العملات
        CheckSymbolsAvailability();
    } else {
        // تحليل قائمة مخصصة
        StringSplit(InpSymbolsList, ',', activeSymbols);
    }
}

//+------------------------------------------------------------------+
//| التحقق من توفر العملات                                            |
//+------------------------------------------------------------------+
void CheckSymbolsAvailability()
{
    string availableSymbols[];
    ArrayResize(availableSymbols, 0);
    
    for(int i = 0; i < ArraySize(activeSymbols); i++) {
        string symbol = activeSymbols[i];
        
        // محاولة أشكال مختلفة للرمز
        if(SymbolSelect(symbol, true)) {
            ArrayResize(availableSymbols, ArraySize(availableSymbols) + 1);
            availableSymbols[ArraySize(availableSymbols) - 1] = symbol;
        } else if(SymbolSelect(symbol + "m", true)) {
            ArrayResize(availableSymbols, ArraySize(availableSymbols) + 1);
            availableSymbols[ArraySize(availableSymbols) - 1] = symbol + "m";
        } else if(SymbolSelect(symbol + ".a", true)) {
            ArrayResize(availableSymbols, ArraySize(availableSymbols) + 1);
            availableSymbols[ArraySize(availableSymbols) - 1] = symbol + ".a";
        }
    }
    
    // تحديث القائمة
    ArrayResize(activeSymbols, ArraySize(availableSymbols));
    ArrayCopy(activeSymbols, availableSymbols);
}

//+------------------------------------------------------------------+
//| طباعة رأس البرنامج                                                |
//+------------------------------------------------------------------+
void PrintHeader()
{
    Print("\n");
    Print("╔════════════════════════════════════════════════════════╗");
    Print("║      📊 FOREX ML DATA COLLECTOR - ALL TIMEFRAMES 📊    ║");
    Print("║           جامع البيانات مع جميع الفريمات الزمنية        ║");
    Print("╚════════════════════════════════════════════════════════╝");
    Print("");
}

//+------------------------------------------------------------------+
//| طباعة معلومات البدء                                               |
//+------------------------------------------------------------------+
void PrintStartInfo()
{
    Print("📊 معلومات الجمع:");
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("📈 عدد العملات: ", ArraySize(activeSymbols));
    Print("⏰ عدد الفريمات: ", ArraySize(activeTimeframes));
    Print("🔢 إجمالي المجموعات: ", totalCombinations);
    Print("📅 سنوات البيانات: ", InpYearsOfData);
    Print("📦 حجم الدفعة: ", InpBarsPerBatch, " شمعة");
    
    // طباعة الفريمات المختارة
    Print("\n⏰ الفريمات المختارة:");
    for(int i = 0; i < ArraySize(activeTimeframes); i++) {
        string tfName = GetTimeframeString(activeTimeframes[i]);
        string tfDesc = GetTimeframeDescription(activeTimeframes[i]);
        Print("  • ", tfName, " - ", tfDesc);
    }
    
    // طباعة عينة من العملات
    Print("\n📋 عينة من العملات:");
    int sampleSize = MathMin(10, ArraySize(activeSymbols));
    for(int i = 0; i < sampleSize; i++) {
        Print("  • ", activeSymbols[i]);
    }
    if(ArraySize(activeSymbols) > sampleSize) {
        Print("  • ... و ", ArraySize(activeSymbols) - sampleSize, " عملة أخرى");
    }
    
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // تقدير الوقت
    EstimateCompletionTime();
}

//+------------------------------------------------------------------+
//| تقدير وقت الإكمال                                                 |
//+------------------------------------------------------------------+
void EstimateCompletionTime()
{
    // تقدير تقريبي بناءً على الخبرة
    double avgBarsPerCombination = 50000; // متوسط تقديري
    double avgSecondsPerBatch = 1.5; // ثانية ونصف لكل دفعة
    
    double totalBatches = (totalCombinations * avgBarsPerCombination) / InpBarsPerBatch;
    double estimatedSeconds = totalBatches * avgSecondsPerBatch;
    
    int hours = (int)(estimatedSeconds / 3600);
    int minutes = (int)((estimatedSeconds - hours * 3600) / 60);
    
    Print("\n⏱️ الوقت المقدر: ", hours, " ساعة و ", minutes, " دقيقة");
    Print("💡 نصيحة: يمكنك ترك البرنامج يعمل ليلاً");
}

//+------------------------------------------------------------------+
//| Timer function                                                     |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!isProcessing && currentSymbolIndex < ArraySize(activeSymbols)) {
        ProcessNextBatch();
    }
}

//+------------------------------------------------------------------+
//| معالجة الدفعة التالية                                              |
//+------------------------------------------------------------------+
void ProcessNextBatch()
{
    if(currentSymbolIndex >= ArraySize(activeSymbols)) {
        OnComplete();
        return;
    }
    
    isProcessing = true;
    
    string symbol = activeSymbols[currentSymbolIndex];
    ENUM_TIMEFRAMES tf = activeTimeframes[currentTimeframeIndex];
    datetime tfStartTime = TimeCurrent();
    
    // الحصول على البيانات
    datetime endTime = TimeCurrent();
    datetime startTimeData = endTime - (InpYearsOfData * 365 * 24 * 60 * 60);
    
    MqlRates rates[];
    int totalAvailable = CopyRates(symbol, tf, startTimeData, endTime, rates);
    
    if(totalAvailable <= 0) {
        if(InpDebugMode) {
            Print("⚠️ لا توجد بيانات: ", symbol, " ", GetTimeframeString(tf));
        }
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // طباعة بداية معالجة جديدة
    if(currentBatchStart == 0) {
        processedCombinations++;
        double overallProgress = (processedCombinations * 100.0) / totalCombinations;
        
        Print("\n📊 [", processedCombinations, "/", totalCombinations, "] ",
              symbol, " ", GetTimeframeString(tf), 
              " - ", DoubleToString(overallProgress, 1), "% إجمالي");
        Print("   📈 الشموع: ", totalAvailable);
    }
    
    // حساب نطاق الدفعة
    int batchEnd = MathMin(currentBatchStart + InpBarsPerBatch, totalAvailable);
    int batchSize = batchEnd - currentBatchStart;
    
    if(batchSize <= 0) {
        // تحديث إحصائيات الفريم
        int tfIndex = GetTimeframeIndex(tf);
        if(tfIndex >= 0) {
            tfStats[tfIndex].symbolsProcessed++;
            tfStats[tfIndex].totalBars += totalAvailable;
            tfStats[tfIndex].avgBarsPerSymbol = tfStats[tfIndex].totalBars / tfStats[tfIndex].symbolsProcessed;
            
            double timeSpent = (double)(TimeCurrent() - tfStartTime);
            tfStats[tfIndex].avgTimePerSymbol = 
                (tfStats[tfIndex].avgTimePerSymbol * (tfStats[tfIndex].symbolsProcessed - 1) + timeSpent) 
                / tfStats[tfIndex].symbolsProcessed;
        }
        
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // إنشاء وإرسال البيانات
    string jsonData = CreateBatchJSON(symbol, tf, rates, currentBatchStart, batchEnd);
    bool success = SendBatchData(jsonData);
    
    if(success) {
        currentBatchStart = batchEnd;
        
        // طباعة التقدم
        double progress = (currentBatchStart * 100.0) / totalAvailable;
        if(InpDebugMode || (int)progress % 25 == 0) {
            Print("   ", CreateMiniProgressBar(progress), " ", 
                  DoubleToString(progress, 1), "%");
        }
    } else {
        Print("❌ فشل الإرسال - تخطي");
        MoveToNextCombination();
    }
    
    Sleep(InpDelayBetweenBatches);
    isProcessing = false;
}

//+------------------------------------------------------------------+
//| إنشاء شريط تقدم صغير                                              |
//+------------------------------------------------------------------+
string CreateMiniProgressBar(double percentage)
{
    int filled = (int)(percentage / 10); // 10 خانات
    string bar = "";
    
    for(int i = 0; i < 10; i++) {
        if(i < filled) {
            bar += "▰";
        } else {
            bar += "▱";
        }
    }
    
    return bar;
}

//+------------------------------------------------------------------+
//| الحصول على مؤشر الفريم                                            |
//+------------------------------------------------------------------+
int GetTimeframeIndex(ENUM_TIMEFRAMES tf)
{
    for(int i = 0; i < ArraySize(tfStats); i++) {
        if(tfStats[i].timeframe == tf) {
            return i;
        }
    }
    return -1;
}

//+------------------------------------------------------------------+
//| الانتقال للمجموعة التالية                                          |
//+------------------------------------------------------------------+
void MoveToNextCombination()
{
    currentBatchStart = 0;
    currentTimeframeIndex++;
    
    if(currentTimeframeIndex >= ArraySize(activeTimeframes)) {
        currentTimeframeIndex = 0;
        currentSymbolIndex++;
        
        // طباعة ملخص بعد كل عملة
        if(currentSymbolIndex < ArraySize(activeSymbols)) {
            PrintSymbolSummary();
        }
    }
}

//+------------------------------------------------------------------+
//| طباعة ملخص العملة                                                  |
//+------------------------------------------------------------------+
void PrintSymbolSummary()
{
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("✅ اكتمل: ", activeSymbols[currentSymbolIndex - 1]);
    
    // حساب الوقت المتبقي
    double avgTimePerCombination = (double)(TimeCurrent() - startTime) / processedCombinations;
    int remainingCombinations = totalCombinations - processedCombinations;
    int remainingSeconds = (int)(avgTimePerCombination * remainingCombinations);
    
    int hours = remainingSeconds / 3600;
    int minutes = (remainingSeconds % 3600) / 60;
    
    Print("⏱️ الوقت المتبقي المقدر: ", hours, ":", 
          StringFormat("%02d", minutes));
}

//+------------------------------------------------------------------+
//| إنشاء JSON للدفعة                                                  |
//+------------------------------------------------------------------+
string CreateBatchJSON(string symbol, ENUM_TIMEFRAMES tf, MqlRates &rates[], int start, int end)
{
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + GetTimeframeString(tf) + "\",";
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
//| إرسال دفعة البيانات                                                |
//+------------------------------------------------------------------+
bool SendBatchData(string jsonData)
{
    char post_data[];
    StringToCharArray(jsonData, post_data);
    ArrayResize(post_data, ArraySize(post_data) - 1);
    
    char result[];
    string headers = "Content-Type: application/json\r\n";
    
    int res = WebRequest("POST", InpServerURL, headers, 5000, post_data, result, headers);
    
    return (res == 200);
}

//+------------------------------------------------------------------+
//| تحويل الفريم لنص                                                   |
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

//+------------------------------------------------------------------+
//| الحصول على وصف الفريم                                             |
//+------------------------------------------------------------------+
string GetTimeframeDescription(ENUM_TIMEFRAMES tf)
{
    switch(tf) {
        case PERIOD_M1:  return "دقيقة واحدة";
        case PERIOD_M5:  return "5 دقائق";
        case PERIOD_M15: return "15 دقيقة";
        case PERIOD_M30: return "30 دقيقة";
        case PERIOD_H1:  return "ساعة واحدة";
        case PERIOD_H4:  return "4 ساعات";
        case PERIOD_D1:  return "يوم واحد";
        case PERIOD_W1:  return "أسبوع واحد";
        case PERIOD_MN1: return "شهر واحد";
        default: return "غير معروف";
    }
}

//+------------------------------------------------------------------+
//| عند اكتمال الجمع                                                   |
//+------------------------------------------------------------------+
void OnComplete()
{
    EventKillTimer();
    
    datetime totalTime = TimeCurrent() - startTime;
    
    Print("\n");
    Print("╔════════════════════════════════════════════════════════╗");
    Print("║            📊 التقرير النهائي - الفريمات 📊            ║");
    Print("╚════════════════════════════════════════════════════════╝");
    
    // إحصائيات لكل فريم
    Print("\n📊 إحصائيات الفريمات:");
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    int totalBarsAllTimeframes = 0;
    for(int i = 0; i < ArraySize(tfStats); i++) {
        if(tfStats[i].symbolsProcessed > 0) {
            Print("⏰ ", tfStats[i].name, ":");
            Print("   • عملات معالجة: ", tfStats[i].symbolsProcessed);
            Print("   • إجمالي الشموع: ", FormatNumber(tfStats[i].totalBars));
            Print("   • متوسط شموع/عملة: ", tfStats[i].avgBarsPerSymbol);
            Print("   • متوسط وقت/عملة: ", 
                  DoubleToString(tfStats[i].avgTimePerSymbol, 1), " ثانية");
            
            totalBarsAllTimeframes += tfStats[i].totalBars;
        }
    }
    
    Print("\n📈 الإحصائيات الإجمالية:");
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("✅ المجموعات المعالجة: ", processedCombinations, " من ", totalCombinations);
    Print("📊 إجمالي الشموع: ", FormatNumber(totalBarsAllTimeframes));
    Print("⏱️ الوقت الإجمالي: ", TimeToString(totalTime, TIME_MINUTES|TIME_SECONDS));
    
    if(totalTime > 0) {
        double barsPerMinute = (totalBarsAllTimeframes * 60.0) / totalTime;
        Print("⚡ معدل الجمع: ", DoubleToString(barsPerMinute, 0), " شمعة/دقيقة");
    }
    
    Print("\n🎉 اكتمل جمع البيانات بنجاح!");
}

//+------------------------------------------------------------------+
//| تنسيق الأرقام                                                      |
//+------------------------------------------------------------------+
string FormatNumber(int number)
{
    if(number >= 1000000) {
        return DoubleToString(number/1000000.0, 1) + "M";
    } else if(number >= 1000) {
        return DoubleToString(number/1000.0, 1) + "K";
    }
    return IntegerToString(number);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("\n🏁 توقف جامع البيانات");
}

//+------------------------------------------------------------------+
//| Tick function                                                      |
//+------------------------------------------------------------------+
void OnTick() {}