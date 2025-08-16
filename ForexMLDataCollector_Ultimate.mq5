//+------------------------------------------------------------------+
//|                        ForexMLDataCollector_Ultimate.mq5          |
//|                    جامع البيانات الشامل - النسخة النهائية         |
//|                         يدعم جميع الأسواق المتاحة                  |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System Ultimate"
#property link      "https://forexmltrading.com"
#property version   "4.00"
#property description "جامع بيانات شامل مع تصنيف ذكي وإدارة متقدمة"

// إعدادات الإدخال
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpBarsPerBatch = 100;        // عدد الشموع في كل دفعة
input int      InpDelayBetweenBatches = 500; // تأخير بين الدفعات (ملي ثانية)
input int      InpYearsOfData = 3;           // عدد سنوات البيانات
input bool     InpDebugMode = true;          // وضع التصحيح

// إعدادات الأسواق المطلوبة
input bool     InpCollectForexMajors = true;    // جمع العملات الرئيسية
input bool     InpCollectForexMinors = true;    // جمع العملات الثانوية
input bool     InpCollectForexCrosses = true;   // جمع العملات التقاطعية
input bool     InpCollectMetals = true;         // جمع المعادن
input bool     InpCollectCrypto = true;         // جمع العملات الرقمية
input bool     InpCollectEnergy = true;         // جمع الطاقة والسلع
input bool     InpCollectIndices = true;        // جمع المؤشرات
input bool     InpCheckSymbolExists = true;     // التحقق من توفر الرمز قبل الجمع
input int      InpMaxConcurrentSymbols = 100;   // الحد الأقصى للعملات المعالجة

// قوائم العملات حسب الفئة
string ForexMajors[] = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
};

string ForexMinors[] = {
    "USDMXN", "USDZAR", "USDTRY", "USDNOK", "USDSEK", "USDSGD", "USDHKD",
    "USDDKK", "USDPLN", "USDHUF", "USDCZK", "USDILS", "USDRUB", "USDCNH",
    "USDINR", "USDTHB", "USDKRW", "USDARS", "USDBRL", "USDCLP"
};

string ForexCrosses[] = {
    // EUR Crosses
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
    "EURNOK", "EURSEK", "EURPLN", "EURTRY", "EURZAR", "EURSGD",
    
    // GBP Crosses
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "GBPNOK", "GBPSEK", "GBPPLN", "GBPTRY", "GBPZAR", "GBPSGD",
    
    // JPY Crosses
    "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "SGDJPY", "ZARJPY",
    
    // Other Crosses
    "AUDCAD", "AUDNZD", "AUDCHF", "AUDSGD",
    "NZDCAD", "NZDCHF", "NZDSGD",
    "CADCHF", "CADSGD", "CHFSGD"
};

string Metals[] = {
    // Precious Metals vs USD
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    
    // Precious Metals vs Other Currencies
    "XAUEUR", "XAGEUR", "XAUAUD", "XAUGBP", "XAUJPY", "XAUCHF",
    "XAGAUD", "XAGGBP", "XAGJPY", "XAGCHF",
    
    // Industrial Metals (if available)
    "COPPER", "ALUMINUM", "ZINC", "NICKEL", "LEAD", "TIN"
};

string Crypto[] = {
    // Major Cryptocurrencies
    "BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD", "ADAUSD", "DOTUSD",
    "SOLUSD", "AVAXUSD", "MATICUSD", "LINKUSD", "UNIUSD",
    
    // Secondary Cryptocurrencies
    "LTCUSD", "BCHUSD", "ETCUSD", "XLMUSD", "EOSUSD", "XTZUSD",
    "ATOMUSD", "ALGOUSD", "VETUSD", "FILUSD", "AAVEUSD",
    
    // Crypto Pairs
    "ETHBTC", "BNBBTC", "XRPBTC", "ADABTC", "DOTBTC"
};

string Energy[] = {
    // Oil
    "WTIUSD", "XBRUSD", "XNGUSD",
    
    // Natural Gas and Others
    "GASUSD", "HEATUSD", "GASOILUSD",
    
    // Agricultural (if available)
    "WHEATUSD", "CORNUSD", "SOYUSD", "SUGARUSD", "COFFEEUSD", "COCOAUSD"
};

string Indices[] = {
    // US Indices
    "US30", "US500", "US100", "USTEC", "US2000",
    
    // European Indices
    "DE30", "DE40", "UK100", "FR40", "EU50", "ES35", "IT40",
    
    // Asian Indices
    "JP225", "CN50", "HK50", "AU200", "SG30", "IN50",
    
    // Other Indices
    "CA60", "ZA40", "MX35"
};

// الفريمات المطلوبة
ENUM_TIMEFRAMES timeframes[] = {
    PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H4, PERIOD_D1
};

// متغيرات عامة
string activeSymbols[];        // قائمة العملات النشطة
int totalSymbolsToProcess = 0;
int currentSymbolIndex = 0;
int currentTimeframeIndex = 0;
int currentBatchStart = 0;
bool isProcessing = false;
datetime startTime;

// إحصائيات
struct CollectionStats {
    int totalSymbolsChecked;
    int totalSymbolsAvailable;
    int totalSymbolsProcessed;
    int totalBarsCollected;
    int totalBatchesSent;
    int totalBatchesFailed;
    string failedSymbols[];
};

CollectionStats stats;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    PrintHeader();
    
    // بناء قائمة العملات النشطة
    BuildActiveSymbolsList();
    
    if(ArraySize(activeSymbols) == 0) {
        Print("❌ لا توجد عملات متاحة للمعالجة!");
        return(INIT_FAILED);
    }
    
    // طباعة معلومات البدء
    PrintStartInfo();
    
    startTime = TimeCurrent();
    
    // بدء المعالجة
    EventSetMillisecondTimer(100);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| بناء قائمة العملات النشطة                                         |
//+------------------------------------------------------------------+
void BuildActiveSymbolsList()
{
    Print("🔍 جاري فحص العملات المتاحة...");
    
    ArrayResize(activeSymbols, 0);
    stats.totalSymbolsChecked = 0;
    
    // جمع العملات حسب الفئات المختارة
    if(InpCollectForexMajors) {
        AddSymbolsToList(ForexMajors, "Forex Majors");
    }
    
    if(InpCollectForexMinors) {
        AddSymbolsToList(ForexMinors, "Forex Minors");
    }
    
    if(InpCollectForexCrosses) {
        AddSymbolsToList(ForexCrosses, "Forex Crosses");
    }
    
    if(InpCollectMetals) {
        AddSymbolsToList(Metals, "Metals");
    }
    
    if(InpCollectCrypto) {
        AddSymbolsToList(Crypto, "Cryptocurrencies");
    }
    
    if(InpCollectEnergy) {
        AddSymbolsToList(Energy, "Energy & Commodities");
    }
    
    if(InpCollectIndices) {
        AddSymbolsToList(Indices, "Indices");
    }
    
    // تطبيق الحد الأقصى
    if(ArraySize(activeSymbols) > InpMaxConcurrentSymbols) {
        Print("⚠️ تم تقليل عدد العملات من ", ArraySize(activeSymbols), 
              " إلى ", InpMaxConcurrentSymbols);
        ArrayResize(activeSymbols, InpMaxConcurrentSymbols);
    }
    
    stats.totalSymbolsAvailable = ArraySize(activeSymbols);
    totalSymbolsToProcess = ArraySize(activeSymbols);
}

//+------------------------------------------------------------------+
//| إضافة عملات إلى القائمة                                           |
//+------------------------------------------------------------------+
void AddSymbolsToList(string &symbolArray[], string category)
{
    int addedCount = 0;
    
    for(int i = 0; i < ArraySize(symbolArray); i++) {
        string symbol = symbolArray[i];
        stats.totalSymbolsChecked++;
        
        // إضافة suffix إذا لزم الأمر
        string symbolToCheck = symbol;
        if(SymbolSelect(symbol + "m", false)) {
            symbolToCheck = symbol + "m";
        } else if(SymbolSelect(symbol + ".a", false)) {
            symbolToCheck = symbol + ".a";
        }
        
        // التحقق من توفر الرمز
        if(!InpCheckSymbolExists || SymbolSelect(symbolToCheck, true)) {
            // التحقق من عدم التكرار
            bool exists = false;
            for(int j = 0; j < ArraySize(activeSymbols); j++) {
                if(activeSymbols[j] == symbolToCheck) {
                    exists = true;
                    break;
                }
            }
            
            if(!exists) {
                ArrayResize(activeSymbols, ArraySize(activeSymbols) + 1);
                activeSymbols[ArraySize(activeSymbols) - 1] = symbolToCheck;
                addedCount++;
            }
        }
    }
    
    if(InpDebugMode && addedCount > 0) {
        Print("✅ ", category, ": تم إضافة ", addedCount, " عملة");
    }
}

//+------------------------------------------------------------------+
//| طباعة رأس البرنامج                                                |
//+------------------------------------------------------------------+
void PrintHeader()
{
    Print("\n");
    Print("╔══════════════════════════════════════════════════════╗");
    Print("║     🌟 FOREX ML DATA COLLECTOR ULTIMATE 🌟           ║");
    Print("║          جامع البيانات الشامل النهائي                 ║");
    Print("║              يدعم جميع الأسواق المتاحة                 ║");
    Print("╚══════════════════════════════════════════════════════╝");
    Print("");
}

//+------------------------------------------------------------------+
//| طباعة معلومات البدء                                               |
//+------------------------------------------------------------------+
void PrintStartInfo()
{
    Print("📊 ملخص العملات المتاحة:");
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("🔍 تم فحص: ", stats.totalSymbolsChecked, " رمز");
    Print("✅ متاح للمعالجة: ", stats.totalSymbolsAvailable, " عملة");
    Print("⏰ الفريمات: ", ArraySize(timeframes));
    Print("📈 إجمالي المجموعات: ", stats.totalSymbolsAvailable * ArraySize(timeframes));
    Print("📅 سنوات البيانات: ", InpYearsOfData);
    Print("📦 حجم الدفعة: ", InpBarsPerBatch, " شمعة");
    Print("🌐 Server: ", InpServerURL);
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // طباعة عينة من العملات
    Print("\n📋 عينة من العملات المتاحة:");
    int sampleSize = MathMin(10, ArraySize(activeSymbols));
    for(int i = 0; i < sampleSize; i++) {
        Print("  • ", activeSymbols[i]);
    }
    if(ArraySize(activeSymbols) > sampleSize) {
        Print("  • ... و ", ArraySize(activeSymbols) - sampleSize, " عملة أخرى");
    }
    Print("");
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
    ENUM_TIMEFRAMES tf = timeframes[currentTimeframeIndex];
    
    // الحصول على البيانات
    datetime endTime = TimeCurrent();
    datetime startTimeData = endTime - (InpYearsOfData * 365 * 24 * 60 * 60);
    
    MqlRates rates[];
    int totalAvailable = CopyRates(symbol, tf, startTimeData, endTime, rates);
    
    if(totalAvailable <= 0) {
        if(InpDebugMode) {
            Print("⚠️ لا توجد بيانات: ", symbol, " ", EnumToString(tf));
        }
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // طباعة بداية معالجة جديدة
    if(currentBatchStart == 0) {
        Print("\n📊 معالجة: ", symbol, " ", EnumToString(tf), 
              " [", currentSymbolIndex + 1, "/", ArraySize(activeSymbols), "]");
        Print("   📈 الشموع المتاحة: ", totalAvailable);
    }
    
    // حساب نطاق الدفعة
    int batchEnd = MathMin(currentBatchStart + InpBarsPerBatch, totalAvailable);
    int batchSize = batchEnd - currentBatchStart;
    
    if(batchSize <= 0) {
        stats.totalSymbolsProcessed++;
        PrintProgress(symbol, tf, totalAvailable, totalAvailable);
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // إنشاء وإرسال البيانات
    string jsonData = CreateBatchJSON(symbol, tf, rates, currentBatchStart, batchEnd);
    bool success = SendBatchData(jsonData);
    
    if(success) {
        stats.totalBarsCollected += batchSize;
        stats.totalBatchesSent++;
        currentBatchStart = batchEnd;
        
        PrintProgress(symbol, tf, currentBatchStart, totalAvailable);
    } else {
        stats.totalBatchesFailed++;
        
        // إضافة للعملات الفاشلة
        bool alreadyFailed = false;
        for(int i = 0; i < ArraySize(stats.failedSymbols); i++) {
            if(stats.failedSymbols[i] == symbol) {
                alreadyFailed = true;
                break;
            }
        }
        
        if(!alreadyFailed) {
            ArrayResize(stats.failedSymbols, ArraySize(stats.failedSymbols) + 1);
            stats.failedSymbols[ArraySize(stats.failedSymbols) - 1] = symbol;
        }
        
        // تخطي هذه المجموعة بعد 3 محاولات فاشلة
        Print("❌ فشل الإرسال - تخطي ", symbol, " ", EnumToString(tf));
        MoveToNextCombination();
    }
    
    Sleep(InpDelayBetweenBatches);
    isProcessing = false;
}

//+------------------------------------------------------------------+
//| طباعة التقدم                                                       |
//+------------------------------------------------------------------+
void PrintProgress(string symbol, ENUM_TIMEFRAMES tf, int current, int total)
{
    double percentage = (current * 100.0) / total;
    string progressBar = CreateProgressBar(percentage);
    
    Print("   ", progressBar, " ", DoubleToString(percentage, 1), "% ",
          "(", current, "/", total, ")");
}

//+------------------------------------------------------------------+
//| إنشاء شريط التقدم                                                  |
//+------------------------------------------------------------------+
string CreateProgressBar(double percentage)
{
    int filled = (int)(percentage / 5); // 20 خانة
    string bar = "[";
    
    for(int i = 0; i < 20; i++) {
        if(i < filled) {
            bar += "█";
        } else {
            bar += "░";
        }
    }
    
    bar += "]";
    return bar;
}

//+------------------------------------------------------------------+
//| الانتقال للمجموعة التالية                                          |
//+------------------------------------------------------------------+
void MoveToNextCombination()
{
    currentBatchStart = 0;
    currentTimeframeIndex++;
    
    if(currentTimeframeIndex >= ArraySize(timeframes)) {
        currentTimeframeIndex = 0;
        currentSymbolIndex++;
        
        // طباعة ملخص سريع كل 10 عملات
        if(currentSymbolIndex % 10 == 0 && currentSymbolIndex < ArraySize(activeSymbols)) {
            PrintQuickSummary();
        }
    }
}

//+------------------------------------------------------------------+
//| طباعة ملخص سريع                                                   |
//+------------------------------------------------------------------+
void PrintQuickSummary()
{
    double overallProgress = (currentSymbolIndex * 100.0) / ArraySize(activeSymbols);
    datetime elapsed = TimeCurrent() - startTime;
    
    Print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("📊 ملخص التقدم: ", DoubleToString(overallProgress, 1), "%");
    Print("✅ تمت معالجة: ", currentSymbolIndex, " من ", ArraySize(activeSymbols), " عملة");
    Print("📦 الدفعات المرسلة: ", stats.totalBatchesSent);
    Print("📈 الشموع المجمعة: ", stats.totalBarsCollected);
    
    if(stats.totalBatchesFailed > 0) {
        Print("❌ الدفعات الفاشلة: ", stats.totalBatchesFailed);
    }
    
    Print("⏱️ الوقت المنقضي: ", TimeToString(elapsed, TIME_MINUTES));
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
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
    
    if(res == 200) {
        if(InpDebugMode) {
            string response = CharArrayToString(result);
            // عرض استجابة مختصرة
            if(StringLen(response) > 100) {
                Print("   ✅ ", StringSubstr(response, 0, 100), "...");
            } else {
                Print("   ✅ ", response);
            }
        }
        return true;
    } else if(res == -1) {
        int error = GetLastError();
        if(error == 4014) {
            Print("❌ خطأ: يجب السماح بـ URL في إعدادات MT5");
            EventKillTimer();
        }
        return false;
    } else {
        if(InpDebugMode) {
            Print("   ❌ HTTP Error: ", res);
        }
        return false;
    }
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
//| عند اكتمال الجمع                                                   |
//+------------------------------------------------------------------+
void OnComplete()
{
    EventKillTimer();
    
    datetime totalTime = TimeCurrent() - startTime;
    
    Print("\n");
    Print("╔══════════════════════════════════════════════════════╗");
    Print("║              📊 التقرير النهائي الشامل 📊              ║");
    Print("╚══════════════════════════════════════════════════════╝");
    Print("");
    Print("📈 الإحصائيات النهائية:");
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Print("🔍 العملات المفحوصة: ", stats.totalSymbolsChecked);
    Print("✅ العملات المتاحة: ", stats.totalSymbolsAvailable);
    Print("📊 العملات المعالجة: ", stats.totalSymbolsProcessed);
    Print("📦 الدفعات المرسلة: ", stats.totalBatchesSent);
    Print("📈 الشموع المجمعة: ", FormatNumber(stats.totalBarsCollected));
    
    if(stats.totalBatchesFailed > 0) {
        Print("❌ الدفعات الفاشلة: ", stats.totalBatchesFailed);
    }
    
    if(ArraySize(stats.failedSymbols) > 0) {
        Print("\n⚠️ العملات التي فشل جمعها:");
        for(int i = 0; i < ArraySize(stats.failedSymbols); i++) {
            Print("  • ", stats.failedSymbols[i]);
        }
    }
    
    Print("\n⏱️ الوقت الإجمالي: ", TimeToString(totalTime, TIME_MINUTES|TIME_SECONDS));
    
    double avgBarsPerMinute = 0;
    if(totalTime > 0) {
        avgBarsPerMinute = (stats.totalBarsCollected * 60.0) / totalTime;
        Print("⚡ معدل الجمع: ", DoubleToString(avgBarsPerMinute, 0), " شمعة/دقيقة");
    }
    
    Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    if(stats.totalBatchesFailed == 0) {
        Print("\n🎉 تهانينا! تم جمع جميع البيانات بنجاح! 🎉");
    } else {
        Print("\n✅ تم الانتهاء مع بعض الأخطاء. راجع التفاصيل أعلاه.");
    }
    
    Print("\n💡 نصيحة: يمكنك الآن بدء تدريب النماذج على البيانات المجمعة!");
}

//+------------------------------------------------------------------+
//| تنسيق الأرقام الكبيرة                                              |
//+------------------------------------------------------------------+
string FormatNumber(int number)
{
    string result = IntegerToString(number);
    int len = StringLen(result);
    
    if(len > 3) {
        string formatted = "";
        int count = 0;
        
        for(int i = len - 1; i >= 0; i--) {
            if(count == 3) {
                formatted = "," + formatted;
                count = 0;
            }
            formatted = StringSubstr(result, i, 1) + formatted;
            count++;
        }
        
        return formatted;
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    if(reason != REASON_CHARTCHANGE && reason != REASON_RECOMPILE) {
        Print("\n🏁 جامع البيانات الشامل - تم الإيقاف");
        Print("   السبب: ", GetDeInitReasonText(reason));
    }
}

//+------------------------------------------------------------------+
//| الحصول على نص سبب الإيقاف                                        |
//+------------------------------------------------------------------+
string GetDeInitReasonText(int reason)
{
    switch(reason) {
        case REASON_PROGRAM:     return "تم الإيقاف بواسطة المستخدم";
        case REASON_REMOVE:      return "تم إزالة البرنامج من الشارت";
        case REASON_RECOMPILE:   return "إعادة تجميع";
        case REASON_CHARTCHANGE: return "تغيير الشارت أو الفريم";
        case REASON_CHARTCLOSE:  return "إغلاق الشارت";
        case REASON_PARAMETERS:  return "تغيير المعاملات";
        case REASON_ACCOUNT:     return "تغيير الحساب";
        default:                 return "سبب غير معروف";
    }
}

//+------------------------------------------------------------------+
//| Tick function                                                      |
//+------------------------------------------------------------------+
void OnTick()
{
    // لا نحتاج معالجة التيك
}