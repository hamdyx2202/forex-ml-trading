//+------------------------------------------------------------------+
//|                            ForexMLDataCollector_Advanced.mq5      |
//|                    جامع البيانات المتقدم لجميع العملات والفريمات   |
//|                              مع تجزئة الإرسال ورسائل تفصيلية        |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://forexmltrading.com"
#property version   "2.00"
#property description "جامع بيانات متقدم مع تجزئة الإرسال لـ 3 سنوات"

// إعدادات الإدخال
input string   InpServerURL = "http://69.62.121.53:5000/api/historical_data"; // Server URL
input int      InpBarsPerBatch = 100;        // عدد الشموع في كل دفعة
input int      InpDelayBetweenBatches = 500; // تأخير بين الدفعات (ملي ثانية)
input int      InpYearsOfData = 3;           // عدد سنوات البيانات
input bool     InpDebugMode = true;          // وضع التصحيح (رسائل إضافية)

// العملات المطلوبة - قائمة شاملة
string symbols[] = {
    // أزواج الدولار الرئيسية (USD Majors)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    
    // أزواج الدولار الثانوية (USD Minors)
    "USDMXN", "USDZAR", "USDTRY", "USDNOK", "USDSEK", "USDSGD", "USDHKD",
    "USDDKK", "USDPLN", "USDHUF", "USDCZK", "USDILS", "USDRUB", "USDCNH",
    
    // العملات التقاطعية الرئيسية (Major Crosses)
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
    "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "AUDCAD", "AUDNZD", "AUDCHF",
    "NZDCAD", "NZDCHF", "CADCHF",
    
    // العملات التقاطعية الثانوية (Minor Crosses)
    "EURNOK", "EURSEK", "EURPLN", "EURTRY", "EURZAR",
    "GBPNOK", "GBPSEK", "GBPPLN", "GBPTRY", "GBPZAR",
    "AUDSGD", "NZDSGD",
    
    // المعادن الثمينة (Precious Metals)
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    "XAUEUR", "XAGEUR", "XAUAUD", "XAUGBP",
    
    // العملات الرقمية (Cryptocurrencies)
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD",
    "EOSUSD", "ADAUSD", "DOTUSD", "LINKUSD", "BNBUSD",
    
    // السلع والطاقة (Commodities & Energy)
    "WTIUSD", "XBRUSD", "XNGUSD",  // النفط والغاز
    "COPPER", "ALUMINUM", "ZINC",   // المعادن الصناعية
    
    // المؤشرات (إن كانت متاحة)
    "US30", "US500", "US100", "DE30", "UK100", "JP225", "AUS200"
};

// الفريمات المطلوبة
ENUM_TIMEFRAMES timeframes[] = {
    PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H4, PERIOD_D1
};

// متغيرات عامة
struct SendProgress {
    string symbol;
    ENUM_TIMEFRAMES timeframe;
    int totalBars;
    int sentBars;
    int failedBatches;
    bool completed;
};

SendProgress progress[];
int currentSymbolIndex = 0;
int currentTimeframeIndex = 0;
int currentBatchStart = 0;
bool isProcessing = false;
datetime startTime;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("🚀 بدء تشغيل جامع البيانات المتقدم");
    Print("========================================");
    
    // التحقق من الإعدادات
    if(InpBarsPerBatch < 10 || InpBarsPerBatch > 1000) {
        Print("❌ خطأ: حجم الدفعة يجب أن يكون بين 10 و 1000");
        return(INIT_PARAMETERS_INCORRECT);
    }
    
    // إعداد مصفوفة التقدم
    int totalCombinations = ArraySize(symbols) * ArraySize(timeframes);
    ArrayResize(progress, totalCombinations);
    
    // تهيئة التقدم
    int idx = 0;
    for(int i = 0; i < ArraySize(symbols); i++) {
        for(int j = 0; j < ArraySize(timeframes); j++) {
            progress[idx].symbol = symbols[i];
            progress[idx].timeframe = timeframes[j];
            progress[idx].totalBars = 0;
            progress[idx].sentBars = 0;
            progress[idx].failedBatches = 0;
            progress[idx].completed = false;
            idx++;
        }
    }
    
    // طباعة معلومات البدء
    Print("📊 عدد العملات: ", ArraySize(symbols));
    Print("⏰ عدد الفريمات: ", ArraySize(timeframes));
    Print("📈 إجمالي المجموعات: ", totalCombinations);
    Print("📅 سنوات البيانات: ", InpYearsOfData);
    Print("📦 حجم الدفعة: ", InpBarsPerBatch, " شمعة");
    Print("⏱️ التأخير بين الدفعات: ", InpDelayBetweenBatches, " ملي ثانية");
    Print("🌐 Server URL: ", InpServerURL);
    
    startTime = TimeCurrent();
    
    // بدء المعالجة
    EventSetMillisecondTimer(100);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    
    Print("========================================");
    Print("🏁 إيقاف جامع البيانات");
    PrintFinalReport();
    Print("========================================");
}

//+------------------------------------------------------------------+
//| Timer function                                                     |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!isProcessing) {
        ProcessNextBatch();
    }
}

//+------------------------------------------------------------------+
//| معالجة الدفعة التالية                                              |
//+------------------------------------------------------------------+
void ProcessNextBatch()
{
    // التحقق من اكتمال جميع العملات
    if(currentSymbolIndex >= ArraySize(symbols)) {
        EventKillTimer();
        Print("✅ اكتمل إرسال جميع البيانات!");
        PrintFinalReport();
        return;
    }
    
    isProcessing = true;
    
    string symbol = symbols[currentSymbolIndex];
    ENUM_TIMEFRAMES tf = timeframes[currentTimeframeIndex];
    
    // التحقق من توفر الرمز
    if(!SymbolSelect(symbol, true)) {
        Print("⚠️ تحذير: لا يمكن تحديد الرمز ", symbol);
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // حساب عدد الشموع المطلوبة
    datetime endTime = TimeCurrent();
    datetime startTimeData = endTime - (InpYearsOfData * 365 * 24 * 60 * 60);
    
    // الحصول على البيانات
    MqlRates rates[];
    int totalAvailable = CopyRates(symbol, tf, startTimeData, endTime, rates);
    
    if(totalAvailable <= 0) {
        Print("⚠️ لا توجد بيانات متاحة لـ ", symbol, " ", EnumToString(tf));
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // تحديث إجمالي الشموع
    int progressIdx = GetProgressIndex(symbol, tf);
    if(progress[progressIdx].totalBars == 0) {
        progress[progressIdx].totalBars = totalAvailable;
        Print("\n📊 بدء معالجة: ", symbol, " ", EnumToString(tf));
        Print("📈 إجمالي الشموع المتاحة: ", totalAvailable);
    }
    
    // حساب نطاق الدفعة الحالية
    int batchEnd = MathMin(currentBatchStart + InpBarsPerBatch, totalAvailable);
    int batchSize = batchEnd - currentBatchStart;
    
    if(batchSize <= 0) {
        // انتهت الشموع لهذه المجموعة
        progress[progressIdx].completed = true;
        Print("✅ اكتمل: ", symbol, " ", EnumToString(tf), 
              " - تم إرسال ", progress[progressIdx].sentBars, " من ", progress[progressIdx].totalBars);
        MoveToNextCombination();
        isProcessing = false;
        return;
    }
    
    // إنشاء دفعة البيانات
    string jsonData = CreateBatchJSON(symbol, tf, rates, currentBatchStart, batchEnd);
    
    // إرسال البيانات
    if(InpDebugMode) {
        Print("📤 إرسال دفعة: ", currentBatchStart + 1, "-", batchEnd, 
              " من ", totalAvailable, " (", batchSize, " شمعة)");
    }
    
    bool success = SendBatchData(jsonData);
    
    if(success) {
        progress[progressIdx].sentBars += batchSize;
        currentBatchStart = batchEnd;
        
        // طباعة التقدم
        double percentage = (progress[progressIdx].sentBars * 100.0) / progress[progressIdx].totalBars;
        if((int)percentage % 10 == 0 && InpDebugMode) {
            Print("📊 التقدم: ", symbol, " ", EnumToString(tf), 
                  " - ", DoubleToString(percentage, 1), "%",
                  " (", progress[progressIdx].sentBars, "/", progress[progressIdx].totalBars, ")");
        }
    } else {
        progress[progressIdx].failedBatches++;
        Print("❌ فشل إرسال الدفعة! المحاولة: ", progress[progressIdx].failedBatches);
        
        // إذا فشلت 3 محاولات، انتقل للمجموعة التالية
        if(progress[progressIdx].failedBatches >= 3) {
            Print("⚠️ تخطي ", symbol, " ", EnumToString(tf), " بعد 3 محاولات فاشلة");
            MoveToNextCombination();
        }
    }
    
    // تأخير قبل الدفعة التالية
    Sleep(InpDelayBetweenBatches);
    isProcessing = false;
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
        string response = CharArrayToString(result);
        if(InpDebugMode) {
            Print("✅ استجابة السيرفر: ", response);
        }
        return true;
    } else if(res == -1) {
        int error = GetLastError();
        Print("❌ خطأ WebRequest: ", error);
        if(error == 4014) {
            Print("⚠️ تحذير: يجب السماح بـ URL في إعدادات MT5");
            Print("   Tools -> Options -> Expert Advisors -> Allow WebRequest");
            Print("   أضف: ", InpServerURL);
        }
        return false;
    } else {
        Print("❌ خطأ HTTP: ", res);
        string response = CharArrayToString(result);
        Print("   الاستجابة: ", response);
        return false;
    }
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
    }
}

//+------------------------------------------------------------------+
//| الحصول على مؤشر التقدم                                            |
//+------------------------------------------------------------------+
int GetProgressIndex(string symbol, ENUM_TIMEFRAMES tf)
{
    for(int i = 0; i < ArraySize(progress); i++) {
        if(progress[i].symbol == symbol && progress[i].timeframe == tf) {
            return i;
        }
    }
    return -1;
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
        default: return "M5";
    }
}

//+------------------------------------------------------------------+
//| طباعة التقرير النهائي                                              |
//+------------------------------------------------------------------+
void PrintFinalReport()
{
    Print("\n========================================");
    Print("📊 التقرير النهائي لجمع البيانات");
    Print("========================================");
    
    int totalSuccess = 0;
    int totalFailed = 0;
    int totalBarsCollected = 0;
    
    for(int i = 0; i < ArraySize(progress); i++) {
        if(progress[i].completed) {
            totalSuccess++;
            totalBarsCollected += progress[i].sentBars;
        } else if(progress[i].failedBatches >= 3) {
            totalFailed++;
        }
        
        // طباعة التفاصيل للمجموعات غير المكتملة
        if(!progress[i].completed && progress[i].totalBars > 0) {
            Print("⚠️ غير مكتمل: ", progress[i].symbol, " ", EnumToString(progress[i].timeframe),
                  " - تم إرسال ", progress[i].sentBars, " من ", progress[i].totalBars);
        }
    }
    
    datetime endTime = TimeCurrent();
    int elapsedSeconds = (int)(endTime - startTime);
    int minutes = elapsedSeconds / 60;
    int seconds = elapsedSeconds % 60;
    
    Print("\n📈 الملخص:");
    Print("✅ مجموعات مكتملة: ", totalSuccess, " من ", ArraySize(progress));
    Print("❌ مجموعات فاشلة: ", totalFailed);
    Print("📊 إجمالي الشموع المرسلة: ", totalBarsCollected);
    Print("⏱️ الوقت المستغرق: ", minutes, " دقيقة و ", seconds, " ثانية");
    
    if(totalSuccess == ArraySize(progress)) {
        Print("\n🎉 تهانينا! تم جمع جميع البيانات بنجاح!");
    } else {
        Print("\n⚠️ لم يكتمل جمع بعض البيانات. راجع التفاصيل أعلاه.");
    }
}

//+------------------------------------------------------------------+
//| Tick function                                                      |
//+------------------------------------------------------------------+
void OnTick()
{
    // لا نحتاج معالجة التيك
}