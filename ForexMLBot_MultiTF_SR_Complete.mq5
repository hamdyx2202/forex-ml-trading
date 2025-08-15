//+------------------------------------------------------------------+
//|                                ForexMLBot_MultiTF_SR_Complete.mq5|
//|           Multi-Pair + Multi-Timeframe + Support/Resistance      |
//|                    Version 5.1 - Complete Edition                |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System with S/R"
#property link      "https://github.com/hamdysoltan/forex-ml-trading"
#property version   "5.1"
#property strict

// ============== إعدادات الخادم ==============
input string   ServerUrl = "http://69.62.121.53:5000";  // عنوان الخادم
input int      UpdateIntervalSeconds = 60;               // فترة التحديث (ثانية)
input int      BarsToSend = 200;                        // عدد الشموع المرسلة

// ============== إعدادات التداول ==============
input bool     EnableTrading = true;                    // تفعيل التداول
input double   RiskPerTrade = 0.01;                     // المخاطرة لكل صفقة (1%)
input int      MagicNumber = 123456;                    // الرقم السحري
input double   MinCombinedConfidence = 0.75;            // الحد الأدنى للثقة المجمعة

// ============== إعدادات الدعم والمقاومة ==============
input bool     UseSupportResistance = true;             // استخدام الدعم والمقاومة
input int      SRLookbackPeriod = 100;                  // فترة البحث عن S/R
input double   MinSRStrength = 2.0;                     // الحد الأدنى لقوة المستوى (عدد اللمسات)
input double   SRBuffer = 5.0;                          // المسافة الآمنة من المستوى (نقاط)
input bool     PreferSROverATR = true;                  // تفضيل S/R على ATR

// ============== إعدادات حساب SL/TP ==============
enum ENUM_SL_METHOD {
    SL_SR,          // الدعم والمقاومة
    SL_ATR,         // ATR
    SL_HYBRID,      // هجين (الأفضل من الاثنين)
    SL_FIXED        // ثابت
};

enum ENUM_TP_METHOD {
    TP_SR,          // الدعم والمقاومة
    TP_FIXED_RR,    // نسبة Risk:Reward ثابتة
    TP_DYNAMIC      // ديناميكي حسب السوق
};

input ENUM_SL_METHOD SLMethod = SL_HYBRID;              // طريقة حساب SL
input ENUM_TP_METHOD TPMethod = TP_SR;                  // طريقة حساب TP
input double   DefaultRiskReward = 2.0;                  // نسبة Risk:Reward الافتراضية

// ============== إعدادات الحماية ==============
input double   MaxSLPips = 100.0;                       // الحد الأقصى لـ SL (نقاط)
input double   MinSLPips = 10.0;                        // الحد الأدنى لـ SL (نقاط)
input bool     Force70Features = false;                 // فرض 70 ميزة (false = 75)
input bool     CheckModelNames = true;                  // التحقق من أسماء النماذج

// ============== إعدادات Break Even و Trailing ==============
input bool     UseBreakEven = true;                     // استخدام Break Even
input double   BreakEvenTriggerPips = 20.0;             // نقاط التفعيل
input double   BreakEvenProfitPips = 2.0;               // الربح عند Break Even

input bool     UseTrailingStop = true;                  // استخدام Trailing Stop
input double   TrailingStartPips = 30.0;                // بداية Trailing
input double   TrailingStepPips = 10.0;                 // خطوة Trailing
input double   TrailingMinDistance = 10.0;              // أقل مسافة

// ============== إعدادات الأزواج ==============
input bool     UseAllInstruments = true;                // استخدام جميع الأدوات
input string   CustomPairs = "EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCAD,USDCHF,EURJPY,GBPJPY,EURGBP,XAUUSD,XAGUSD,USOIL,US30,NAS100,BTCUSD,ETHUSD"; // أزواج مخصصة

// ============== إعدادات العرض ==============
input bool     ShowDashboard = true;                    // عرض لوحة المعلومات
input bool     ShowSRLevels = true;                     // عرض مستويات S/R على الشارت
input color    SupportColor = clrGreen;                  // لون الدعم
input color    ResistanceColor = clrRed;                // لون المقاومة

// الأطر الزمنية المستخدمة
ENUM_TIMEFRAMES timeframes[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};
string timeframeNames[] = {"M5", "M15", "H1", "H4"};

// هياكل البيانات
struct InstrumentInfo {
    string symbol;
    string type;        // forex_major, forex_minor, metals, energy, indices, crypto, stocks
    double pipValue;
    double minLot;
    double maxLot;
    double lotStep;
    double typicalSpread;
    int    dailyRangePips;
    double maxSLPips;   // حسب نوع الأداة
    double minSLPips;   // حسب نوع الأداة
};

struct SRLevel {
    double price;
    string type;        // support/resistance
    double strength;
    int    touches;
    datetime lastTouch;
};

// متغيرات عامة
InstrumentInfo instruments[];
string activePairs[];
SRLevel supportLevels[];
SRLevel resistanceLevels[];
datetime lastUpdateTime[];
string lastSignals[];
double lastConfidence[];
double combinedConfidence[];
int totalPairs = 0;

// متغيرات للوحة المعلومات
int dashboardX = 10;
int dashboardY = 30;
int lineHeight = 20;
color textColor = clrWhite;
color headerColor = clrGold;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 ForexML Bot v5.1 with Support/Resistance Started");
    Print("📡 Server URL: ", ServerUrl);
    Print("📊 Features: ", Force70Features ? "70" : "75", " (S/R included)");
    Print("🎯 SL Method: ", EnumToString(SLMethod));
    Print("🎯 TP Method: ", EnumToString(TPMethod));
    
    // تحضير قائمة الأدوات
    if(!InitializeInstruments()) {
        Alert("❌ Failed to initialize instruments!");
        return(INIT_FAILED);
    }
    
    // تهيئة المصفوفات
    ArrayResize(lastUpdateTime, totalPairs);
    ArrayResize(lastSignals, totalPairs);
    ArrayResize(lastConfidence, totalPairs);
    ArrayResize(combinedConfidence, totalPairs);
    
    for(int i = 0; i < totalPairs; i++) {
        lastUpdateTime[i] = 0;
        lastSignals[i] = "NONE";
        lastConfidence[i] = 0;
        combinedConfidence[i] = 0;
    }
    
    Print("✅ Monitoring ", totalPairs, " instruments");
    Print("✅ Using ", ArraySize(timeframes), " timeframes");
    Print("✅ Total models: ", totalPairs * ArraySize(timeframes));
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Initialize instruments based on settings                         |
//+------------------------------------------------------------------+
bool InitializeInstruments()
{
    // إعداد معلومات الأدوات
    SetupInstrumentDatabase();
    
    if(UseAllInstruments) {
        // استخدام جميع الأدوات المفعلة
        totalPairs = ArraySize(instruments);
        ArrayResize(activePairs, totalPairs);
        
        for(int i = 0; i < totalPairs; i++) {
            activePairs[i] = instruments[i].symbol;
            Print("📊 Added: ", activePairs[i], " (", instruments[i].type, ")");
        }
    }
    else {
        // استخدام الأزواج المخصصة
        string pairs[];
        totalPairs = StringSplit(CustomPairs, ',', pairs);
        ArrayResize(activePairs, totalPairs);
        
        for(int i = 0; i < totalPairs; i++) {
            StringTrimLeft(pairs[i]);
            StringTrimRight(pairs[i]);
            
            // إضافة suffix إذا لزم
            string suffix = GetSymbolSuffix();
            if(suffix != "" && StringFind(pairs[i], suffix) < 0) {
                pairs[i] += suffix;
            }
            
            activePairs[i] = pairs[i];
            Print("📊 Monitoring: ", activePairs[i]);
        }
    }
    
    return totalPairs > 0;
}

//+------------------------------------------------------------------+
//| Setup instrument database                                        |
//+------------------------------------------------------------------+
void SetupInstrumentDatabase()
{
    ArrayResize(instruments, 0);
    
    // Forex Majors
    AddInstrument("EURUSD", "forex_major", 0.0001, 0.01, 100.0, 0.01, 1.2, 80, 100, 10);
    AddInstrument("GBPUSD", "forex_major", 0.0001, 0.01, 100.0, 0.01, 1.5, 100, 100, 10);
    AddInstrument("USDJPY", "forex_major", 0.01, 0.01, 100.0, 0.01, 1.0, 70, 100, 10);
    AddInstrument("USDCHF", "forex_major", 0.0001, 0.01, 100.0, 0.01, 1.8, 60, 100, 10);
    AddInstrument("AUDUSD", "forex_major", 0.0001, 0.01, 100.0, 0.01, 1.3, 70, 100, 10);
    AddInstrument("NZDUSD", "forex_major", 0.0001, 0.01, 100.0, 0.01, 2.0, 65, 100, 10);
    AddInstrument("USDCAD", "forex_major", 0.0001, 0.01, 100.0, 0.01, 1.5, 75, 100, 10);
    
    // Forex Minors
    AddInstrument("EURJPY", "forex_minor", 0.01, 0.01, 100.0, 0.01, 2.0, 100, 150, 15);
    AddInstrument("GBPJPY", "forex_minor", 0.01, 0.01, 100.0, 0.01, 2.5, 150, 150, 15);
    AddInstrument("EURGBP", "forex_minor", 0.0001, 0.01, 100.0, 0.01, 1.8, 60, 150, 15);
    
    // Metals
    AddInstrument("XAUUSD", "metals", 0.01, 0.01, 50.0, 0.01, 30, 2000, 500, 50);
    AddInstrument("XAGUSD", "metals", 0.001, 0.01, 50.0, 0.01, 3, 50, 300, 30);
    
    // Energy
    AddInstrument("USOIL", "energy", 0.01, 0.1, 100.0, 0.1, 3, 200, 300, 30);
    
    // Indices
    AddInstrument("US30", "indices", 1.0, 0.1, 50.0, 0.1, 3, 300, 200, 20);
    AddInstrument("NAS100", "indices", 0.1, 0.1, 50.0, 0.1, 2, 200, 200, 20);
    
    // Crypto
    AddInstrument("BTCUSD", "crypto", 1.0, 0.01, 5.0, 0.01, 50, 2000, 1000, 100);
    AddInstrument("ETHUSD", "crypto", 0.1, 0.01, 10.0, 0.01, 5, 150, 500, 50);
}

//+------------------------------------------------------------------+
//| Add instrument to database                                       |
//+------------------------------------------------------------------+
void AddInstrument(string symbol, string type, double pipVal, double minLot, 
                   double maxLot, double lotStep, double spread, int range,
                   double maxSL, double minSL)
{
    int size = ArraySize(instruments);
    ArrayResize(instruments, size + 1);
    
    instruments[size].symbol = symbol;
    instruments[size].type = type;
    instruments[size].pipValue = pipVal;
    instruments[size].minLot = minLot;
    instruments[size].maxLot = maxLot;
    instruments[size].lotStep = lotStep;
    instruments[size].typicalSpread = spread;
    instruments[size].dailyRangePips = range;
    instruments[size].maxSLPips = maxSL;
    instruments[size].minSLPips = minSL;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // معالجة كل زوج
    for(int i = 0; i < totalPairs; i++) {
        if(TimeCurrent() - lastUpdateTime[i] >= UpdateIntervalSeconds) {
            ProcessPair(i);
            lastUpdateTime[i] = TimeCurrent();
        }
    }
    
    // تحديث الصفقات المفتوحة
    if(UseBreakEven || UseTrailingStop) {
        UpdateOpenPositions();
    }
    
    // عرض لوحة المعلومات
    if(ShowDashboard) {
        UpdateDashboard();
    }
    
    // عرض مستويات S/R
    if(ShowSRLevels && UseSupportResistance) {
        DrawSRLevels();
    }
}

//+------------------------------------------------------------------+
//| Process a specific pair                                          |
//+------------------------------------------------------------------+
void ProcessPair(int pairIndex)
{
    string symbol = activePairs[pairIndex];
    
    // جمع البيانات لجميع الأطر الزمنية
    string jsonData = "{";
    jsonData += "\"symbol\": \"" + symbol + "\",";
    jsonData += "\"features_version\": " + (Force70Features ? "70" : "75") + ",";
    jsonData += "\"timeframes\": {";
    
    double tfConfidences[];
    string tfSignals[];
    ArrayResize(tfConfidences, ArraySize(timeframes));
    ArrayResize(tfSignals, ArraySize(timeframes));
    
    for(int tf = 0; tf < ArraySize(timeframes); tf++) {
        string tfData = CollectDataForTimeframe(symbol, timeframes[tf]);
        jsonData += "\"" + timeframeNames[tf] + "\": " + tfData;
        
        if(tf < ArraySize(timeframes) - 1) jsonData += ",";
        
        // إرسال البيانات للخادم والحصول على التنبؤ
        string response = SendToServer(jsonData);
        ParseResponse(response, tfSignals[tf], tfConfidences[tf]);
    }
    
    jsonData += "}";
    
    // حساب الإشارة المجمعة
    string combinedSignal;
    double combinedConf;
    CalculateCombinedSignal(tfSignals, tfConfidences, combinedSignal, combinedConf);
    
    lastSignals[pairIndex] = combinedSignal;
    combinedConfidence[pairIndex] = combinedConf;
    
    // التداول إذا كانت الثقة كافية
    if(EnableTrading && combinedConf >= MinCombinedConfidence) {
        if(combinedSignal == "BUY" || combinedSignal == "SELL") {
            ExecuteTrade(symbol, combinedSignal, combinedConf);
        }
    }
}

//+------------------------------------------------------------------+
//| Collect data for specific timeframe                              |
//+------------------------------------------------------------------+
string CollectDataForTimeframe(string symbol, ENUM_TIMEFRAMES timeframe)
{
    string data = "{";
    data += "\"bars\": [";
    
    for(int i = BarsToSend - 1; i >= 0; i--) {
        data += "{";
        data += "\"time\": \"" + TimeToString(iTime(symbol, timeframe, i)) + "\",";
        data += "\"open\": " + DoubleToString(iOpen(symbol, timeframe, i), 5) + ",";
        data += "\"high\": " + DoubleToString(iHigh(symbol, timeframe, i), 5) + ",";
        data += "\"low\": " + DoubleToString(iLow(symbol, timeframe, i), 5) + ",";
        data += "\"close\": " + DoubleToString(iClose(symbol, timeframe, i), 5) + ",";
        data += "\"volume\": " + IntegerToString(iVolume(symbol, timeframe, i));
        data += "}";
        
        if(i > 0) data += ",";
    }
    
    data += "]";
    
    // إضافة مستويات S/R إذا كانت مفعلة
    if(UseSupportResistance) {
        CalculateSRLevels(symbol, timeframe);
        data += ",\"sr_levels\": " + GetSRLevelsJSON();
    }
    
    data += "}";
    
    return data;
}

//+------------------------------------------------------------------+
//| Send data to server and get response                            |
//+------------------------------------------------------------------+
string SendToServer(string jsonData)
{
    string headers = "Content-Type: application/json\r\n";
    string url = ServerUrl + "/predict";
    
    char postData[];
    StringToCharArray(jsonData, postData);
    
    char result[];
    string response = "";
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, 5000, postData, result, response);
    
    if(res < 0) {
        int error = GetLastError();
        Print("❌ WebRequest failed: ", error);
        return "";
    }
    
    return response;
}

//+------------------------------------------------------------------+
//| Parse server response                                            |
//+------------------------------------------------------------------+
void ParseResponse(string response, string &signal, double &confidence)
{
    signal = "NONE";
    confidence = 0;
    
    if(response == "") return;
    
    // بحث بسيط عن القيم
    int signalPos = StringFind(response, "\"signal\":");
    int confPos = StringFind(response, "\"confidence\":");
    
    if(signalPos >= 0 && confPos >= 0) {
        // استخراج الإشارة
        int signalStart = signalPos + 10;
        int signalEnd = StringFind(response, "\"", signalStart + 1);
        if(signalEnd > signalStart) {
            signal = StringSubstr(response, signalStart + 1, signalEnd - signalStart - 1);
        }
        
        // استخراج الثقة
        int confStart = confPos + 13;
        int confEnd = StringFind(response, ",", confStart);
        if(confEnd < 0) confEnd = StringFind(response, "}", confStart);
        
        if(confEnd > confStart) {
            string confStr = StringSubstr(response, confStart, confEnd - confStart);
            confidence = StringToDouble(confStr);
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate combined signal from multiple timeframes               |
//+------------------------------------------------------------------+
void CalculateCombinedSignal(string &signals[], double &confidences[], 
                             string &combinedSignal, double &combinedConfidence)
{
    int buyCount = 0, sellCount = 0;
    double buyConfSum = 0, sellConfSum = 0;
    
    for(int i = 0; i < ArraySize(signals); i++) {
        if(signals[i] == "BUY") {
            buyCount++;
            buyConfSum += confidences[i];
        }
        else if(signals[i] == "SELL") {
            sellCount++;
            sellConfSum += confidences[i];
        }
    }
    
    // تحديد الإشارة المجمعة
    if(buyCount > sellCount && buyCount >= 3) {
        combinedSignal = "BUY";
        combinedConfidence = buyConfSum / buyCount;
    }
    else if(sellCount > buyCount && sellCount >= 3) {
        combinedSignal = "SELL";
        combinedConfidence = sellConfSum / sellCount;
    }
    else {
        combinedSignal = "NONE";
        combinedConfidence = 0;
    }
}

//+------------------------------------------------------------------+
//| Check if there's an open position for symbol                    |
//+------------------------------------------------------------------+
bool HasOpenPosition(string symbol)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            if(PositionGetString(POSITION_SYMBOL) == symbol && 
               PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double slDistance)
{
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * RiskPerTrade;
    
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double slPips = slDistance / (point * 10);
    
    double lotSize = riskAmount / (slPips * tickValue * 10);
    
    // الحصول على معلومات الأداة
    InstrumentInfo info = GetInstrumentInfo(symbol);
    
    // التقريب حسب lot step
    lotSize = MathFloor(lotSize / info.lotStep) * info.lotStep;
    
    // التحقق من الحدود
    lotSize = MathMax(info.minLot, MathMin(info.maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Get instrument info                                              |
//+------------------------------------------------------------------+
InstrumentInfo GetInstrumentInfo(string symbol)
{
    InstrumentInfo defaultInfo;
    defaultInfo.symbol = symbol;
    defaultInfo.type = "forex_major";
    defaultInfo.pipValue = 0.0001;
    defaultInfo.minLot = 0.01;
    defaultInfo.maxLot = 100.0;
    defaultInfo.lotStep = 0.01;
    defaultInfo.typicalSpread = 2.0;
    defaultInfo.dailyRangePips = 100;
    defaultInfo.maxSLPips = 100;
    defaultInfo.minSLPips = 10;
    
    // البحث عن الأداة في قاعدة البيانات
    for(int i = 0; i < ArraySize(instruments); i++) {
        if(instruments[i].symbol == symbol || 
           StringFind(symbol, instruments[i].symbol) >= 0) {
            return instruments[i];
        }
    }
    
    return defaultInfo;
}

//+------------------------------------------------------------------+
//| Calculate ATR based SL/TP                                       |
//+------------------------------------------------------------------+
void CalculateATRBasedSLTP(string symbol, string signal, double price, double &sl, double &tp)
{
    double atr = iATR(symbol, PERIOD_CURRENT, 14);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    InstrumentInfo info = GetInstrumentInfo(symbol);
    
    double atrMultiplier = 1.5;
    double slDistance = atr * atrMultiplier;
    
    if(signal == "BUY") {
        sl = price - slDistance;
        tp = price + (slDistance * DefaultRiskReward);
    }
    else {
        sl = price + slDistance;
        tp = price - (slDistance * DefaultRiskReward);
    }
}

//+------------------------------------------------------------------+
//| Calculate hybrid SL/TP                                          |
//+------------------------------------------------------------------+
void CalculateHybridSLTP(string symbol, string signal, double price, double &sl, double &tp)
{
    double srSL = 0, srTP = 0;
    double atrSL = 0, atrTP = 0;
    
    // حساب بالطريقتين
    CalculateSRBasedSLTP(symbol, signal, price, srSL, srTP);
    CalculateATRBasedSLTP(symbol, signal, price, atrSL, atrTP);
    
    // اختيار الأفضل
    if(signal == "BUY") {
        // للشراء: SL الأعلى (الأقرب) أكثر أماناً
        sl = (srSL > atrSL && srSL > 0) ? srSL : atrSL;
        // TP حسب الطريقة المفضلة
        tp = PreferSROverATR && srTP > 0 ? srTP : atrTP;
    }
    else {
        // للبيع: SL الأقل (الأقرب) أكثر أماناً
        sl = (srSL < atrSL && srSL > 0) ? srSL : atrSL;
        // TP حسب الطريقة المفضلة
        tp = PreferSROverATR && srTP > 0 ? srTP : atrTP;
    }
}

//+------------------------------------------------------------------+
//| Check and apply break even                                      |
//+------------------------------------------------------------------+
void CheckBreakEven(string symbol, ENUM_POSITION_TYPE posType, double openPrice, 
                    double currentPrice, double currentSL, double point)
{
    double profitPips = 0;
    double newSL = 0;
    
    if(posType == POSITION_TYPE_BUY) {
        profitPips = (currentPrice - openPrice) / (point * 10);
        if(profitPips >= BreakEvenTriggerPips && currentSL < openPrice) {
            newSL = openPrice + (BreakEvenProfitPips * point * 10);
        }
    }
    else {
        profitPips = (openPrice - currentPrice) / (point * 10);
        if(profitPips >= BreakEvenTriggerPips && currentSL > openPrice) {
            newSL = openPrice - (BreakEvenProfitPips * point * 10);
        }
    }
    
    // تعديل SL إذا لزم
    if(newSL != 0 && newSL != currentSL) {
        MqlTradeRequest request = {};
        MqlTradeResult result = {};
        
        request.action = TRADE_ACTION_SLTP;
        request.position = PositionGetInteger(POSITION_TICKET);
        request.sl = NormalizeDouble(newSL, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
        request.tp = PositionGetDouble(POSITION_TP);
        request.symbol = symbol;
        
        if(OrderSend(request, result)) {
            Print("✅ Break Even activated for ", symbol);
        }
    }
}

//+------------------------------------------------------------------+
//| Check and apply trailing stop                                   |
//+------------------------------------------------------------------+
void CheckTrailingStop(string symbol, ENUM_POSITION_TYPE posType, double openPrice,
                       double currentPrice, double currentSL, double point)
{
    double profitPips = 0;
    double newSL = 0;
    
    if(posType == POSITION_TYPE_BUY) {
        profitPips = (currentPrice - openPrice) / (point * 10);
        if(profitPips >= TrailingStartPips) {
            newSL = currentPrice - (TrailingMinDistance * point * 10);
            if(newSL > currentSL + (TrailingStepPips * point * 10)) {
                // تحديث SL
                ModifyPosition(symbol, newSL, 0);
            }
        }
    }
    else {
        profitPips = (openPrice - currentPrice) / (point * 10);
        if(profitPips >= TrailingStartPips) {
            newSL = currentPrice + (TrailingMinDistance * point * 10);
            if(newSL < currentSL - (TrailingStepPips * point * 10)) {
                // تحديث SL
                ModifyPosition(symbol, newSL, 0);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Modify position SL/TP                                           |
//+------------------------------------------------------------------+
void ModifyPosition(string symbol, double newSL, double newTP)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_SLTP;
    request.position = PositionGetInteger(POSITION_TICKET);
    request.symbol = symbol;
    
    if(newSL > 0) {
        request.sl = NormalizeDouble(newSL, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
    } else {
        request.sl = PositionGetDouble(POSITION_SL);
    }
    
    if(newTP > 0) {
        request.tp = NormalizeDouble(newTP, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
    } else {
        request.tp = PositionGetDouble(POSITION_TP);
    }
    
    OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Update dashboard display                                         |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
    // حذف النصوص القديمة
    ObjectsDeleteAll(0, "Dashboard_");
    
    int y = dashboardY;
    
    // العنوان
    CreateLabel("Dashboard_Title", dashboardX, y, "=== FOREX ML BOT v5.1 ===", headerColor, 12);
    y += lineHeight + 5;
    
    // معلومات عامة
    CreateLabel("Dashboard_Pairs", dashboardX, y, "Pairs: " + IntegerToString(totalPairs), textColor, 10);
    y += lineHeight;
    
    CreateLabel("Dashboard_Time", dashboardX, y, "Time: " + TimeToString(TimeCurrent()), textColor, 10);
    y += lineHeight + 5;
    
    // حالة الأزواج
    for(int i = 0; i < MathMin(totalPairs, 10); i++) {
        string status = activePairs[i] + ": " + lastSignals[i] + 
                       " (" + DoubleToString(combinedConfidence[i] * 100, 1) + "%)";
        
        color statusColor = textColor;
        if(lastSignals[i] == "BUY") statusColor = clrLime;
        else if(lastSignals[i] == "SELL") statusColor = clrRed;
        
        CreateLabel("Dashboard_Pair" + IntegerToString(i), dashboardX, y, status, statusColor, 10);
        y += lineHeight;
    }
    
    // معلومات الحساب
    y += 5;
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double profit = equity - balance;
    
    CreateLabel("Dashboard_Balance", dashboardX, y, "Balance: $" + DoubleToString(balance, 2), textColor, 10);
    y += lineHeight;
    
    color profitColor = profit >= 0 ? clrLime : clrRed;
    CreateLabel("Dashboard_Profit", dashboardX, y, "P/L: $" + DoubleToString(profit, 2), profitColor, 10);
}

//+------------------------------------------------------------------+
//| Create text label                                                |
//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text, color clr, int size)
{
    ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
    ObjectSetString(0, name, OBJPROP_FONT, "Arial");
}

//+------------------------------------------------------------------+
//| Draw S/R levels on chart                                        |
//+------------------------------------------------------------------+
void DrawSRLevels()
{
    // حذف المستويات القديمة
    ObjectsDeleteAll(0, "SR_");
    
    // رسم مستويات الدعم
    for(int i = 0; i < MathMin(ArraySize(supportLevels), 5); i++) {
        string name = "SR_Support_" + IntegerToString(i);
        ObjectCreate(0, name, OBJ_HLINE, 0, 0, supportLevels[i].price);
        ObjectSetInteger(0, name, OBJPROP_COLOR, SupportColor);
        ObjectSetInteger(0, name, OBJPROP_WIDTH, 1 + (int)supportLevels[i].strength);
        ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetString(0, name, OBJPROP_TEXT, "Support " + DoubleToString(supportLevels[i].strength, 1));
    }
    
    // رسم مستويات المقاومة
    for(int i = 0; i < MathMin(ArraySize(resistanceLevels), 5); i++) {
        string name = "SR_Resistance_" + IntegerToString(i);
        ObjectCreate(0, name, OBJ_HLINE, 0, 0, resistanceLevels[i].price);
        ObjectSetInteger(0, name, OBJPROP_COLOR, ResistanceColor);
        ObjectSetInteger(0, name, OBJPROP_WIDTH, 1 + (int)resistanceLevels[i].strength);
        ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetString(0, name, OBJPROP_TEXT, "Resistance " + DoubleToString(resistanceLevels[i].strength, 1));
    }
}

//+------------------------------------------------------------------+
//| Get symbol suffix                                                |
//+------------------------------------------------------------------+
string GetSymbolSuffix()
{
    string currentSymbol = _Symbol;
    string basePairs[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"};
    
    for(int i = 0; i < ArraySize(basePairs); i++) {
        int pos = StringFind(currentSymbol, basePairs[i]);
        if(pos >= 0) {
            return StringSubstr(currentSymbol, pos + StringLen(basePairs[i]));
        }
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| Sort S/R levels by distance from current price                  |
//+------------------------------------------------------------------+
void SortSRLevels(double currentPrice)
{
    // ترتيب مستويات الدعم (الأقرب أولاً)
    for(int i = 0; i < ArraySize(supportLevels) - 1; i++) {
        for(int j = i + 1; j < ArraySize(supportLevels); j++) {
            double dist1 = currentPrice - supportLevels[i].price;
            double dist2 = currentPrice - supportLevels[j].price;
            
            if(dist2 < dist1) {
                SRLevel temp = supportLevels[i];
                supportLevels[i] = supportLevels[j];
                supportLevels[j] = temp;
            }
        }
    }
    
    // ترتيب مستويات المقاومة (الأقرب أولاً)
    for(int i = 0; i < ArraySize(resistanceLevels) - 1; i++) {
        for(int j = i + 1; j < ArraySize(resistanceLevels); j++) {
            double dist1 = resistanceLevels[i].price - currentPrice;
            double dist2 = resistanceLevels[j].price - currentPrice;
            
            if(dist2 < dist1) {
                SRLevel temp = resistanceLevels[i];
                resistanceLevels[i] = resistanceLevels[j];
                resistanceLevels[j] = temp;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Get S/R levels as JSON string                                    |
//+------------------------------------------------------------------+
string GetSRLevelsJSON()
{
    string json = "{";
    
    // مستويات الدعم
    json += "\"support\": [";
    for(int i = 0; i < MathMin(ArraySize(supportLevels), 5); i++) {
        if(i > 0) json += ",";
        json += "{";
        json += "\"price\": " + DoubleToString(supportLevels[i].price, 5) + ",";
        json += "\"strength\": " + DoubleToString(supportLevels[i].strength, 2) + ",";
        json += "\"touches\": " + IntegerToString(supportLevels[i].touches);
        json += "}";
    }
    json += "],";
    
    // مستويات المقاومة
    json += "\"resistance\": [";
    for(int i = 0; i < MathMin(ArraySize(resistanceLevels), 5); i++) {
        if(i > 0) json += ",";
        json += "{";
        json += "\"price\": " + DoubleToString(resistanceLevels[i].price, 5) + ",";
        json += "\"strength\": " + DoubleToString(resistanceLevels[i].strength, 2) + ",";
        json += "\"touches\": " + IntegerToString(resistanceLevels[i].touches);
        json += "}";
    }
    json += "]";
    
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // حذف كل الكائنات المرسومة
    ObjectsDeleteAll(0);
    
    Print("👋 ForexML Bot stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Calculate Support/Resistance levels                              |
//+------------------------------------------------------------------+
void CalculateSRLevels(string symbol, ENUM_TIMEFRAMES timeframe)
{
    ArrayResize(supportLevels, 0);
    ArrayResize(resistanceLevels, 0);
    
    double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
    
    // البحث عن القمم والقيعان
    for(int i = 2; i < SRLookbackPeriod - 2; i++) {
        double high = iHigh(symbol, timeframe, i);
        double low = iLow(symbol, timeframe, i);
        
        // فحص القمة
        if(high > iHigh(symbol, timeframe, i-1) && high > iHigh(symbol, timeframe, i-2) &&
           high > iHigh(symbol, timeframe, i+1) && high > iHigh(symbol, timeframe, i+2)) {
            
            int touches = CountTouches(symbol, timeframe, high, true);
            if(touches >= MinSRStrength) {
                AddSRLevel(high, "resistance", touches);
            }
        }
        
        // فحص القاع
        if(low < iLow(symbol, timeframe, i-1) && low < iLow(symbol, timeframe, i-2) &&
           low < iLow(symbol, timeframe, i+1) && low < iLow(symbol, timeframe, i+2)) {
            
            int touches = CountTouches(symbol, timeframe, low, false);
            if(touches >= MinSRStrength) {
                AddSRLevel(low, "support", touches);
            }
        }
    }
    
    // ترتيب المستويات حسب القرب من السعر الحالي
    SortSRLevels(currentPrice);
}

//+------------------------------------------------------------------+
//| Count how many times price touched a level                       |
//+------------------------------------------------------------------+
int CountTouches(string symbol, ENUM_TIMEFRAMES timeframe, double level, bool isResistance)
{
    int touches = 0;
    double tolerance = SRBuffer * SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    for(int i = 0; i < SRLookbackPeriod; i++) {
        if(isResistance) {
            if(MathAbs(iHigh(symbol, timeframe, i) - level) <= tolerance) {
                touches++;
            }
        }
        else {
            if(MathAbs(iLow(symbol, timeframe, i) - level) <= tolerance) {
                touches++;
            }
        }
    }
    
    return touches;
}

//+------------------------------------------------------------------+
//| Add S/R level to array                                           |
//+------------------------------------------------------------------+
void AddSRLevel(double price, string type, int touches)
{
    SRLevel level;
    level.price = price;
    level.type = type;
    level.touches = touches;
    level.strength = (double)touches / MinSRStrength;
    level.lastTouch = TimeCurrent();
    
    if(type == "support") {
        int size = ArraySize(supportLevels);
        ArrayResize(supportLevels, size + 1);
        supportLevels[size] = level;
    }
    else {
        int size = ArraySize(resistanceLevels);
        ArrayResize(resistanceLevels, size + 1);
        resistanceLevels[size] = level;
    }
}

//+------------------------------------------------------------------+
//| Execute trade with S/R based SL/TP                              |
//+------------------------------------------------------------------+
void ExecuteTrade(string symbol, string signal, double confidence)
{
    // التحقق من وجود صفقة مفتوحة
    if(HasOpenPosition(symbol)) return;
    
    double price = (signal == "BUY") ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
    double sl = 0, tp = 0;
    
    // حساب SL/TP
    CalculateSLTP(symbol, signal, price, sl, tp);
    
    // حساب حجم الصفقة
    double lotSize = CalculateLotSize(symbol, MathAbs(price - sl));
    
    // فتح الصفقة
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = lotSize;
    request.type = (signal == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    request.price = price;
    request.sl = sl;
    request.tp = tp;
    request.magic = MagicNumber;
    request.comment = "ML_" + IntegerToString((int)(confidence * 100)) + "%";
    request.deviation = 10;
    
    if(OrderSend(request, result)) {
        Print("✅ ", signal, " ", symbol, " @ ", price, 
              " SL=", sl, " TP=", tp, " Conf=", confidence);
    }
    else {
        Print("❌ Trade failed: ", result.comment);
    }
}

//+------------------------------------------------------------------+
//| Calculate SL/TP based on method                                  |
//+------------------------------------------------------------------+
void CalculateSLTP(string symbol, string signal, double price, double &sl, double &tp)
{
    InstrumentInfo info = GetInstrumentInfo(symbol);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    switch(SLMethod) {
        case SL_SR:
            CalculateSRBasedSLTP(symbol, signal, price, sl, tp);
            break;
            
        case SL_ATR:
            CalculateATRBasedSLTP(symbol, signal, price, sl, tp);
            break;
            
        case SL_HYBRID:
            CalculateHybridSLTP(symbol, signal, price, sl, tp);
            break;
            
        case SL_FIXED:
            if(signal == "BUY") {
                sl = price - (info.minSLPips + 20) * point * 10;
                tp = price + (info.minSLPips + 20) * DefaultRiskReward * point * 10;
            }
            else {
                sl = price + (info.minSLPips + 20) * point * 10;
                tp = price - (info.minSLPips + 20) * DefaultRiskReward * point * 10;
            }
            break;
    }
    
    // التحقق من الحدود
    double slDistance = MathAbs(price - sl) / (point * 10);
    
    if(slDistance < info.minSLPips) {
        if(signal == "BUY") {
            sl = price - info.minSLPips * point * 10;
        }
        else {
            sl = price + info.minSLPips * point * 10;
        }
    }
    
    if(slDistance > info.maxSLPips) {
        if(signal == "BUY") {
            sl = price - info.maxSLPips * point * 10;
        }
        else {
            sl = price + info.maxSLPips * point * 10;
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate S/R based SL/TP                                        |
//+------------------------------------------------------------------+
void CalculateSRBasedSLTP(string symbol, string signal, double price, double &sl, double &tp)
{
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double buffer = SRBuffer * point;
    
    if(signal == "BUY") {
        // SL: تحت أقرب دعم قوي
        for(int i = 0; i < ArraySize(supportLevels); i++) {
            if(supportLevels[i].price < price && supportLevels[i].strength >= 1.0) {
                sl = supportLevels[i].price - buffer;
                break;
            }
        }
        
        // TP: عند أقرب مقاومة قوية
        for(int i = 0; i < ArraySize(resistanceLevels); i++) {
            if(resistanceLevels[i].price > price && resistanceLevels[i].strength >= 1.0) {
                tp = resistanceLevels[i].price - buffer;
                break;
            }
        }
    }
    else {
        // SL: فوق أقرب مقاومة قوية
        for(int i = 0; i < ArraySize(resistanceLevels); i++) {
            if(resistanceLevels[i].price > price && resistanceLevels[i].strength >= 1.0) {
                sl = resistanceLevels[i].price + buffer;
                break;
            }
        }
        
        // TP: عند أقرب دعم قوي
        for(int i = 0; i < ArraySize(supportLevels); i++) {
            if(supportLevels[i].price < price && supportLevels[i].strength >= 1.0) {
                tp = supportLevels[i].price + buffer;
                break;
            }
        }
    }
    
    // إذا لم نجد مستويات مناسبة، استخدم قيم افتراضية
    if(sl == 0 || tp == 0) {
        CalculateATRBasedSLTP(symbol, signal, price, sl, tp);
    }
}

//+------------------------------------------------------------------+
//| Update open positions (Break Even & Trailing Stop)              |
//+------------------------------------------------------------------+
void UpdateOpenPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
            
            string symbol = PositionGetString(POSITION_SYMBOL);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentSL = PositionGetDouble(POSITION_SL);
            double currentTP = PositionGetDouble(POSITION_TP);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
            
            // Break Even
            if(UseBreakEven) {
                CheckBreakEven(symbol, posType, openPrice, currentPrice, currentSL, point);
            }
            
            // Trailing Stop
            if(UseTrailingStop) {
                CheckTrailingStop(symbol, posType, openPrice, currentPrice, currentSL, point);
            }
        }
    }
}