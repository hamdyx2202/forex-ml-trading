//+------------------------------------------------------------------+
//|                                  ForexMLBot_MultiTF_Simple.mq5   |
//|                  Multi-Pair + Multi-Timeframe Simple Version     |
//|                           Version 5.0 - Without S/R              |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://github.com/hamdysoltan/forex-ml-trading"
#property version   "5.0"
#property strict

// إعدادات الإدخال
input string   ServerUrl = "http://69.62.121.53:5000";  // عنوان الخادم
input double   RiskPerTrade = 0.01;                     // المخاطرة لكل صفقة (1%)
input int      MagicNumber = 123456;                    // الرقم السحري
input int      BarsToSend = 200;                        // عدد الشموع المرسلة
input int      UpdateIntervalSeconds = 60;               // فترة التحديث (ثانية)
input bool     EnableTrading = true;                    // تفعيل التداول
input bool     ShowDashboard = true;                    // عرض لوحة المعلومات
input string   PairsToTrade = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURJPY,GBPJPY,XAUUSD"; // الأزواج للتداول
input bool     UseAllTimeframes = true;                 // استخدام جميع الأطر الزمنية
input double   MinCombinedConfidence = 0.75;            // الحد الأدنى للثقة المجمعة

// إعدادات SL/TP الثابتة
input double   FixedSLPips = 30.0;                      // SL ثابت (نقاط)
input double   FixedTPPips = 60.0;                      // TP ثابت (نقاط)
input bool     UseATR = true;                           // استخدام ATR للـ SL/TP
input double   ATRMultiplier = 1.5;                     // مضاعف ATR
input double   RiskRewardRatio = 2.0;                   // نسبة المخاطرة للربح

// إعدادات Break Even و Trailing
input bool     UseBreakEven = true;                     // استخدام Break Even
input double   BreakEvenTriggerPips = 20.0;             // نقاط التفعيل
input double   BreakEvenProfitPips = 2.0;               // الربح عند Break Even

input bool     UseTrailingStop = true;                  // استخدام Trailing Stop
input double   TrailingStartPips = 30.0;                // بداية Trailing
input double   TrailingStepPips = 10.0;                 // خطوة Trailing
input double   TrailingDistance = 15.0;                 // مسافة Trailing

// الأطر الزمنية المستخدمة
ENUM_TIMEFRAMES timeframes[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};
string timeframeNames[] = {"M5", "M15", "H1", "H4"};

// متغيرات عامة
string pairs[];
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
    Print("🚀 ForexML Multi-Timeframe Bot Started (Simple Version)");
    Print("📡 Server URL: ", ServerUrl);
    Print("📊 Using ", UseAllTimeframes ? "ALL timeframes" : "M5 only");
    
    // تقسيم الأزواج
    totalPairs = StringSplit(PairsToTrade, ',', pairs);
    
    // تهيئة المصفوفات
    ArrayResize(lastUpdateTime, totalPairs);
    ArrayResize(lastSignals, totalPairs);
    ArrayResize(lastConfidence, totalPairs);
    ArrayResize(combinedConfidence, totalPairs);
    
    // إضافة اللاحقة إذا لزم الأمر
    for(int i = 0; i < totalPairs; i++)
    {
        StringTrimLeft(pairs[i]);
        StringTrimRight(pairs[i]);
        
        string suffix = GetSymbolSuffix();
        if(suffix != "" && StringFind(pairs[i], suffix) < 0)
            pairs[i] += suffix;
        
        lastUpdateTime[i] = 0;
        lastSignals[i] = "NONE";
        lastConfidence[i] = 0;
        combinedConfidence[i] = 0;
        
        Print("📊 Monitoring: ", pairs[i]);
    }
    
    int modelsUsed = UseAllTimeframes ? totalPairs * ArraySize(timeframes) : totalPairs;
    Print("✅ Will use ", modelsUsed, " models (", UseAllTimeframes ? "100%" : "25%", " utilization)");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    for(int i = 0; i < totalPairs; i++)
    {
        if(TimeCurrent() - lastUpdateTime[i] >= UpdateIntervalSeconds)
        {
            ProcessPair(i);
            lastUpdateTime[i] = TimeCurrent();
        }
    }
    
    // تحديث الصفقات المفتوحة
    if(UseBreakEven || UseTrailingStop)
        UpdateOpenPositions();
    
    if(ShowDashboard)
        UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Process a specific pair                                          |
//+------------------------------------------------------------------+
void ProcessPair(int pairIndex)
{
    string symbol = pairs[pairIndex];
    
    if(UseAllTimeframes)
    {
        // جمع الإشارات من جميع الأطر الزمنية
        string signals[];
        double confidences[];
        ArrayResize(signals, ArraySize(timeframes));
        ArrayResize(confidences, ArraySize(timeframes));
        
        for(int tf = 0; tf < ArraySize(timeframes); tf++)
        {
            string data = CollectData(symbol, timeframes[tf]);
            string response = SendRequest(data);
            ParseResponse(response, signals[tf], confidences[tf]);
        }
        
        // حساب الإشارة المجمعة
        CalculateCombinedSignal(signals, confidences, 
                               lastSignals[pairIndex], combinedConfidence[pairIndex]);
    }
    else
    {
        // M5 فقط
        string data = CollectData(symbol, PERIOD_M5);
        string response = SendRequest(data);
        ParseResponse(response, lastSignals[pairIndex], lastConfidence[pairIndex]);
        combinedConfidence[pairIndex] = lastConfidence[pairIndex];
    }
    
    // تنفيذ التداول
    if(EnableTrading && combinedConfidence[pairIndex] >= MinCombinedConfidence)
    {
        if(lastSignals[pairIndex] == "BUY" || lastSignals[pairIndex] == "SELL")
        {
            if(!HasOpenPosition(symbol))
            {
                ExecuteTrade(symbol, lastSignals[pairIndex], combinedConfidence[pairIndex]);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Collect OHLCV data                                               |
//+------------------------------------------------------------------+
string CollectData(string symbol, ENUM_TIMEFRAMES timeframe)
{
    string data = "{";
    data += "\"symbol\": \"" + symbol + "\",";
    data += "\"timeframe\": \"" + PeriodToString(timeframe) + "\",";
    data += "\"bars\": [";
    
    for(int i = BarsToSend - 1; i >= 0; i--)
    {
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
    
    data += "]}";
    return data;
}

//+------------------------------------------------------------------+
//| Send HTTP request to server                                      |
//+------------------------------------------------------------------+
string SendRequest(string jsonData)
{
    string headers = "Content-Type: application/json\r\n";
    string url = ServerUrl + "/predict";
    
    char postData[];
    StringToCharArray(jsonData, postData);
    
    char result[];
    string response = "";
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, 5000, postData, result, response);
    
    if(res < 0)
    {
        int error = GetLastError();
        Print("❌ Error sending request: ", error);
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
    
    if(signalPos >= 0 && confPos >= 0)
    {
        // استخراج الإشارة
        int signalStart = signalPos + 10;
        int signalEnd = StringFind(response, "\"", signalStart + 1);
        if(signalEnd > signalStart)
        {
            signal = StringSubstr(response, signalStart + 1, signalEnd - signalStart - 1);
        }
        
        // استخراج الثقة
        int confStart = confPos + 13;
        int confEnd = StringFind(response, ",", confStart);
        if(confEnd < 0) confEnd = StringFind(response, "}", confStart);
        
        if(confEnd > confStart)
        {
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
    
    for(int i = 0; i < ArraySize(signals); i++)
    {
        if(signals[i] == "BUY")
        {
            buyCount++;
            buyConfSum += confidences[i];
        }
        else if(signals[i] == "SELL")
        {
            sellCount++;
            sellConfSum += confidences[i];
        }
    }
    
    // الأغلبية تحدد الإشارة
    if(buyCount > sellCount && buyCount >= 3)
    {
        combinedSignal = "BUY";
        combinedConfidence = buyConfSum / buyCount;
    }
    else if(sellCount > buyCount && sellCount >= 3)
    {
        combinedSignal = "SELL";
        combinedConfidence = sellConfSum / sellCount;
    }
    else
    {
        combinedSignal = "NONE";
        combinedConfidence = 0;
    }
}

//+------------------------------------------------------------------+
//| Check if position exists for symbol                             |
//+------------------------------------------------------------------+
bool HasOpenPosition(string symbol)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == symbol && 
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Execute trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(string symbol, string signal, double confidence)
{
    double price = (signal == "BUY") ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
    double sl = 0, tp = 0;
    
    // حساب SL/TP
    CalculateSLTP(symbol, signal, price, sl, tp);
    
    // حساب حجم الصفقة
    double lotSize = CalculateLotSize(symbol, MathAbs(price - sl));
    
    // تنفيذ الصفقة
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
    
    if(OrderSend(request, result))
    {
        Print("✅ ", signal, " ", symbol, " @ ", DoubleToString(price, 5), 
              " SL=", DoubleToString(sl, 5), " TP=", DoubleToString(tp, 5), 
              " Lot=", DoubleToString(lotSize, 2), " Conf=", DoubleToString(confidence * 100, 1), "%");
    }
    else
    {
        Print("❌ Trade failed: ", result.retcode, " - ", result.comment);
    }
}

//+------------------------------------------------------------------+
//| Calculate SL/TP                                                  |
//+------------------------------------------------------------------+
void CalculateSLTP(string symbol, string signal, double price, double &sl, double &tp)
{
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double pip = point * 10;
    
    if(UseATR)
    {
        // استخدام ATR
        double atr = iATR(symbol, PERIOD_CURRENT, 14);
        double slDistance = atr * ATRMultiplier;
        
        if(signal == "BUY")
        {
            sl = price - slDistance;
            tp = price + (slDistance * RiskRewardRatio);
        }
        else
        {
            sl = price + slDistance;
            tp = price - (slDistance * RiskRewardRatio);
        }
    }
    else
    {
        // استخدام قيم ثابتة
        double slPips = GetAdjustedPips(symbol, FixedSLPips);
        double tpPips = GetAdjustedPips(symbol, FixedTPPips);
        
        if(signal == "BUY")
        {
            sl = price - (slPips * pip);
            tp = price + (tpPips * pip);
        }
        else
        {
            sl = price + (slPips * pip);
            tp = price - (tpPips * pip);
        }
    }
    
    // تطبيع الأسعار
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);
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
    
    // التقريب والتحقق من الحدود
    double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Get adjusted pips based on symbol type                          |
//+------------------------------------------------------------------+
double GetAdjustedPips(string symbol, double basePips)
{
    string symbolUpper = StringToUpper(symbol);
    
    // الذهب
    if(StringFind(symbolUpper, "XAU") >= 0)
        return basePips * 10;
    
    // النفط
    if(StringFind(symbolUpper, "OIL") >= 0 || StringFind(symbolUpper, "WTI") >= 0)
        return basePips * 3;
    
    // المؤشرات
    if(StringFind(symbolUpper, "US30") >= 0 || StringFind(symbolUpper, "NAS") >= 0)
        return basePips * 2;
    
    // العملات الرقمية
    if(StringFind(symbolUpper, "BTC") >= 0)
        return basePips * 50;
    
    return basePips;
}

//+------------------------------------------------------------------+
//| Update open positions                                            |
//+------------------------------------------------------------------+
void UpdateOpenPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
            
            string symbol = PositionGetString(POSITION_SYMBOL);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentSL = PositionGetDouble(POSITION_SL);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
            double pip = point * 10;
            
            // Break Even
            if(UseBreakEven)
            {
                double profitPips = 0;
                
                if(posType == POSITION_TYPE_BUY)
                {
                    profitPips = (currentPrice - openPrice) / pip;
                    if(profitPips >= BreakEvenTriggerPips && currentSL < openPrice)
                    {
                        double newSL = openPrice + (BreakEvenProfitPips * pip);
                        ModifyPosition(PositionGetInteger(POSITION_TICKET), newSL, 0);
                    }
                }
                else
                {
                    profitPips = (openPrice - currentPrice) / pip;
                    if(profitPips >= BreakEvenTriggerPips && currentSL > openPrice)
                    {
                        double newSL = openPrice - (BreakEvenProfitPips * pip);
                        ModifyPosition(PositionGetInteger(POSITION_TICKET), newSL, 0);
                    }
                }
            }
            
            // Trailing Stop
            if(UseTrailingStop)
            {
                double profitPips = 0;
                
                if(posType == POSITION_TYPE_BUY)
                {
                    profitPips = (currentPrice - openPrice) / pip;
                    if(profitPips >= TrailingStartPips)
                    {
                        double newSL = currentPrice - (TrailingDistance * pip);
                        if(newSL > currentSL + (TrailingStepPips * pip))
                        {
                            ModifyPosition(PositionGetInteger(POSITION_TICKET), newSL, 0);
                        }
                    }
                }
                else
                {
                    profitPips = (openPrice - currentPrice) / pip;
                    if(profitPips >= TrailingStartPips)
                    {
                        double newSL = currentPrice + (TrailingDistance * pip);
                        if(newSL < currentSL - (TrailingStepPips * pip))
                        {
                            ModifyPosition(PositionGetInteger(POSITION_TICKET), newSL, 0);
                        }
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Modify position                                                  |
//+------------------------------------------------------------------+
void ModifyPosition(ulong ticket, double newSL, double newTP)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_SLTP;
    request.position = ticket;
    
    if(newSL > 0)
        request.sl = NormalizeDouble(newSL, (int)SymbolInfoInteger(PositionGetString(POSITION_SYMBOL), SYMBOL_DIGITS));
    else
        request.sl = PositionGetDouble(POSITION_SL);
    
    if(newTP > 0)
        request.tp = NormalizeDouble(newTP, (int)SymbolInfoInteger(PositionGetString(POSITION_SYMBOL), SYMBOL_DIGITS));
    else
        request.tp = PositionGetDouble(POSITION_TP);
    
    OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Update dashboard                                                 |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
    // حذف النصوص القديمة
    ObjectsDeleteAll(0, "ML_");
    
    int y = dashboardY;
    
    // العنوان
    CreateLabel("ML_Title", dashboardX, y, "=== FOREX ML BOT ===", headerColor, 12);
    y += lineHeight + 5;
    
    // معلومات عامة
    CreateLabel("ML_Pairs", dashboardX, y, "Pairs: " + IntegerToString(totalPairs), textColor, 10);
    y += lineHeight;
    
    CreateLabel("ML_Models", dashboardX, y, "Models: " + IntegerToString(UseAllTimeframes ? totalPairs * 4 : totalPairs), textColor, 10);
    y += lineHeight;
    
    CreateLabel("ML_Time", dashboardX, y, "Time: " + TimeToString(TimeCurrent()), textColor, 10);
    y += lineHeight + 5;
    
    // حالة الأزواج
    for(int i = 0; i < totalPairs; i++)
    {
        string text = pairs[i] + ": " + lastSignals[i];
        if(combinedConfidence[i] > 0)
            text += " (" + DoubleToString(combinedConfidence[i] * 100, 1) + "%)";
        
        color clr = textColor;
        if(lastSignals[i] == "BUY") clr = clrLime;
        else if(lastSignals[i] == "SELL") clr = clrRed;
        
        CreateLabel("ML_Pair" + IntegerToString(i), dashboardX, y, text, clr, 10);
        y += lineHeight;
    }
    
    // معلومات الحساب
    y += 5;
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double profit = equity - balance;
    
    CreateLabel("ML_Balance", dashboardX, y, "Balance: $" + DoubleToString(balance, 2), textColor, 10);
    y += lineHeight;
    
    color profitColor = profit >= 0 ? clrLime : clrRed;
    CreateLabel("ML_Profit", dashboardX, y, "P/L: $" + DoubleToString(profit, 2), profitColor, 10);
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
//| Get symbol suffix                                                |
//+------------------------------------------------------------------+
string GetSymbolSuffix()
{
    string currentSymbol = _Symbol;
    string basePairs[] = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"};
    
    for(int i = 0; i < ArraySize(basePairs); i++)
    {
        int pos = StringFind(currentSymbol, basePairs[i]);
        if(pos >= 0)
        {
            return StringSubstr(currentSymbol, pos + StringLen(basePairs[i]));
        }
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| Convert period to string                                        |
//+------------------------------------------------------------------+
string PeriodToString(ENUM_TIMEFRAMES period)
{
    switch(period)
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
        default:         return "M5";
    }
}

//+------------------------------------------------------------------+
//| String to upper case                                             |
//+------------------------------------------------------------------+
string StringToUpper(string str)
{
    string result = str;
    StringToUpper(result);
    return result;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // حذف الكائنات المرسومة
    ObjectsDeleteAll(0, "ML_");
    
    Print("👋 ForexML Bot stopped. Reason: ", reason);
}