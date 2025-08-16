//+------------------------------------------------------------------+
//|                              ForexMLBot_Advanced_V2.mq5          |
//|                      نظام التداول الآلي المتقدم - النسخة 2      |
//|                    متوافق مع نظام التعلم الآلي المحدث          |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System V2"
#property link      "https://forex-ml-trading.com"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- إعدادات الإدخال
input string   InpServerURL = "http://localhost:5000/api/predict_advanced"; // رابط السيرفر
input int      InpMagicNumber = 12345;        // الرقم السحري
input double   InpRiskPercent = 1.0;          // نسبة المخاطرة % من الرصيد
input int      InpMaxTrades = 10;             // أقصى عدد صفقات مفتوحة
input double   InpMinConfidence = 0.75;       // الحد الأدنى للثقة
input int      InpCandlesHistory = 200;       // عدد الشموع للإرسال
input bool     InpUseTrailingStop = true;    // استخدام Trailing Stop
input bool     InpUseMoveToBreakeven = true; // نقل SL للتعادل

//--- إعدادات الاستراتيجيات
input bool     InpUseUltraShort = false;     // استخدام Ultra Short (30 دقيقة)
input bool     InpUseScalping = true;        // استخدام Scalping (1 ساعة)
input bool     InpUseShortTerm = true;       // استخدام Short Term (2 ساعات)
input bool     InpUseMediumTerm = true;      // استخدام Medium Term (4 ساعات)
input bool     InpUseLongTerm = false;       // استخدام Long Term (24 ساعة)

//--- المتغيرات العامة
CTrade trade;
CPositionInfo position;
CAccountInfo account;

//--- قائمة الأزواج والفترات
string symbols[] = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
    "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD",
    "US30", "NAS100", "SP500", "OIL", "NATGAS"
};

ENUM_TIMEFRAMES timeframes[] = {
    PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H4
};

//--- تتبع الصفقات النشطة
struct ActiveTrade {
    ulong ticket;
    string symbol;
    string timeframe;
    string strategy;
    double sl;
    double tp1;
    double tp2;
    double tp3;
    int currentTP;
    bool trailingActive;
    double trailingDistance;
    double breakevenLevel;
    datetime entryTime;
};

ActiveTrade activeTrades[];

//--- آخر وقت للفحص
datetime lastCheckTime = 0;
int checkInterval = 60; // فحص كل دقيقة

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    PrintFormat("🚀 بدء تشغيل ForexMLBot Advanced V2");
    PrintFormat("📊 السيرفر: %s", InpServerURL);
    PrintFormat("💰 المخاطرة: %.1f%%", InpRiskPercent);
    PrintFormat("🎯 الحد الأدنى للثقة: %.2f", InpMinConfidence);
    
    // تكوين التداول
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    // طباعة الاستراتيجيات المفعلة
    Print("📋 الاستراتيجيات المفعلة:");
    if(InpUseUltraShort) Print("   ✅ Ultra Short");
    if(InpUseScalping) Print("   ✅ Scalping");
    if(InpUseShortTerm) Print("   ✅ Short Term");
    if(InpUseMediumTerm) Print("   ✅ Medium Term");
    if(InpUseLongTerm) Print("   ✅ Long Term");
    
    // التحقق من الأزواج المتاحة
    int availableSymbols = 0;
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        if(SymbolSelect(symbols[i], true))
        {
            availableSymbols++;
        }
    }
    PrintFormat("✅ عدد الأزواج المتاحة: %d من %d", availableSymbols, ArraySize(symbols));
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    PrintFormat("🛑 إيقاف ForexMLBot - السبب: %d", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // تحديث الصفقات النشطة
    UpdateActiveTrades();
    
    // فحص دوري للإشارات الجديدة
    if(TimeCurrent() - lastCheckTime >= checkInterval)
    {
        CheckForNewSignals();
        lastCheckTime = TimeCurrent();
    }
}

//+------------------------------------------------------------------+
//| فحص الإشارات الجديدة                                           |
//+------------------------------------------------------------------+
void CheckForNewSignals()
{
    // التحقق من عدد الصفقات المفتوحة
    if(PositionsTotal() >= InpMaxTrades)
    {
        PrintFormat("⚠️ الحد الأقصى للصفقات: %d", InpMaxTrades);
        return;
    }
    
    // فحص كل زوج وفترة زمنية
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        string symbol = symbols[i];
        
        // التحقق من توفر الرمز
        if(!SymbolSelect(symbol, true))
            continue;
            
        for(int j = 0; j < ArraySize(timeframes); j++)
        {
            ENUM_TIMEFRAMES timeframe = timeframes[j];
            
            // جمع البيانات وإرسالها للتنبؤ
            if(!ProcessSymbolTimeframe(symbol, timeframe))
            {
                // PrintFormat("❌ فشل معالجة %s %s", symbol, EnumToString(timeframe));
                continue;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| معالجة زوج وفترة زمنية محددة                                   |
//+------------------------------------------------------------------+
bool ProcessSymbolTimeframe(string symbol, ENUM_TIMEFRAMES timeframe)
{
    // التحقق من وجود صفقة مفتوحة
    if(HasOpenPosition(symbol, timeframe))
        return true;
    
    // جمع بيانات الشموع
    MqlRates rates[];
    if(CopyRates(symbol, timeframe, 0, InpCandlesHistory, rates) <= 0)
    {
        PrintFormat("❌ فشل جمع بيانات %s %s", symbol, EnumToString(timeframe));
        return false;
    }
    
    // إعداد البيانات للإرسال
    string jsonData = PrepareDataForPrediction(symbol, timeframe, rates);
    
    // إرسال البيانات والحصول على التنبؤ
    string response = SendPredictionRequest(jsonData);
    
    if(response == "")
    {
        // لا نطبع خطأ لكل فشل في الاتصال
        return false;
    }
    
    // معالجة الرد
    return ProcessPredictionResponse(symbol, timeframe, response);
}

//+------------------------------------------------------------------+
//| إعداد البيانات للتنبؤ                                          |
//+------------------------------------------------------------------+
string PrepareDataForPrediction(string symbol, ENUM_TIMEFRAMES timeframe, MqlRates &rates[])
{
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + TimeframeToString(timeframe) + "\",";
    json += "\"candles\":[";
    
    // إرسال آخر 200 شمعة
    for(int i = 0; i < ArraySize(rates); i++)
    {
        if(i > 0) json += ",";
        
        json += "{";
        json += "\"time\":" + IntegerToString(rates[i].time) + ",";
        json += "\"open\":" + DoubleToString(rates[i].open, 5) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, 5) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, 5) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, 5) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
        json += "}";
    }
    
    json += "],";
    
    // إضافة معلومات إضافية
    json += "\"account_balance\":" + DoubleToString(account.Balance(), 2) + ",";
    json += "\"account_equity\":" + DoubleToString(account.Equity(), 2) + ",";
    json += "\"open_positions\":" + IntegerToString(PositionsTotal()) + ",";
    
    // الاستراتيجيات المطلوبة
    json += "\"strategies\":[";
    bool first = true;
    
    if(InpUseUltraShort) { if(!first) json += ","; json += "\"ultra_short\""; first = false; }
    if(InpUseScalping) { if(!first) json += ","; json += "\"scalping\""; first = false; }
    if(InpUseShortTerm) { if(!first) json += ","; json += "\"short_term\""; first = false; }
    if(InpUseMediumTerm) { if(!first) json += ","; json += "\"medium_term\""; first = false; }
    if(InpUseLongTerm) { if(!first) json += ","; json += "\"long_term\""; first = false; }
    
    json += "]";
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| إرسال طلب التنبؤ                                               |
//+------------------------------------------------------------------+
string SendPredictionRequest(string jsonData)
{
    char post[];
    char result[];
    string headers;
    
    StringToCharArray(jsonData, post, 0, StringLen(jsonData));
    
    headers = "Content-Type: application/json\r\n";
    
    int res = WebRequest(
        "POST",
        InpServerURL,
        headers,
        5000,
        post,
        result,
        headers
    );
    
    if(res == -1)
    {
        int error = GetLastError();
        if(error != 4014) // لا نطبع خطأ URL غير مسموح
        {
            PrintFormat("❌ خطأ WebRequest: %d", error);
        }
        return "";
    }
    
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| معالجة رد التنبؤ                                               |
//+------------------------------------------------------------------+
bool ProcessPredictionResponse(string symbol, ENUM_TIMEFRAMES timeframe, string response)
{
    // تحليل JSON (نسخة مبسطة)
    // في الواقع، يجب استخدام مكتبة JSON مناسبة
    
    // البحث عن أفضل إشارة
    double bestConfidence = 0;
    int bestSignal = 1; // 0=Sell, 1=Hold, 2=Buy
    string bestStrategy = "";
    double bestSL = 0;
    double bestTP1 = 0;
    double bestTP2 = 0;
    double bestTP3 = 0;
    
    // استخراج البيانات من JSON
    if(!ParsePredictionResponse(response, bestSignal, bestConfidence, bestStrategy, 
                               bestSL, bestTP1, bestTP2, bestTP3))
    {
        return false;
    }
    
    // التحقق من الثقة
    if(bestConfidence < InpMinConfidence)
    {
        PrintFormat("📊 %s %s - ثقة منخفضة: %.2f%%", 
                   symbol, TimeframeToString(timeframe), bestConfidence * 100);
        return false;
    }
    
    // التحقق من الإشارة
    if(bestSignal == 1) // Hold/No Trade
    {
        return false;
    }
    
    PrintFormat("🎯 إشارة جديدة! %s %s", symbol, TimeframeToString(timeframe));
    PrintFormat("   📊 الاستراتيجية: %s", bestStrategy);
    PrintFormat("   🎯 الإشارة: %s", bestSignal == 2 ? "شراء" : "بيع");
    PrintFormat("   📈 الثقة: %.2f%%", bestConfidence * 100);
    PrintFormat("   🛑 SL: %.5f", bestSL);
    PrintFormat("   🎯 TP1: %.5f, TP2: %.5f, TP3: %.5f", bestTP1, bestTP2, bestTP3);
    
    // فتح الصفقة
    return OpenTrade(symbol, timeframe, bestSignal, bestStrategy, 
                     bestSL, bestTP1, bestTP2, bestTP3, bestConfidence);
}

//+------------------------------------------------------------------+
//| تحليل رد JSON                                                   |
//+------------------------------------------------------------------+
bool ParsePredictionResponse(string json, int &signal, double &confidence, 
                           string &strategy, double &sl, double &tp1, 
                           double &tp2, double &tp3)
{
    // تحليل مبسط - في الإنتاج يجب استخدام مكتبة JSON
    
    // مثال على الرد المتوقع:
    // {
    //   "predictions": {
    //     "scalping": {
    //       "signal": 2,
    //       "confidence": 0.85,
    //       "stop_loss": 1.0850,
    //       "take_profit_1": 1.0870,
    //       "take_profit_2": 1.0890,
    //       "take_profit_3": 1.0920
    //     }
    //   }
    // }
    
    // البحث عن أفضل استراتيجية
    string strategies[] = {"ultra_short", "scalping", "short_term", "medium_term", "long_term"};
    
    for(int i = 0; i < ArraySize(strategies); i++)
    {
        string strat = strategies[i];
        
        // البحث عن الاستراتيجية في JSON
        int stratPos = StringFind(json, "\"" + strat + "\"");
        if(stratPos == -1) continue;
        
        // استخراج البيانات
        double stratConfidence = ExtractDouble(json, "\"confidence\":", stratPos);
        if(stratConfidence > confidence)
        {
            confidence = stratConfidence;
            signal = (int)ExtractDouble(json, "\"signal\":", stratPos);
            strategy = strat;
            sl = ExtractDouble(json, "\"stop_loss\":", stratPos);
            tp1 = ExtractDouble(json, "\"take_profit_1\":", stratPos);
            tp2 = ExtractDouble(json, "\"take_profit_2\":", stratPos);
            tp3 = ExtractDouble(json, "\"take_profit_3\":", stratPos);
        }
    }
    
    return (confidence > 0);
}

//+------------------------------------------------------------------+
//| استخراج قيمة من JSON                                           |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key, int startPos = 0)
{
    int pos = StringFind(json, key, startPos);
    if(pos == -1) return 0;
    
    pos += StringLen(key);
    int endPos = StringFind(json, ",", pos);
    if(endPos == -1) endPos = StringFind(json, "}", pos);
    
    string value = StringSubstr(json, pos, endPos - pos);
    StringTrimLeft(value);
    StringTrimRight(value);
    
    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| فتح صفقة جديدة                                                  |
//+------------------------------------------------------------------+
bool OpenTrade(string symbol, ENUM_TIMEFRAMES timeframe, int signal, 
               string strategy, double sl, double tp1, double tp2, 
               double tp3, double confidence)
{
    ENUM_ORDER_TYPE orderType = (signal == 2) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    
    // حساب حجم الصفقة
    double lotSize = CalculateLotSize(symbol, sl);
    if(lotSize <= 0)
    {
        PrintFormat("❌ فشل حساب حجم الصفقة لـ %s", symbol);
        return false;
    }
    
    // الحصول على السعر الحالي
    double price = (orderType == ORDER_TYPE_BUY) ? 
                   SymbolInfoDouble(symbol, SYMBOL_ASK) : 
                   SymbolInfoDouble(symbol, SYMBOL_BID);
    
    // التحقق من SL/TP
    if(!ValidateSLTP(symbol, orderType, price, sl, tp1))
    {
        PrintFormat("❌ SL/TP غير صالح لـ %s", symbol);
        return false;
    }
    
    // إعداد التعليق
    string comment = StringFormat("%s_%s_%.0f%%", 
                                 strategy, 
                                 TimeframeToString(timeframe), 
                                 confidence * 100);
    
    // فتح الصفقة
    if(trade.PositionOpen(symbol, orderType, lotSize, price, sl, tp1, comment))
    {
        ulong ticket = trade.ResultOrder();
        
        PrintFormat("✅ تم فتح صفقة #%d", ticket);
        PrintFormat("   📊 %s %s %.2f لوت", 
                   symbol, 
                   orderType == ORDER_TYPE_BUY ? "شراء" : "بيع", 
                   lotSize);
        PrintFormat("   💵 السعر: %.5f, SL: %.5f, TP1: %.5f", price, sl, tp1);
        
        // إضافة للصفقات النشطة
        AddToActiveTrades(ticket, symbol, timeframe, strategy, sl, tp1, tp2, tp3);
        
        return true;
    }
    else
    {
        PrintFormat("❌ فشل فتح الصفقة: %s", trade.ResultComment());
        return false;
    }
}

//+------------------------------------------------------------------+
//| حساب حجم الصفقة بناءً على إدارة المخاطر                       |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double stopLoss)
{
    double balance = account.Balance();
    double riskAmount = balance * InpRiskPercent / 100.0;
    
    double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double slDistance = MathAbs(price - stopLoss);
    
    // حساب قيمة النقطة
    double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double pointValue = tickValue * SymbolInfoDouble(symbol, SYMBOL_POINT) / tickSize;
    
    // حساب عدد النقاط
    double points = slDistance / SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    // حساب حجم اللوت
    double lotSize = riskAmount / (points * pointValue);
    
    // التقريب للحد المسموح
    double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| التحقق من صحة SL/TP                                            |
//+------------------------------------------------------------------+
bool ValidateSLTP(string symbol, ENUM_ORDER_TYPE orderType, double price, 
                  double sl, double tp)
{
    double minStop = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL) * 
                     SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    if(orderType == ORDER_TYPE_BUY)
    {
        if(sl >= price - minStop) return false;
        if(tp <= price + minStop) return false;
    }
    else
    {
        if(sl <= price + minStop) return false;
        if(tp >= price - minStop) return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| إضافة صفقة للمتابعة                                             |
//+------------------------------------------------------------------+
void AddToActiveTrades(ulong ticket, string symbol, ENUM_TIMEFRAMES timeframe,
                      string strategy, double sl, double tp1, double tp2, double tp3)
{
    int size = ArraySize(activeTrades);
    ArrayResize(activeTrades, size + 1);
    
    activeTrades[size].ticket = ticket;
    activeTrades[size].symbol = symbol;
    activeTrades[size].timeframe = TimeframeToString(timeframe);
    activeTrades[size].strategy = strategy;
    activeTrades[size].sl = sl;
    activeTrades[size].tp1 = tp1;
    activeTrades[size].tp2 = tp2;
    activeTrades[size].tp3 = tp3;
    activeTrades[size].currentTP = 1;
    activeTrades[size].trailingActive = false;
    activeTrades[size].entryTime = TimeCurrent();
}

//+------------------------------------------------------------------+
//| تحديث الصفقات النشطة                                           |
//+------------------------------------------------------------------+
void UpdateActiveTrades()
{
    for(int i = ArraySize(activeTrades) - 1; i >= 0; i--)
    {
        if(!position.SelectByTicket(activeTrades[i].ticket))
        {
            // الصفقة مغلقة
            PrintFormat("📊 الصفقة #%d مغلقة", activeTrades[i].ticket);
            ArrayRemove(activeTrades, i, 1);
            continue;
        }
        
        // إدارة الصفقة
        ManageTrade(activeTrades[i]);
    }
}

//+------------------------------------------------------------------+
//| إدارة صفقة واحدة                                               |
//+------------------------------------------------------------------+
void ManageTrade(ActiveTrade &trade)
{
    double currentPrice = position.PriceCurrent();
    double entryPrice = position.PriceOpen();
    double currentSL = position.StopLoss();
    double currentTP = position.TakeProfit();
    
    // حساب الربح بالنقاط
    double profitPoints = 0;
    if(position.PositionType() == POSITION_TYPE_BUY)
    {
        profitPoints = (currentPrice - entryPrice) / SymbolInfoDouble(trade.symbol, SYMBOL_POINT);
    }
    else
    {
        profitPoints = (entryPrice - currentPrice) / SymbolInfoDouble(trade.symbol, SYMBOL_POINT);
    }
    
    // نقل SL للتعادل
    if(InpUseMoveToBreakeven && profitPoints >= 20 && currentSL != entryPrice)
    {
        if(trade.currentTP == 1 && !trade.trailingActive)
        {
            MoveToBreakeven(trade);
        }
    }
    
    // إدارة Take Profit المتدرج
    if(trade.currentTP == 1 && profitPoints >= 30)
    {
        // الانتقال إلى TP2
        UpdateTakeProfit(trade, 2);
    }
    else if(trade.currentTP == 2 && profitPoints >= 50)
    {
        // الانتقال إلى TP3 وتفعيل Trailing
        UpdateTakeProfit(trade, 3);
        
        if(InpUseTrailingStop)
        {
            trade.trailingActive = true;
            trade.trailingDistance = GetTrailingDistance(trade.strategy);
        }
    }
    
    // Trailing Stop
    if(trade.trailingActive && InpUseTrailingStop)
    {
        UpdateTrailingStop(trade);
    }
}

//+------------------------------------------------------------------+
//| نقل Stop Loss للتعادل                                          |
//+------------------------------------------------------------------+
void MoveToBreakeven(ActiveTrade &trade)
{
    double entryPrice = position.PriceOpen();
    double spread = SymbolInfoInteger(trade.symbol, SYMBOL_SPREAD) * 
                    SymbolInfoDouble(trade.symbol, SYMBOL_POINT);
    
    double newSL = entryPrice;
    if(position.PositionType() == POSITION_TYPE_BUY)
        newSL += spread;
    else
        newSL -= spread;
    
    if(trade.Modify(trade.ticket, newSL, position.TakeProfit()))
    {
        PrintFormat("✅ نقل SL للتعادل للصفقة #%d", trade.ticket);
        trade.breakevenLevel = newSL;
    }
}

//+------------------------------------------------------------------+
//| تحديث Take Profit                                               |
//+------------------------------------------------------------------+
void UpdateTakeProfit(ActiveTrade &trade, int tpLevel)
{
    double newTP = 0;
    
    switch(tpLevel)
    {
        case 2: newTP = trade.tp2; break;
        case 3: newTP = trade.tp3; break;
        default: return;
    }
    
    if(trade.Modify(trade.ticket, position.StopLoss(), newTP))
    {
        PrintFormat("✅ تحديث TP%d للصفقة #%d", tpLevel, trade.ticket);
        trade.currentTP = tpLevel;
    }
}

//+------------------------------------------------------------------+
//| الحصول على مسافة Trailing Stop                                 |
//+------------------------------------------------------------------+
double GetTrailingDistance(string strategy)
{
    // مسافات مختلفة حسب الاستراتيجية
    if(strategy == "ultra_short") return 10;
    else if(strategy == "scalping") return 15;
    else if(strategy == "short_term") return 20;
    else if(strategy == "medium_term") return 30;
    else if(strategy == "long_term") return 50;
    
    return 20; // افتراضي
}

//+------------------------------------------------------------------+
//| تحديث Trailing Stop                                             |
//+------------------------------------------------------------------+
void UpdateTrailingStop(ActiveTrade &trade)
{
    double currentPrice = position.PriceCurrent();
    double currentSL = position.StopLoss();
    double distance = trade.trailingDistance * SymbolInfoDouble(trade.symbol, SYMBOL_POINT);
    
    double newSL = 0;
    
    if(position.PositionType() == POSITION_TYPE_BUY)
    {
        newSL = currentPrice - distance;
        if(newSL > currentSL)
        {
            if(trade.Modify(trade.ticket, newSL, position.TakeProfit()))
            {
                PrintFormat("📈 Trailing Stop محدث للصفقة #%d: %.5f", trade.ticket, newSL);
            }
        }
    }
    else
    {
        newSL = currentPrice + distance;
        if(newSL < currentSL)
        {
            if(trade.Modify(trade.ticket, newSL, position.TakeProfit()))
            {
                PrintFormat("📉 Trailing Stop محدث للصفقة #%d: %.5f", trade.ticket, newSL);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| التحقق من وجود صفقة مفتوحة                                      |
//+------------------------------------------------------------------+
bool HasOpenPosition(string symbol, ENUM_TIMEFRAMES timeframe)
{
    string tfStr = TimeframeToString(timeframe);
    
    for(int i = 0; i < ArraySize(activeTrades); i++)
    {
        if(activeTrades[i].symbol == symbol && activeTrades[i].timeframe == tfStr)
        {
            return true;
        }
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| تحويل الفترة الزمنية لنص                                       |
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
        default: return "Unknown";
    }
}

//+------------------------------------------------------------------+
//| معالج أحداث التداول                                            |
//+------------------------------------------------------------------+
void OnTrade()
{
    // تحديث الصفقات النشطة عند حدوث تغيير
    UpdateActiveTrades();
}

//+------------------------------------------------------------------+
//| معالج الأخطاء المخصص                                           |
//+------------------------------------------------------------------+
void LogError(string message, int error = 0)
{
    string fullMessage = message;
    if(error > 0)
    {
        fullMessage += StringFormat(" (Error: %d)", error);
    }
    
    Print(fullMessage);
    
    // يمكن إضافة كتابة لملف سجل خارجي
}