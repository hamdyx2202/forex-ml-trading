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
input bool     InpUseLongTerm = false;       // استخدام Long Term (24 ساعات)

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
    double sl;
    double tp1, tp2, tp3;
    double breakevenLevel;
    double trailingDistance;
    int currentTP;
    bool isBreakeven;
    bool isTrailing;
};

ActiveTrade activeTrades[];

//--- متغيرات التحكم
datetime lastCheck = 0;
int checkInterval = 30; // ثانية

//+------------------------------------------------------------------+
//| دالة البداية                                                      |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- إعداد المتداول
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());
    trade.SetDeviationInPoints(30);
    
    //--- فحص الاتصال
    if(!CheckConnection())
    {
        Print("❌ فشل الاتصال بالسيرفر");
        return(INIT_FAILED);
    }
    
    Print("✅ تم تشغيل الروبوت بنجاح");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| دالة التدمير                                                     |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("⏹️ تم إيقاف الروبوت");
}

//+------------------------------------------------------------------+
//| دالة التيك                                                       |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- التحقق من الوقت
    if(TimeCurrent() - lastCheck < checkInterval)
        return;
    
    lastCheck = TimeCurrent();
    
    //--- إدارة الصفقات الحالية
    ManageOpenTrades();
    
    //--- البحث عن فرص جديدة
    if(CanOpenNewTrade())
    {
        CheckAllPairs();
    }
}

//+------------------------------------------------------------------+
//| فحص الاتصال بالسيرفر                                            |
//+------------------------------------------------------------------+
bool CheckConnection()
{
    string headers = "Content-Type: application/json\r\n";
    string test_data = "{\"test\": true}";
    char post_data[];
    char result[];
    string result_headers;
    
    StringToCharArray(test_data, post_data);
    
    int res = WebRequest("POST", InpServerURL + "/health", headers, 5000, 
                        post_data, result, result_headers);
    
    return (res == 200);
}

//+------------------------------------------------------------------+
//| فحص إمكانية فتح صفقات جديدة                                    |
//+------------------------------------------------------------------+
bool CanOpenNewTrade()
{
    int totalTrades = ArraySize(activeTrades);
    
    if(totalTrades >= InpMaxTrades)
    {
        PrintFormat("⚠️ وصلت للحد الأقصى من الصفقات: %d", InpMaxTrades);
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| فحص جميع الأزواج                                                |
//+------------------------------------------------------------------+
void CheckAllPairs()
{
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        for(int j = 0; j < ArraySize(timeframes); j++)
        {
            //--- تحقق من الاستراتيجية المناسبة
            if(!IsStrategyEnabled(timeframes[j]))
                continue;
            
            //--- تحقق من وجود صفقة مفتوحة
            if(HasOpenPosition(symbols[i], timeframes[j]))
                continue;
            
            //--- احصل على التنبؤ
            CheckPair(symbols[i], timeframes[j]);
        }
    }
}

//+------------------------------------------------------------------+
//| فحص زوج واحد                                                    |
//+------------------------------------------------------------------+
void CheckPair(string symbol, ENUM_TIMEFRAMES timeframe)
{
    //--- جمع البيانات
    string jsonData = PrepareData(symbol, timeframe);
    if(jsonData == "")
        return;
    
    //--- إرسال للسيرفر
    string response = SendPredictionRequest(jsonData);
    if(response == "")
        return;
    
    //--- تحليل الرد
    ProcessPrediction(response, symbol, timeframe);
}

//+------------------------------------------------------------------+
//| تحضير البيانات للإرسال                                         |
//+------------------------------------------------------------------+
string PrepareData(string symbol, ENUM_TIMEFRAMES timeframe)
{
    MqlRates rates[];
    int copied = CopyRates(symbol, timeframe, 0, InpCandlesHistory, rates);
    
    if(copied < InpCandlesHistory)
    {
        PrintFormat("⚠️ بيانات غير كافية لـ %s %s", symbol, EnumToString(timeframe));
        return "";
    }
    
    //--- بناء JSON
    string json = "{";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timeframe\":\"" + TimeframeToString(timeframe) + "\",";
    json += "\"candles\":[";
    
    for(int i = 0; i < copied; i++)
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
    json += "\"balance\":" + DoubleToString(account.Balance(), 2) + ",";
    json += "\"risk_percent\":" + DoubleToString(InpRiskPercent, 2);
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| إرسال طلب التنبؤ                                                |
//+------------------------------------------------------------------+
string SendPredictionRequest(string jsonData)
{
    string headers = "Content-Type: application/json\r\n";
    char post_data[];
    char result[];
    string result_headers;
    
    StringToCharArray(jsonData, post_data);
    
    int res = WebRequest("POST", InpServerURL, headers, 10000, 
                        post_data, result, result_headers);
    
    if(res != 200)
    {
        PrintFormat("❌ خطأ في الاتصال: %d", res);
        return "";
    }
    
    return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| معالجة التنبؤ                                                    |
//+------------------------------------------------------------------+
void ProcessPrediction(string response, string symbol, ENUM_TIMEFRAMES timeframe)
{
    //--- تحليل JSON (مبسط)
    double confidence = ExtractDouble(response, "confidence");
    string signal = ExtractString(response, "signal");
    double sl = ExtractDouble(response, "sl");
    double tp1 = ExtractDouble(response, "tp1");
    double tp2 = ExtractDouble(response, "tp2");
    double tp3 = ExtractDouble(response, "tp3");
    double lot_size = ExtractDouble(response, "lot_size");
    string strategy = ExtractString(response, "strategy");
    
    //--- التحقق من الثقة
    if(confidence < InpMinConfidence)
    {
        PrintFormat("⚠️ ثقة منخفضة: %.2f%% لـ %s %s", 
                   confidence * 100, symbol, EnumToString(timeframe));
        return;
    }
    
    //--- فتح الصفقة
    if(signal == "BUY" || signal == "SELL")
    {
        OpenTrade(symbol, timeframe, signal, sl, tp1, tp2, tp3, 
                 lot_size, confidence, strategy);
    }
}

//+------------------------------------------------------------------+
//| فتح صفقة جديدة                                                   |
//+------------------------------------------------------------------+
void OpenTrade(string symbol, ENUM_TIMEFRAMES timeframe, string signal,
               double sl, double tp1, double tp2, double tp3, 
               double lot_size, double confidence, string strategy)
{
    ENUM_ORDER_TYPE orderType = (signal == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    
    //--- فتح الصفقة
    if(trade.PositionOpen(symbol, orderType, lot_size, 0, sl, tp1, 
                         "ML Bot | " + strategy + " | " + DoubleToString(confidence * 100, 1) + "%"))
    {
        //--- إضافة للصفقات النشطة
        ActiveTrade newTrade;
        newTrade.ticket = trade.ResultOrder();
        newTrade.symbol = symbol;
        newTrade.timeframe = TimeframeToString(timeframe);
        newTrade.sl = sl;
        newTrade.tp1 = tp1;
        newTrade.tp2 = tp2;
        newTrade.tp3 = tp3;
        newTrade.breakevenLevel = 0;
        newTrade.trailingDistance = GetTrailingDistance(strategy);
        newTrade.currentTP = 1;
        newTrade.isBreakeven = false;
        newTrade.isTrailing = false;
        
        int size = ArraySize(activeTrades);
        ArrayResize(activeTrades, size + 1);
        activeTrades[size] = newTrade;
        
        PrintFormat("✅ فتح صفقة %s على %s %s | الثقة: %.1f%% | الاستراتيجية: %s",
                   signal, symbol, TimeframeToString(timeframe), 
                   confidence * 100, strategy);
    }
    else
    {
        PrintFormat("❌ فشل فتح الصفقة: %s", trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| إدارة الصفقات المفتوحة                                          |
//+------------------------------------------------------------------+
void ManageOpenTrades()
{
    for(int i = ArraySize(activeTrades) - 1; i >= 0; i--)
    {
        if(!position.SelectByTicket(activeTrades[i].ticket))
        {
            //--- الصفقة مغلقة
            ArrayRemove(activeTrades, i, 1);
            continue;
        }
        
        //--- تحديث معلومات الصفقة
        double currentPrice = position.PriceCurrent();
        double entryPrice = position.PriceOpen();
        double currentProfit = position.Profit();
        
        //--- نقل للتعادل
        if(InpUseMoveToBreakeven && !activeTrades[i].isBreakeven)
        {
            CheckBreakeven(activeTrades[i], currentPrice, entryPrice);
        }
        
        //--- تحديث TP
        CheckTakeProfitLevels(activeTrades[i], currentPrice);
        
        //--- Trailing Stop
        if(InpUseTrailingStop && activeTrades[i].isBreakeven)
        {
            UpdateTrailingStop(activeTrades[i]);
        }
    }
}

//+------------------------------------------------------------------+
//| فحص نقل للتعادل                                                |
//+------------------------------------------------------------------+
void CheckBreakeven(ActiveTrade &activeTrade, double currentPrice, double entryPrice)
{
    double distance = MathAbs(currentPrice - entryPrice);
    double tp1Distance = MathAbs(activeTrade.tp1 - entryPrice);
    
    //--- إذا وصل السعر لـ 50% من TP1
    if(distance >= tp1Distance * 0.5)
    {
        MoveToBreakeven(activeTrade);
        activeTrade.isBreakeven = true;
    }
}

//+------------------------------------------------------------------+
//| نقل SL للتعادل                                                 |
//+------------------------------------------------------------------+
void MoveToBreakeven(ActiveTrade &activeTrade)
{
    double entryPrice = position.PriceOpen();
    double spread = SymbolInfoInteger(activeTrade.symbol, SYMBOL_SPREAD) * 
                    SymbolInfoDouble(activeTrade.symbol, SYMBOL_POINT);
    
    double newSL = entryPrice;
    if(position.PositionType() == POSITION_TYPE_BUY)
        newSL += spread;
    else
        newSL -= spread;
    
    if(trade.PositionModify(activeTrade.ticket, newSL, position.TakeProfit()))
    {
        PrintFormat("✅ نقل SL للتعادل للصفقة #%d", activeTrade.ticket);
        activeTrade.breakevenLevel = newSL;
    }
}

//+------------------------------------------------------------------+
//| فحص مستويات Take Profit                                         |
//+------------------------------------------------------------------+
void CheckTakeProfitLevels(ActiveTrade &activeTrade, double currentPrice)
{
    double entryPrice = position.PriceOpen();
    bool isBuy = (position.PositionType() == POSITION_TYPE_BUY);
    
    //--- فحص TP2
    if(activeTrade.currentTP == 1)
    {
        if((isBuy && currentPrice >= activeTrade.tp1) ||
           (!isBuy && currentPrice <= activeTrade.tp1))
        {
            UpdateTakeProfit(activeTrade, 2);
        }
    }
    //--- فحص TP3
    else if(activeTrade.currentTP == 2)
    {
        if((isBuy && currentPrice >= activeTrade.tp2) ||
           (!isBuy && currentPrice <= activeTrade.tp2))
        {
            UpdateTakeProfit(activeTrade, 3);
        }
    }
}

//+------------------------------------------------------------------+
//| تحديث Take Profit                                               |
//+------------------------------------------------------------------+
void UpdateTakeProfit(ActiveTrade &activeTrade, int tpLevel)
{
    double newTP = 0;
    
    switch(tpLevel)
    {
        case 2: newTP = activeTrade.tp2; break;
        case 3: newTP = activeTrade.tp3; break;
        default: return;
    }
    
    if(trade.PositionModify(activeTrade.ticket, position.StopLoss(), newTP))
    {
        PrintFormat("✅ تحديث TP%d للصفقة #%d", tpLevel, activeTrade.ticket);
        activeTrade.currentTP = tpLevel;
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
void UpdateTrailingStop(ActiveTrade &activeTrade)
{
    double currentPrice = position.PriceCurrent();
    double currentSL = position.StopLoss();
    double distance = activeTrade.trailingDistance * SymbolInfoDouble(activeTrade.symbol, SYMBOL_POINT);
    
    double newSL = 0;
    
    if(position.PositionType() == POSITION_TYPE_BUY)
    {
        newSL = currentPrice - distance;
        if(newSL > currentSL)
        {
            if(trade.PositionModify(activeTrade.ticket, newSL, position.TakeProfit()))
            {
                PrintFormat("📈 Trailing Stop محدث للصفقة #%d: %.5f", activeTrade.ticket, newSL);
            }
        }
    }
    else
    {
        newSL = currentPrice + distance;
        if(newSL < currentSL)
        {
            if(trade.PositionModify(activeTrade.ticket, newSL, position.TakeProfit()))
            {
                PrintFormat("📉 Trailing Stop محدث للصفقة #%d: %.5f", activeTrade.ticket, newSL);
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
            return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| التحقق من تفعيل الاستراتيجية                                   |
//+------------------------------------------------------------------+
bool IsStrategyEnabled(ENUM_TIMEFRAMES timeframe)
{
    switch(timeframe)
    {
        case PERIOD_M1:
        case PERIOD_M5:
            return InpUseUltraShort;
        
        case PERIOD_M15:
        case PERIOD_M30:
            return InpUseScalping;
        
        case PERIOD_H1:
            return InpUseShortTerm;
        
        case PERIOD_H4:
            return InpUseMediumTerm;
        
        default:
            return InpUseLongTerm;
    }
}

//+------------------------------------------------------------------+
//| تحويل الإطار الزمني لنص                                        |
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
//| استخراج قيمة double من JSON                                     |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
{
    int start = StringFind(json, "\"" + key + "\":");
    if(start == -1) return 0;
    
    start = StringFind(json, ":", start) + 1;
    int end = StringFind(json, ",", start);
    if(end == -1) end = StringFind(json, "}", start);
    
    string value = StringSubstr(json, start, end - start);
    StringTrimLeft(value);
    StringTrimRight(value);
    
    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| استخراج نص من JSON                                             |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
{
    int start = StringFind(json, "\"" + key + "\":");
    if(start == -1) return "";
    
    start = StringFind(json, "\"", start + StringLen(key) + 3) + 1;
    int end = StringFind(json, "\"", start);
    
    return StringSubstr(json, start, end - start);
}