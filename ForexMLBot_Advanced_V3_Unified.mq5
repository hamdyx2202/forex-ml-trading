//+------------------------------------------------------------------+
//|                                 ForexMLBot_Advanced_V3_Unified.mq5|
//|                     🚀 Unified ML Trading with Server Communication|
//|                          📊 Sends 200 candles, receives signals   |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://github.com/forex-ml"
#property version   "3.00"
#property description "🤖 Advanced ML Trading Bot with Unified Server"
#property description "📊 Sends 200 candles to server for prediction"
#property description "🎯 Receives signals with dynamic SL/TP levels"

// Include files
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

// Input parameters
input group "=== Server Settings ==="
input string   ServerURL = "http://69.62.121.53:5000";  // Server URL
input int      ServerTimeout = 5000;                 // Server timeout (ms)
input bool     UseRemoteServer = false;             // Use remote server

input group "=== Trading Settings ==="
input double   LotSize = 0.01;                      // Lot size
input int      MagicNumber = 234567;                // Magic number
input int      MaxPositions = 3;                    // Max open positions
input double   MinConfidence = 0.65;                // Minimum confidence
input int      CandlesToSend = 200;                // Number of candles to send

input group "=== Risk Management ==="
input bool     UseServerSLTP = true;                // Use SL/TP from server
input double   DefaultSL = 50;                      // Default SL (pips)
input double   DefaultTP = 100;                     // Default TP (pips)
input bool     MoveToBreakeven = true;             // Move SL to breakeven
input int      BreakevenPips = 30;                 // Pips to move to breakeven

input group "=== Time Settings ==="
input bool     TradeMonday = true;                  // Trade on Monday
input bool     TradeTuesday = true;                 // Trade on Tuesday
input bool     TradeWednesday = true;               // Trade on Wednesday
input bool     TradeThursday = true;                // Trade on Thursday
input bool     TradeFriday = true;                  // Trade on Friday
input int      StartHour = 0;                       // Start hour
input int      EndHour = 24;                        // End hour

input group "=== Display Settings ==="
input bool     ShowPanel = true;                    // Show info panel
input color    PanelColor = clrBlack;              // Panel background color
input color    TextColor = clrWhite;               // Text color

// Global variables
CTrade trade;
CPositionInfo position;
CAccountInfo account;

// Server communication variables
string serverEndpoint = "/predict";
string tradeResultEndpoint = "/trade_result";
datetime lastServerCheck = 0;
int serverCheckInterval = 60; // seconds

// Performance tracking
struct PerformanceStats {
    int totalTrades;
    int winTrades;
    int lossTrades;
    double totalPips;
    double maxDrawdown;
    double currentStreak;
    datetime lastTradeTime;
};

PerformanceStats stats;

// Signal structure
struct MLSignal {
    string symbol;
    string timeframe;
    string action;
    double confidence;
    double currentPrice;
    double slPrice;
    double tp1Price;
    double tp2Price;
    double slPips;
    double tp1Pips;
    double tp2Pips;
    double riskReward;
    string timestamp;
    bool valid;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize trade object
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    // Reset statistics
    stats.totalTrades = 0;
    stats.winTrades = 0;
    stats.lossTrades = 0;
    stats.totalPips = 0;
    stats.maxDrawdown = 0;
    stats.currentStreak = 0;
    
    // Create panel if enabled
    if (ShowPanel) {
        CreateInfoPanel();
    }
    
    Print("✅ ForexMLBot V3 Unified initialized");
    Print("📊 Server URL: ", ServerURL);
    Print("🎯 Min Confidence: ", MinConfidence);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Clean up panel
    if (ShowPanel) {
        ObjectsDeleteAll(0, "ML_");
    }
    
    // Print final statistics
    PrintStats();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Check if we should trade
    if (!ShouldTrade()) return;
    
    // Monitor open positions
    MonitorPositions();
    
    // Check for new signals every interval
    if (TimeCurrent() - lastServerCheck >= serverCheckInterval) {
        CheckForSignals();
        lastServerCheck = TimeCurrent();
    }
    
    // Update panel
    if (ShowPanel) {
        UpdateInfoPanel();
    }
}

//+------------------------------------------------------------------+
//| Check if we should trade                                        |
//+------------------------------------------------------------------+
bool ShouldTrade() {
    // Check time
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);
    
    if (time.hour < StartHour || time.hour >= EndHour) return false;
    
    // Check day of week
    switch(time.day_of_week) {
        case 1: return TradeMonday;
        case 2: return TradeTuesday;
        case 3: return TradeWednesday;
        case 4: return TradeThursday;
        case 5: return TradeFriday;
        default: return false;
    }
}

//+------------------------------------------------------------------+
//| Check for ML signals from server                                 |
//+------------------------------------------------------------------+
void CheckForSignals() {
    // Check if we can open more positions
    int currentPositions = CountOpenPositions();
    if (currentPositions >= MaxPositions) {
        Print("Max positions reached: ", currentPositions);
        return;
    }
    
    // Get signal from server
    MLSignal signal = GetSignalFromServer();
    
    if (!signal.valid) {
        return;
    }
    
    // Check confidence
    if (signal.confidence < MinConfidence) {
        Print("Signal confidence too low: ", signal.confidence);
        return;
    }
    
    // Execute trade
    ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
//| Get signal from server                                           |
//+------------------------------------------------------------------+
MLSignal GetSignalFromServer() {
    MLSignal signal;
    signal.valid = false;
    
    // Prepare candle data
    string jsonData = PrepareCandles();
    if (jsonData == "") {
        Print("Failed to prepare candle data");
        return signal;
    }
    
    // Send to server
    char postData[], resultData[];
    string resultHeaders;
    
    StringToCharArray(jsonData, postData);
    
    string url = ServerURL + serverEndpoint;
    
    int res = WebRequest(
        "POST",
        url,
        "Content-Type: application/json\r\n",
        ServerTimeout,
        postData,
        resultData,
        resultHeaders
    );
    
    if (res == -1) {
        Print("❌ Server request failed: ", GetLastError());
        return signal;
    }
    
    // Parse response
    string response = CharArrayToString(resultData);
    if (response == "") {
        Print("Empty server response");
        return signal;
    }
    
    // Parse JSON response
    signal = ParseServerResponse(response);
    
    if (signal.valid) {
        Print("✅ Received signal: ", signal.action, " with ", 
              DoubleToString(signal.confidence * 100, 1), "% confidence");
        Print("📍 SL: ", signal.slPips, " pips | TP1: ", signal.tp1Pips, 
              " pips | TP2: ", signal.tp2Pips, " pips");
    }
    
    return signal;
}

//+------------------------------------------------------------------+
//| Prepare candles data for server                                 |
//+------------------------------------------------------------------+
string PrepareCandles() {
    MqlRates rates[];
    int copied = CopyRates(_Symbol, _Period, 0, CandlesToSend, rates);
    
    if (copied != CandlesToSend) {
        Print("Failed to copy ", CandlesToSend, " candles");
        return "";
    }
    
    // Build JSON
    string json = "{";
    json += "\"symbol\":\"" + _Symbol + "\",";
    json += "\"timeframe\":\"" + PeriodToString(_Period) + "\",";
    json += "\"candles\":[";
    
    for (int i = 0; i < copied; i++) {
        if (i > 0) json += ",";
        
        json += "{";
        json += "\"time\":\"" + TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES) + "\",";
        json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume) + ",";
        json += "\"spread\":" + IntegerToString(rates[i].spread);
        json += "}";
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Parse server response                                            |
//+------------------------------------------------------------------+
MLSignal ParseServerResponse(string response) {
    MLSignal signal;
    signal.valid = false;
    
    // Simple JSON parsing (in production, use proper JSON library)
    if (StringFind(response, "\"action\":\"NONE\"") >= 0) {
        return signal;
    }
    
    // Extract values
    signal.symbol = _Symbol;
    signal.timeframe = PeriodToString(_Period);
    
    // Extract action
    int actionPos = StringFind(response, "\"action\":\"");
    if (actionPos >= 0) {
        int start = actionPos + 10;
        int end = StringFind(response, "\"", start);
        signal.action = StringSubstr(response, start, end - start);
    }
    
    // Extract confidence
    int confPos = StringFind(response, "\"confidence\":");
    if (confPos >= 0) {
        int start = confPos + 13;
        int end = StringFind(response, ",", start);
        if (end < 0) end = StringFind(response, "}", start);
        signal.confidence = StringToDouble(StringSubstr(response, start, end - start));
    }
    
    // Extract prices
    signal.currentPrice = ExtractDouble(response, "\"current_price\":");
    signal.slPrice = ExtractDouble(response, "\"sl_price\":");
    signal.tp1Price = ExtractDouble(response, "\"tp1_price\":");
    signal.tp2Price = ExtractDouble(response, "\"tp2_price\":");
    signal.slPips = ExtractDouble(response, "\"sl_pips\":");
    signal.tp1Pips = ExtractDouble(response, "\"tp1_pips\":");
    signal.tp2Pips = ExtractDouble(response, "\"tp2_pips\":");
    signal.riskReward = ExtractDouble(response, "\"risk_reward_ratio\":");
    
    signal.valid = (signal.action == "BUY" || signal.action == "SELL");
    
    return signal;
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                   |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key) {
    int pos = StringFind(json, key);
    if (pos < 0) return 0;
    
    int start = pos + StringLen(key);
    int end = StringFind(json, ",", start);
    if (end < 0) end = StringFind(json, "}", start);
    
    string value = StringSubstr(json, start, end - start);
    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(MLSignal &signal) {
    // Check if position already exists
    if (PositionExists(signal.symbol)) {
        Print("Position already exists for ", signal.symbol);
        return;
    }
    
    double lot = LotSize;
    double sl, tp;
    
    // Use server SL/TP or defaults
    if (UseServerSLTP) {
        sl = signal.slPrice;
        tp = signal.tp1Price; // Use first target
    } else {
        double point = SymbolInfoDouble(signal.symbol, SYMBOL_POINT);
        if (signal.action == "BUY") {
            sl = signal.currentPrice - DefaultSL * point * 10;
            tp = signal.currentPrice + DefaultTP * point * 10;
        } else {
            sl = signal.currentPrice + DefaultSL * point * 10;
            tp = signal.currentPrice - DefaultTP * point * 10;
        }
    }
    
    // Execute trade
    bool result = false;
    string comment = "ML_" + DoubleToString(signal.confidence, 2);
    
    if (signal.action == "BUY") {
        result = trade.Buy(lot, signal.symbol, 0, sl, tp, comment);
    } else if (signal.action == "SELL") {
        result = trade.Sell(lot, signal.symbol, 0, sl, tp, comment);
    }
    
    if (result) {
        stats.totalTrades++;
        stats.lastTradeTime = TimeCurrent();
        
        Print("✅ Trade executed: ", signal.action, " ", signal.symbol, 
              " @ ", signal.currentPrice);
        Print("📍 SL: ", sl, " | TP: ", tp);
        
        // Save trade info for later analysis
        SaveTradeInfo(trade.ResultOrder(), signal);
    } else {
        Print("❌ Trade failed: ", trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Monitor open positions                                           |
//+------------------------------------------------------------------+
void MonitorPositions() {
    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        if (!position.SelectByIndex(i)) continue;
        if (position.Magic() != MagicNumber) continue;
        
        // Check for breakeven
        if (MoveToBreakeven) {
            CheckBreakeven(position);
        }
        
        // Check if position closed
        if (position.Volume() == 0) {
            OnPositionClosed(position);
        }
    }
}

//+------------------------------------------------------------------+
//| Check and move to breakeven                                     |
//+------------------------------------------------------------------+
void CheckBreakeven(CPositionInfo &pos) {
    double currentPrice = pos.PriceCurrent();
    double openPrice = pos.PriceOpen();
    double sl = pos.StopLoss();
    double point = SymbolInfoDouble(pos.Symbol(), SYMBOL_POINT);
    
    bool shouldMove = false;
    double newSL = 0;
    
    if (pos.PositionType() == POSITION_TYPE_BUY) {
        if (currentPrice - openPrice >= BreakevenPips * point * 10 && sl < openPrice) {
            shouldMove = true;
            newSL = openPrice + 2 * point * 10; // Small profit
        }
    } else {
        if (openPrice - currentPrice >= BreakevenPips * point * 10 && sl > openPrice) {
            shouldMove = true;
            newSL = openPrice - 2 * point * 10;
        }
    }
    
    if (shouldMove) {
        trade.PositionModify(pos.Ticket(), newSL, pos.TakeProfit());
        Print("✅ Moved to breakeven: ", pos.Symbol());
    }
}

//+------------------------------------------------------------------+
//| Handle closed position                                           |
//+------------------------------------------------------------------+
void OnPositionClosed(CPositionInfo &pos) {
    double profit = pos.Profit();
    double pips = CalculatePips(pos);
    
    if (profit > 0) {
        stats.winTrades++;
        stats.currentStreak = MathMax(0, stats.currentStreak) + 1;
    } else {
        stats.lossTrades++;
        stats.currentStreak = MathMin(0, stats.currentStreak) - 1;
    }
    
    stats.totalPips += pips;
    
    // Send result to server
    SendTradeResult(pos, pips);
    
    Print("📊 Position closed: ", pos.Symbol(), 
          " | Profit: ", DoubleToString(profit, 2),
          " | Pips: ", DoubleToString(pips, 1));
}

//+------------------------------------------------------------------+
//| Send trade result to server                                      |
//+------------------------------------------------------------------+
void SendTradeResult(CPositionInfo &pos, double pips) {
    string json = "{";
    json += "\"symbol\":\"" + pos.Symbol() + "\",";
    json += "\"result\":\"" + (pos.Profit() > 0 ? "WIN" : "LOSS") + "\",";
    json += "\"entry_price\":" + DoubleToString(pos.PriceOpen(), _Digits) + ",";
    json += "\"exit_price\":" + DoubleToString(pos.PriceCurrent(), _Digits) + ",";
    json += "\"pips\":" + DoubleToString(pips, 1) + ",";
    json += "\"profit\":" + DoubleToString(pos.Profit(), 2) + ",";
    json += "\"action\":\"" + (pos.PositionType() == POSITION_TYPE_BUY ? "BUY" : "SELL") + "\",";
    json += "\"exit_reason\":\"" + GetExitReason(pos) + "\"";
    json += "}";
    
    // Send to server
    char postData[], resultData[];
    string resultHeaders;
    
    StringToCharArray(json, postData);
    
    string url = ServerURL + tradeResultEndpoint;
    
    WebRequest(
        "POST",
        url,
        "Content-Type: application/json\r\n",
        ServerTimeout,
        postData,
        resultData,
        resultHeaders
    );
}

//+------------------------------------------------------------------+
//| Get exit reason                                                  |
//+------------------------------------------------------------------+
string GetExitReason(CPositionInfo &pos) {
    if (MathAbs(pos.PriceCurrent() - pos.StopLoss()) < _Point) {
        return "SL_HIT";
    } else if (MathAbs(pos.PriceCurrent() - pos.TakeProfit()) < _Point) {
        return "TP_HIT";
    } else {
        return "MANUAL_CLOSE";
    }
}

//+------------------------------------------------------------------+
//| Calculate pips                                                   |
//+------------------------------------------------------------------+
double CalculatePips(CPositionInfo &pos) {
    double point = SymbolInfoDouble(pos.Symbol(), SYMBOL_POINT);
    double multiplier = 1;
    
    if (StringFind(pos.Symbol(), "JPY") >= 0) {
        multiplier = 100;
    } else {
        multiplier = 10000;
    }
    
    double pips = 0;
    if (pos.PositionType() == POSITION_TYPE_BUY) {
        pips = (pos.PriceCurrent() - pos.PriceOpen()) * multiplier;
    } else {
        pips = (pos.PriceOpen() - pos.PriceCurrent()) * multiplier;
    }
    
    return pips;
}

//+------------------------------------------------------------------+
//| Save trade info                                                  |
//+------------------------------------------------------------------+
void SaveTradeInfo(ulong ticket, MLSignal &signal) {
    GlobalVariableSet("ML_" + IntegerToString(ticket) + "_conf", signal.confidence);
    GlobalVariableSet("ML_" + IntegerToString(ticket) + "_rr", signal.riskReward);
}

//+------------------------------------------------------------------+
//| Count open positions                                             |
//+------------------------------------------------------------------+
int CountOpenPositions() {
    int count = 0;
    for (int i = 0; i < PositionsTotal(); i++) {
        if (position.SelectByIndex(i) && position.Magic() == MagicNumber) {
            count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Check if position exists                                         |
//+------------------------------------------------------------------+
bool PositionExists(string symbol) {
    for (int i = 0; i < PositionsTotal(); i++) {
        if (position.SelectByIndex(i) && 
            position.Magic() == MagicNumber &&
            position.Symbol() == symbol) {
            return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Convert period to string                                         |
//+------------------------------------------------------------------+
string PeriodToString(ENUM_TIMEFRAMES period) {
    switch(period) {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        case PERIOD_W1:  return "W1";
        case PERIOD_MN1: return "MN1";
        default: return "M15";
    }
}

//+------------------------------------------------------------------+
//| Create info panel                                                |
//+------------------------------------------------------------------+
void CreateInfoPanel() {
    int x = 10, y = 30;
    int width = 250, height = 200;
    
    // Background
    ObjectCreate(0, "ML_Panel", OBJ_RECTANGLE_LABEL, 0, 0, 0);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_YDISTANCE, y);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_XSIZE, width);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_YSIZE, height);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_BGCOLOR, PanelColor);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(0, "ML_Panel", OBJPROP_CORNER, CORNER_LEFT_UPPER);
    
    // Title
    CreateLabel("ML_Title", x + 10, y + 10, "🤖 Forex ML Bot V3", 10, TextColor);
    
    // Stats labels
    CreateLabel("ML_Trades", x + 10, y + 40, "Trades: 0", 9, TextColor);
    CreateLabel("ML_WinRate", x + 10, y + 60, "Win Rate: 0%", 9, TextColor);
    CreateLabel("ML_Pips", x + 10, y + 80, "Total Pips: 0", 9, TextColor);
    CreateLabel("ML_Streak", x + 10, y + 100, "Streak: 0", 9, TextColor);
    CreateLabel("ML_Positions", x + 10, y + 120, "Positions: 0/3", 9, TextColor);
    CreateLabel("ML_Server", x + 10, y + 140, "Server: ✅", 9, TextColor);
    CreateLabel("ML_LastSignal", x + 10, y + 160, "Last Signal: --", 8, TextColor);
}

//+------------------------------------------------------------------+
//| Create label                                                     |
//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text, int size, color clr) {
    ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetString(0, name, OBJPROP_FONT, "Arial");
}

//+------------------------------------------------------------------+
//| Update info panel                                                |
//+------------------------------------------------------------------+
void UpdateInfoPanel() {
    double winRate = stats.totalTrades > 0 ? 
        (double)stats.winTrades / stats.totalTrades * 100 : 0;
    
    ObjectSetString(0, "ML_Trades", OBJPROP_TEXT, 
        "Trades: " + IntegerToString(stats.totalTrades));
    ObjectSetString(0, "ML_WinRate", OBJPROP_TEXT, 
        "Win Rate: " + DoubleToString(winRate, 1) + "%");
    ObjectSetString(0, "ML_Pips", OBJPROP_TEXT, 
        "Total Pips: " + DoubleToString(stats.totalPips, 1));
    ObjectSetString(0, "ML_Streak", OBJPROP_TEXT, 
        "Streak: " + IntegerToString((int)stats.currentStreak));
    ObjectSetString(0, "ML_Positions", OBJPROP_TEXT, 
        "Positions: " + IntegerToString(CountOpenPositions()) + "/" + 
        IntegerToString(MaxPositions));
    
    // Update last signal time
    if (stats.lastTradeTime > 0) {
        int secondsAgo = (int)(TimeCurrent() - stats.lastTradeTime);
        string timeStr = "";
        if (secondsAgo < 60) {
            timeStr = IntegerToString(secondsAgo) + "s ago";
        } else if (secondsAgo < 3600) {
            timeStr = IntegerToString(secondsAgo / 60) + "m ago";
        } else {
            timeStr = IntegerToString(secondsAgo / 3600) + "h ago";
        }
        ObjectSetString(0, "ML_LastSignal", OBJPROP_TEXT, "Last Signal: " + timeStr);
    }
}

//+------------------------------------------------------------------+
//| Print statistics                                                 |
//+------------------------------------------------------------------+
void PrintStats() {
    Print("========================================");
    Print("📊 Final Statistics:");
    Print("Total Trades: ", stats.totalTrades);
    Print("Win Trades: ", stats.winTrades);
    Print("Loss Trades: ", stats.lossTrades);
    Print("Win Rate: ", stats.totalTrades > 0 ? 
        DoubleToString((double)stats.winTrades / stats.totalTrades * 100, 1) + "%" : "0%");
    Print("Total Pips: ", DoubleToString(stats.totalPips, 1));
    Print("========================================");
}