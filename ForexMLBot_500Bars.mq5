//+------------------------------------------------------------------+
//|                                          ForexMLBot_500Bars.mq5  |
//|                     Forex ML Trading Bot - 500 Bars Version      |
//|                     يرسل 500 شمعة لتحسين دقة المؤشرات            |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://github.com/your-repo"
#property version   "1.05"
#property strict

// Input parameters
input string   ServerUrl = "http://69.62.121.53:5000";  // عنوان الخادم
input int      BarsToSend = 500;                       // عدد الشموع المرسلة
input int      UpdateIntervalSeconds = 5;               // فترة التحديث (ثواني)
input double   LotSize = 0.01;                         // حجم اللوت
input int      MagicNumber = 12345;                    // الرقم السحري
input int      MaxSpread = 30;                         // أقصى سبريد
input bool     EnableTrading = true;                   // تفعيل التداول
input bool     ShowDashboard = true;                   // عرض لوحة المعلومات

// Global variables
datetime lastUpdateTime = 0;
string symbols[] = {"EURUSDm", "GBPUSDm", "USDJPYm", "USDCHFm", 
                   "AUDUSDm", "USDCADm", "NZDUSDm", "XAUUSDm"};
string currentSignals[];
double currentConfidences[];

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 ForexMLBot 500 Bars initialized");
    Print("📊 Will send ", BarsToSend, " bars for better indicator accuracy");
    Print("🔗 Server URL: ", ServerUrl);
    
    // Initialize arrays
    ArrayResize(currentSignals, ArraySize(symbols));
    ArrayResize(currentConfidences, ArraySize(symbols));
    
    // Initialize signals
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        currentSignals[i] = "WAIT";
        currentConfidences[i] = 0.0;
    }
    
    // Create dashboard
    if(ShowDashboard)
    {
        CreateDashboard();
    }
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Clean up dashboard
    if(ShowDashboard)
    {
        ObjectsDeleteAll(0, "ML_");
    }
    
    Print("👋 ForexMLBot deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if it's time to update
    if(TimeCurrent() - lastUpdateTime < UpdateIntervalSeconds)
        return;
    
    lastUpdateTime = TimeCurrent();
    
    // Process each symbol
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        ProcessSymbol(symbols[i], i);
    }
    
    // Update dashboard
    if(ShowDashboard)
    {
        UpdateDashboard();
    }
}

//+------------------------------------------------------------------+
//| Process individual symbol                                         |
//+------------------------------------------------------------------+
void ProcessSymbol(string symbol, int index)
{
    // Check if symbol exists
    if(!SymbolSelect(symbol, true))
    {
        Print("❌ Symbol not found: ", symbol);
        return;
    }
    
    // Get current price
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double spread = (ask - bid) / SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    // Check spread
    if(spread > MaxSpread)
    {
        currentSignals[index] = "HIGH_SPREAD";
        return;
    }
    
    // Get signal from server
    string signal = GetMLSignal(symbol);
    
    if(signal != "")
    {
        currentSignals[index] = signal;
        
        // Execute trade if enabled
        if(EnableTrading && (signal == "BUY" || signal == "SELL"))
        {
            ExecuteTrade(symbol, signal);
        }
    }
}

//+------------------------------------------------------------------+
//| Get ML signal from server                                         |
//+------------------------------------------------------------------+
string GetMLSignal(string symbol)
{
    // Prepare the data
    string jsonData = CreateFullDataJSON(symbol);
    
    if(jsonData == "")
    {
        Print("❌ Failed to create JSON data for ", symbol);
        return "";
    }
    
    // Prepare request
    string url = ServerUrl + "/get_signal";
    string headers = "Content-Type: application/json\r\n";
    char post_data[];
    char result_data[];
    string result_headers;
    
    StringToCharArray(jsonData, post_data);
    
    // Send request
    ResetLastError();
    int timeout = 5000; // 5 seconds timeout
    
    int res = WebRequest("POST", url, headers, timeout, post_data, result_data, result_headers);
    
    if(res == -1)
    {
        int error = GetLastError();
        if(error != 0)
        {
            Print("❌ WebRequest error for ", symbol, ": ", error);
        }
        return "";
    }
    
    // Parse response
    string response = CharArrayToString(result_data);
    
    // Simple JSON parsing
    string action = "";
    double confidence = 0;
    
    int actionPos = StringFind(response, "\"action\":");
    if(actionPos != -1)
    {
        int start = StringFind(response, "\"", actionPos + 9) + 1;
        int end = StringFind(response, "\"", start);
        action = StringSubstr(response, start, end - start);
    }
    
    int confPos = StringFind(response, "\"confidence\":");
    if(confPos != -1)
    {
        int start = confPos + 13;
        int end = StringFind(response, ",", start);
        if(end == -1) end = StringFind(response, "}", start);
        string confStr = StringSubstr(response, start, end - start);
        confidence = StringToDouble(confStr);
    }
    
    // Store confidence
    int idx = GetSymbolIndex(symbol);
    if(idx != -1)
    {
        currentConfidences[idx] = confidence;
    }
    
    Print("📊 ", symbol, " Signal: ", action, " (", 
          DoubleToString(confidence * 100, 1), "% confidence)");
    
    return action;
}

//+------------------------------------------------------------------+
//| Create full data JSON with 500 bars                              |
//+------------------------------------------------------------------+
string CreateFullDataJSON(string symbol)
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    
    // Get historical data - 500 bars
    int copied = CopyRates(symbol, PERIOD_M5, 0, BarsToSend, rates);
    
    if(copied < 100) // Minimum 100 bars required
    {
        Print("❌ Not enough historical data for ", symbol, 
              " (got ", copied, " bars)");
        return "";
    }
    
    // Build JSON request with multiple timeframes
    string json = "{";
    json += "\"requests\":[";
    
    // Send data for multiple timeframes
    ENUM_TIMEFRAMES timeframes[] = {PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4};
    string tf_names[] = {"M5", "M15", "H1", "H4"};
    
    for(int tf = 0; tf < ArraySize(timeframes); tf++)
    {
        if(tf > 0) json += ",";
        
        // Get data for this timeframe
        MqlRates tf_rates[];
        ArraySetAsSeries(tf_rates, true);
        int tf_copied = CopyRates(symbol, timeframes[tf], 0, BarsToSend, tf_rates);
        
        if(tf_copied < 50) continue; // Skip if not enough data
        
        json += "{";
        json += "\"symbol\":\"" + symbol + "\",";
        json += "\"timeframe\":\"" + tf_names[tf] + "\",";
        json += "\"data\":[";
        
        // Add bars (newest first)
        for(int i = 0; i < MathMin(tf_copied, BarsToSend); i++)
        {
            if(i > 0) json += ",";
            
            json += "{";
            json += "\"time\":" + IntegerToString(tf_rates[i].time) + ",";
            json += "\"open\":" + DoubleToString(tf_rates[i].open, 5) + ",";
            json += "\"high\":" + DoubleToString(tf_rates[i].high, 5) + ",";
            json += "\"low\":" + DoubleToString(tf_rates[i].low, 5) + ",";
            json += "\"close\":" + DoubleToString(tf_rates[i].close, 5) + ",";
            json += "\"volume\":" + IntegerToString(tf_rates[i].tick_volume);
            json += "}";
        }
        
        json += "]"; // end data array
        json += "}"; // end timeframe object
    }
    
    json += "]"; // end requests array
    json += "}"; // end main object
    
    return json;
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(string symbol, string signal)
{
    // Check if already have position
    if(HasPosition(symbol))
    {
        return;
    }
    
    MqlTradeRequest request;
    MqlTradeResult result;
    
    ZeroMemory(request);
    ZeroMemory(result);
    
    // Prepare trade request
    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = LotSize;
    request.magic = MagicNumber;
    request.deviation = 10;
    
    if(signal == "BUY")
    {
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
        request.sl = request.price - 200 * SymbolInfoDouble(symbol, SYMBOL_POINT);
        request.tp = request.price + 400 * SymbolInfoDouble(symbol, SYMBOL_POINT);
        request.comment = "ML_BUY";
    }
    else if(signal == "SELL")
    {
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
        request.sl = request.price + 200 * SymbolInfoDouble(symbol, SYMBOL_POINT);
        request.tp = request.price - 400 * SymbolInfoDouble(symbol, SYMBOL_POINT);
        request.comment = "ML_SELL";
    }
    else
    {
        return;
    }
    
    // Send order
    if(OrderSend(request, result))
    {
        Print("✅ Order placed for ", symbol, ": ", signal, 
              " at ", request.price);
    }
    else
    {
        Print("❌ Order failed for ", symbol, ": ", result.comment);
    }
}

//+------------------------------------------------------------------+
//| Check if position exists for symbol                              |
//+------------------------------------------------------------------+
bool HasPosition(string symbol)
{
    for(int i = 0; i < PositionsTotal(); i++)
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
//| Get symbol index in arrays                                       |
//+------------------------------------------------------------------+
int GetSymbolIndex(string symbol)
{
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        if(symbols[i] == symbol)
            return i;
    }
    return -1;
}

//+------------------------------------------------------------------+
//| Create dashboard                                                  |
//+------------------------------------------------------------------+
void CreateDashboard()
{
    int y = 20;
    
    // Title
    ObjectCreate(0, "ML_Title", OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, "ML_Title", OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, "ML_Title", OBJPROP_YDISTANCE, y);
    ObjectSetString(0, "ML_Title", OBJPROP_TEXT, "🤖 Forex ML Bot - 500 Bars");
    ObjectSetInteger(0, "ML_Title", OBJPROP_FONTSIZE, 12);
    ObjectSetInteger(0, "ML_Title", OBJPROP_COLOR, clrGold);
    
    y += 30;
    
    // Create labels for each symbol
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        string name = "ML_Symbol_" + IntegerToString(i);
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y + i * 20);
        ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 10);
    }
}

//+------------------------------------------------------------------+
//| Update dashboard                                                  |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
    for(int i = 0; i < ArraySize(symbols); i++)
    {
        string name = "ML_Symbol_" + IntegerToString(i);
        string text = symbols[i] + ": " + currentSignals[i];
        
        if(currentConfidences[i] > 0)
        {
            text += " (" + DoubleToString(currentConfidences[i] * 100, 1) + "%)";
        }
        
        ObjectSetString(0, name, OBJPROP_TEXT, text);
        
        // Set color based on signal
        color textColor = clrSilver;
        if(currentSignals[i] == "BUY") textColor = clrLime;
        else if(currentSignals[i] == "SELL") textColor = clrRed;
        else if(currentSignals[i] == "NO_TRADE") textColor = clrGray;
        
        ObjectSetInteger(0, name, OBJPROP_COLOR, textColor);
    }
    
    ChartRedraw();
}

//+------------------------------------------------------------------+