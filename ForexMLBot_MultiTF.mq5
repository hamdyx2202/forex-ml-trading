//+------------------------------------------------------------------+
//|                                          ForexMLBot_MultiTF.mq5  |
//|                  Multi-Pair + Multi-Timeframe Version            |
//|                        Uses ALL 32 Models (100%)                 |
//+------------------------------------------------------------------+
#property copyright "Forex ML Trading System"
#property link      "https://github.com/hamdysoltan/forex-ml-trading"
#property version   "4.0"

// إعدادات الإدخال
input string   ServerUrl = "http://localhost:5000";     // عنوان الخادم
input double   RiskPerTrade = 0.01;                     // المخاطرة لكل صفقة (1%)
input int      MagicNumber = 123456;                    // الرقم السحري
input int      BarsToSend = 100;                        // عدد الشموع المرسلة
input int      UpdateIntervalSeconds = 60;               // فترة التحديث (ثانية)
input bool     EnableTrading = true;                    // تفعيل التداول
input bool     ShowDashboard = true;                    // عرض لوحة المعلومات
input string   PairsToTrade = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,EURJPY,GBPJPY,XAUUSD"; // الأزواج للتداول
input bool     UseAllTimeframes = true;                 // استخدام جميع الأطر الزمنية
input double   MinCombinedConfidence = 0.75;            // الحد الأدنى للثقة المجمعة

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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🚀 ForexML Multi-Timeframe Bot Started");
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
      
      string currentSymbol = _Symbol;
      if(StringFind(currentSymbol, "m") > 0 && StringFind(pairs[i], "m") < 0)
         pairs[i] += "m";
      
      lastUpdateTime[i] = 0;
      lastSignals[i] = "NONE";
      lastConfidence[i] = 0;
      combinedConfidence[i] = 0;
      
      Print("📊 Monitoring: ", pairs[i]);
   }
   
   if(!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED))
   {
      Alert("❌ Please allow DLL imports in Terminal settings!");
      return(INIT_FAILED);
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
   
   if(ShowDashboard)
      UpdateDashboard();
}

//+------------------------------------------------------------------+
//| معالجة زوج واحد بجميع الأطر الزمنية                              |
//+------------------------------------------------------------------+
void ProcessPair(int pairIndex)
{
   string symbol = pairs[pairIndex];
   
   if(!SymbolSelect(symbol, true))
   {
      Print("⚠️ Symbol not available: ", symbol);
      return;
   }
   
   if(UseAllTimeframes)
   {
      // جمع التنبؤات من جميع الأطر الزمنية
      double totalConfidence = 0;
      int buySignals = 0;
      int sellSignals = 0;
      int validSignals = 0;
      
      for(int tf = 0; tf < ArraySize(timeframes); tf++)
      {
         string signal;
         double confidence;
         
         if(GetSignalForTimeframe(symbol, timeframes[tf], timeframeNames[tf], signal, confidence))
         {
            if(confidence > 0.5)
            {
               totalConfidence += confidence;
               validSignals++;
               
               if(signal == "BUY" || signal == "STRONG_BUY")
                  buySignals++;
               else if(signal == "SELL" || signal == "STRONG_SELL")
                  sellSignals++;
            }
         }
      }
      
      // حساب الإشارة المجمعة
      if(validSignals > 0)
      {
         double avgConfidence = totalConfidence / validSignals;
         combinedConfidence[pairIndex] = avgConfidence;
         
         // تحديد الإشارة النهائية بناءً على الأغلبية
         if(buySignals > sellSignals && buySignals >= 2)
         {
            lastSignals[pairIndex] = avgConfidence > 0.85 ? "STRONG_BUY" : "BUY";
            lastConfidence[pairIndex] = avgConfidence;
         }
         else if(sellSignals > buySignals && sellSignals >= 2)
         {
            lastSignals[pairIndex] = avgConfidence > 0.85 ? "STRONG_SELL" : "SELL";
            lastConfidence[pairIndex] = avgConfidence;
         }
         else
         {
            lastSignals[pairIndex] = "NEUTRAL";
            lastConfidence[pairIndex] = avgConfidence;
         }
         
         Print("📊 ", symbol, " Combined Signal: ", lastSignals[pairIndex], 
               " | Confidence: ", DoubleToString(avgConfidence * 100, 1), "%",
               " | Buy votes: ", buySignals, " | Sell votes: ", sellSignals);
         
         // التداول إذا كانت الثقة عالية
         if(EnableTrading && avgConfidence >= MinCombinedConfidence && !HasOpenPosition(symbol))
         {
            if(lastSignals[pairIndex] == "BUY" || lastSignals[pairIndex] == "STRONG_BUY")
               OpenBuyPosition(symbol, avgConfidence);
            else if(lastSignals[pairIndex] == "SELL" || lastSignals[pairIndex] == "STRONG_SELL")
               OpenSellPosition(symbol, avgConfidence);
         }
      }
   }
   else
   {
      // M5 فقط (النظام القديم)
      string signal;
      double confidence;
      
      if(GetSignalForTimeframe(symbol, PERIOD_M5, "M5", signal, confidence))
      {
         lastSignals[pairIndex] = signal;
         lastConfidence[pairIndex] = confidence;
         
         if(EnableTrading && confidence >= 0.7 && !HasOpenPosition(symbol))
         {
            if(signal == "BUY" || signal == "STRONG_BUY")
               OpenBuyPosition(symbol, confidence);
            else if(signal == "SELL" || signal == "STRONG_SELL")
               OpenSellPosition(symbol, confidence);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| الحصول على إشارة لإطار زمني محدد                                |
//+------------------------------------------------------------------+
bool GetSignalForTimeframe(string symbol, ENUM_TIMEFRAMES timeframe, string tfName, string &signal, double &confidence)
{
   // جلب البيانات
   MqlRates rates[];
   int copied = CopyRates(symbol, timeframe, 0, BarsToSend, rates);
   
   if(copied < BarsToSend)
   {
      Print("⚠️ Not enough bars for ", symbol, " ", tfName);
      return false;
   }
   
   // إنشاء JSON مع تحديد الإطار الزمني
   string json = CreateTimeframeJSON(symbol, tfName, rates);
   
   // إرسال للخادم
   string response = SendToServer(json);
   
   if(response != "")
   {
      signal = ExtractValue(response, "signal");
      confidence = StringToDouble(ExtractValue(response, "confidence"));
      
      Print("  • ", symbol, " ", tfName, ": ", signal, " (", DoubleToString(confidence * 100, 1), "%)");
      
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| إنشاء JSON لإطار زمني محدد                                       |
//+------------------------------------------------------------------+
string CreateTimeframeJSON(string symbol, string tfName, MqlRates &rates[])
{
   string json = "{";
   json += "\"symbol\":\"" + symbol + "\",";
   json += "\"timeframe\":\"" + tfName + "\",";  // مهم: إرسال الإطار الزمني الصحيح
   json += "\"data\":[";
   
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
   
   json += "]}";
   
   return json;
}

//+------------------------------------------------------------------+
//| تحديث لوحة المعلومات المتقدمة                                    |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   string dashboard = "🤖 ForexML Multi-TF Bot v4.0\n";
   dashboard += UseAllTimeframes ? "📊 Using ALL Timeframes (100%)\n" : "📊 Using M5 Only (25%)\n";
   dashboard += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   
   for(int i = 0; i < totalPairs; i++)
   {
      string symbol = pairs[i];
      string signal = lastSignals[i];
      double conf = lastConfidence[i];
      
      string signalIcon = "⚪";
      if(signal == "BUY" || signal == "STRONG_BUY") signalIcon = "🟢";
      else if(signal == "SELL" || signal == "STRONG_SELL") signalIcon = "🔴";
      else if(signal == "NEUTRAL") signalIcon = "🟡";
      
      dashboard += signalIcon + " " + symbol + ": " + signal;
      
      if(conf > 0)
      {
         dashboard += " (" + DoubleToString(conf * 100, 0) + "%";
         if(UseAllTimeframes && combinedConfidence[i] > 0)
            dashboard += " avg)";
         else
            dashboard += ")";
      }
      
      dashboard += "\n";
   }
   
   dashboard += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   dashboard += "⏰ " + TimeToString(TimeCurrent());
   if(UseAllTimeframes)
      dashboard += " | 32 models active";
   
   Comment(dashboard);
}

//+------------------------------------------------------------------+
//| باقي الدوال مثل السابق                                           |
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

void OpenBuyPosition(string symbol, double confidence)
{
   double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   
   double lotSize = CalculateLotSize(symbol);
   double sl = NormalizeDouble(price - 300 * point, digits);
   double tp = NormalizeDouble(price + 600 * point, digits);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = MagicNumber;
   request.comment = "ML_MTF_Buy_" + DoubleToString(confidence, 2);
   
   if(OrderSend(request, result))
   {
      Print("✅ Buy position opened for ", symbol, ". Ticket: ", result.order);
   }
}

void OpenSellPosition(string symbol, double confidence)
{
   double price = SymbolInfoDouble(symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   
   double lotSize = CalculateLotSize(symbol);
   double sl = NormalizeDouble(price + 300 * point, digits);
   double tp = NormalizeDouble(price - 600 * point, digits);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = MagicNumber;
   request.comment = "ML_MTF_Sell_" + DoubleToString(confidence, 2);
   
   if(OrderSend(request, result))
   {
      Print("✅ Sell position opened for ", symbol, ". Ticket: ", result.order);
   }
}

double CalculateLotSize(string symbol)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * RiskPerTrade;
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double stopLossPoints = 300;
   
   double lotSize = riskAmount / (stopLossPoints * tickValue);
   
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   return NormalizeDouble(lotSize, 2);
}

string ExtractValue(string json, string key)
{
   int keyPos = StringFind(json, "\"" + key + "\":");
   if(keyPos == -1) return "";
   
   int start = keyPos + StringLen(key) + 3;
   int end = StringFind(json, ",", start);
   if(end == -1) end = StringFind(json, "}", start);
   
   string value = StringSubstr(json, start, end - start);
   StringReplace(value, "\"", "");
   StringTrimLeft(value);
   StringTrimRight(value);
   
   return value;
}

string SendToServer(string jsonData)
{
   string headers = "Content-Type: application/json\r\n";
   char post[], result[];
   string resultHeaders;
   
   StringToCharArray(jsonData, post, 0, StringLen(jsonData));
   
   int timeout = 5000;
   string url = ServerUrl + "/predict";
   
   int res = WebRequest("POST", url, headers, timeout, post, result, resultHeaders);
   
   if(res == -1)
   {
      return "";
   }
   
   return CharArrayToString(result);
}

void OnDeinit(const int reason)
{
   Comment("");
   Print("👋 ForexML Multi-TF Bot Stopped");
}
//+------------------------------------------------------------------+