//+------------------------------------------------------------------+
//|                                                   ForexMLBot.mq5 |
//|                                 Forex ML Trading Bot EA          |
//+------------------------------------------------------------------+
#property copyright "Forex ML System"
#property version   "1.00"
#property description "يتصل بـ Python Bot للحصول على الإشارات وينفذ الصفقات"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

// إعدادات الاتصال
input string   PythonServerURL = "http://69.62.121.53:5000";  // عنوان الخادم
input int      SignalCheckInterval = 60;                       // فترة فحص الإشارات (ثانية)
input double   RiskPerTrade = 0.01;                           // المخاطرة لكل صفقة (1%)
input double   DefaultLotSize = 0.01;                         // حجم اللوت الافتراضي
input int      MagicNumber = 123456;                          // الرقم السحري
input bool     SendTradeReports = true;                       // إرسال تقارير الصفقات
input int      MaxPositions = 3;                              // أقصى عدد صفقات مفتوحة
input int      Slippage = 30;                                 // الانزلاق المسموح

// المتغيرات العامة
CTrade trade;
CSymbolInfo symbolInfo;
CPositionInfo positionInfo;

datetime lastSignalCheck = 0;
string allowedPairs[] = {
    // العملات الأساسية
    "EURUSDm",   // اليورو/دولار
    "GBPUSDm",   // الباوند/دولار
    "USDJPYm",   // دولار/ين
    "USDCHFm",   // دولار/فرنك سويسري
    "AUDUSDm",   // دولار أسترالي
    "USDCADm",   // دولار/كندي
    "NZDUSDm",   // دولار نيوزيلندي
    
    // المعادن الثمينة
    "XAUUSDm",   // الذهب
    "XAGUSDm",   // الفضة
    
    // أزواج إضافية مهمة
    "EURGBPm",   // يورو/باوند
    "EURJPYm",   // يورو/ين
    "GBPJPYm"    // باوند/ين
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // إعداد التداول
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // التحقق من الاتصال
   if(!CheckServerConnection())
   {
      Print("❌ فشل الاتصال بخادم Python");
      return(INIT_FAILED);
   }
   
   Print("✅ تم تهيئة Forex ML Bot بنجاح");
   Print("📡 متصل بـ: ", PythonServerURL);
   
   // اختبار شامل للاتصال
   TestServerCommunication();
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🛑 إيقاف Forex ML Bot");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // فحص الإشارات كل فترة محددة
   if(TimeCurrent() - lastSignalCheck >= SignalCheckInterval)
   {
      CheckAndExecuteSignals();
      lastSignalCheck = TimeCurrent();
   }
   
   // تحديث وقف الخسارة المتحرك
   UpdateTrailingStops();
   
   // إرسال تقارير الصفقات المغلقة
   if(SendTradeReports)
   {
      ReportClosedTrades();
   }
}

//+------------------------------------------------------------------+
//| فحص وتنفيذ الإشارات                                              |
//+------------------------------------------------------------------+
void CheckAndExecuteSignals()
{
   // التحقق من عدد الصفقات المفتوحة
   int openPositions = CountOpenPositions();
   if(openPositions >= MaxPositions)
   {
      Print("⚠️ وصلنا للحد الأقصى من الصفقات: ", openPositions);
      return;
   }
   
   // فحص كل زوج مسموح
   for(int i = 0; i < ArraySize(allowedPairs); i++)
   {
      string symbol = allowedPairs[i];
      
      // التحقق من وجود صفقة مفتوحة لهذا الزوج
      if(IsPositionOpen(symbol)) continue;
      
      // الحصول على السعر الحالي
      if(!symbolInfo.Name(symbol)) continue;
      symbolInfo.RefreshRates();
      
      double currentPrice = symbolInfo.Ask();
      
      // طلب إشارة من الخادم
      string signal = GetSignalFromServer(symbol, currentPrice);
      
      if(signal != "")
      {
         ExecuteSignal(symbol, signal);
      }
   }
}

//+------------------------------------------------------------------+
//| الحصول على إشارة من خادم Python                                  |
//+------------------------------------------------------------------+
string GetSignalFromServer(string symbol, double price)
{
   string url = PythonServerURL + "/get_signal";
   
   // إعداد JSON
   string jsonData = "{\"symbol\":\"" + symbol + "\",\"price\":" + DoubleToString(price, 5) + "}";
   
   // Debug print
   Print("📤 Sending to server: ", jsonData);
   Print("🌐 URL: ", url);
   
   // تحضير البيانات
   char postData[], resultData[];
   string resultHeaders;
   StringToCharArray(jsonData, postData);
   
   // إضافة null terminator
   int dataSize = ArraySize(postData);
   ArrayResize(postData, dataSize + 1);
   postData[dataSize] = 0;
   
   string headers = "Content-Type: application/json\r\n";
   int timeout = 10000; // 10 ثواني
   
   // إرسال الطلب
   int res = WebRequest(
      "POST",
      url,
      headers,
      timeout,
      postData,
      resultData,
      resultHeaders
   );
   
   // Debug response
   Print("📥 Response code: ", res);
   
   if(res == -1)
   {
      int error = GetLastError();
      Print("❌ WebRequest error: ", error);
      
      // محاولة تشخيص المشكلة
      if(error == 4060) Print("❌ URL not allowed in MT5 settings");
      if(error == 4014) Print("❌ WebRequest not allowed");
      
      return "";
   }
   
   string result = CharArrayToString(resultData);
   Print("📥 Server response: ", result);
   
   // تحليل الاستجابة
   if(StringFind(result, "\"action\"") >= 0)
   {
      return result;
   }
   
   Print("⚠️ Invalid response format");
   return "";
}

//+------------------------------------------------------------------+
//| تنفيذ الإشارة                                                    |
//+------------------------------------------------------------------+
void ExecuteSignal(string symbol, string signalJson)
{
   // استخراج البيانات من JSON (نسخة مبسطة)
   string action = ExtractValue(signalJson, "action");
   double confidence = StringToDouble(ExtractValue(signalJson, "confidence"));
   double sl = StringToDouble(ExtractValue(signalJson, "sl"));
   double tp = StringToDouble(ExtractValue(signalJson, "tp"));
   double lot = StringToDouble(ExtractValue(signalJson, "lot"));
   
   // التحقق من الثقة
   if(confidence < 0.7)
   {
      Print("⚠️ ثقة منخفضة: ", confidence, " لـ ", symbol);
      return;
   }
   
   // حساب حجم الصفقة
   if(lot <= 0) lot = CalculateLotSize(symbol, sl);
   
   symbolInfo.Name(symbol);
   symbolInfo.RefreshRates();
   
   bool success = false;
   
   if(action == "BUY")
   {
      double price = symbolInfo.Ask();
      success = trade.Buy(lot, symbol, price, sl, tp, "ML Signal");
   }
   else if(action == "SELL")
   {
      double price = symbolInfo.Bid();
      success = trade.Sell(lot, symbol, price, sl, tp, "ML Signal");
   }
   
   if(success)
   {
      Print("✅ تم تنفيذ ", action, " ", symbol, " بحجم ", lot);
      
      // إرسال تأكيد للخادم
      SendTradeConfirmation(symbol, action, lot);
   }
   else
   {
      Print("❌ فشل تنفيذ ", action, " ", symbol, " - خطأ: ", trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| حساب حجم الصفقة بناءً على المخاطرة                               |
//+------------------------------------------------------------------+
double CalculateLotSize(string symbol, double stopLoss)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * RiskPerTrade;
   
   symbolInfo.Name(symbol);
   
   double tickValue = symbolInfo.TickValue();
   double tickSize = symbolInfo.TickSize();
   double point = symbolInfo.Point();
   
   double slPoints = MathAbs(symbolInfo.Ask() - stopLoss) / point;
   
   double lotSize = riskAmount / (slPoints * tickValue);
   
   // التقريب لأقرب حجم صحيح
   double minLot = symbolInfo.LotsMin();
   double maxLot = symbolInfo.LotsMax();
   double stepLot = symbolInfo.LotsStep();
   
   lotSize = MathFloor(lotSize / stepLot) * stepLot;
   lotSize = MathMax(minLot, MathMin(lotSize, maxLot));
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| تحديث وقف الخسارة المتحرك                                        |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(positionInfo.SelectByIndex(i))
      {
         if(positionInfo.Magic() != MagicNumber) continue;
         
         string symbol = positionInfo.Symbol();
         symbolInfo.Name(symbol);
         symbolInfo.RefreshRates();
         
         double currentPrice = positionInfo.PositionType() == POSITION_TYPE_BUY ? 
                              symbolInfo.Bid() : symbolInfo.Ask();
         
         double currentSL = positionInfo.StopLoss();
         double openPrice = positionInfo.PriceOpen();
         double atr = GetATR(symbol);
         double newSL = 0;
         
         if(positionInfo.PositionType() == POSITION_TYPE_BUY)
         {
            // للشراء: رفع SL إذا ارتفع السعر
            if(currentPrice > openPrice + atr)
            {
               newSL = currentPrice - atr * 1.5;
               if(newSL > currentSL && newSL < currentPrice - symbolInfo.StopsLevel() * symbolInfo.Point())
               {
                  trade.PositionModify(positionInfo.Ticket(), newSL, positionInfo.TakeProfit());
               }
            }
         }
         else
         {
            // للبيع: خفض SL إذا انخفض السعر
            if(currentPrice < openPrice - atr)
            {
               newSL = currentPrice + atr * 1.5;
               if(newSL < currentSL && newSL > currentPrice + symbolInfo.StopsLevel() * symbolInfo.Point())
               {
                  trade.PositionModify(positionInfo.Ticket(), newSL, positionInfo.TakeProfit());
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| إرسال تقارير الصفقات المغلقة                                      |
//+------------------------------------------------------------------+
void ReportClosedTrades()
{
   static datetime lastReportTime = 0;
   
   // فحص آخر 10 صفقات
   for(int i = HistoryDealsTotal() - 1; i >= MathMax(0, HistoryDealsTotal() - 10); i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      
      // التحقق من أنها صفقة جديدة
      datetime dealTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      if(dealTime <= lastReportTime) continue;
      
      // التحقق من أنها صفقة إغلاق
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      
      // التحقق من الرقم السحري
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != MagicNumber) continue;
      
      // إرسال التقرير
      SendTradeReport(ticket);
      lastReportTime = dealTime;
   }
}

//+------------------------------------------------------------------+
//| إرسال تقرير صفقة للخادم                                          |
//+------------------------------------------------------------------+
void SendTradeReport(ulong dealTicket)
{
   string symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
   double volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
   double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
   double price = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
   datetime time = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
   
   string url = PythonServerURL + "/report_trade";
   string headers = "Content-Type: application/json\r\n";
   
   string jsonData = StringFormat(
      "{\"symbol\":\"%s\",\"volume\":%.2f,\"profit\":%.2f,\"price\":%.5f,\"timestamp\":\"%s\"}",
      symbol, volume, profit, price, TimeToString(time)
   );
   
   char postData[], resultData[];
   StringToCharArray(jsonData, postData);
   
   WebRequest("POST", url, headers, 5000, postData, resultData, headers);
   
   Print("📊 تقرير صفقة: ", symbol, " ربح/خسارة: ", profit);
}

//+------------------------------------------------------------------+
//| وظائف مساعدة                                                     |
//+------------------------------------------------------------------+

// عد الصفقات المفتوحة
int CountOpenPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(positionInfo.SelectByIndex(i))
      {
         if(positionInfo.Magic() == MagicNumber)
            count++;
      }
   }
   return count;
}

// التحقق من وجود صفقة مفتوحة
bool IsPositionOpen(string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(positionInfo.SelectByIndex(i))
      {
         if(positionInfo.Magic() == MagicNumber && positionInfo.Symbol() == symbol)
            return true;
      }
   }
   return false;
}

// الحصول على ATR
double GetATR(string symbol, int period = 14)
{
   double atr[];
   ArraySetAsSeries(atr, true);
   
   int handle = iATR(symbol, PERIOD_CURRENT, period);
   if(handle != INVALID_HANDLE)
   {
      CopyBuffer(handle, 0, 0, 1, atr);
      IndicatorRelease(handle);
      return atr[0];
   }
   
   return symbolInfo.Ask() * 0.001; // 0.1% كقيمة افتراضية
}

// استخراج قيمة من JSON
string ExtractValue(string json, string key)
{
   int start = StringFind(json, "\"" + key + "\":");
   if(start < 0) return "";
   
   start = StringFind(json, ":", start) + 1;
   
   // تخطي المسافات
   while(StringGetCharacter(json, start) == ' ') start++;
   
   // إذا كانت القيمة نصية
   if(StringGetCharacter(json, start) == '"')
   {
      start++;
      int end = StringFind(json, "\"", start);
      return StringSubstr(json, start, end - start);
   }
   
   // إذا كانت القيمة رقمية
   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ' ') break;
      end++;
   }
   
   return StringSubstr(json, start, end - start);
}

// التحقق من الاتصال بالخادم
bool CheckServerConnection()
{
   string url = PythonServerURL + "/health";
   char data[], result[];
   string headers;
   
   Print("🔍 Testing connection to: ", url);
   
   int res = WebRequest("GET", url, "", 5000, data, result, headers);
   
   if(res != -1)
   {
      string response = CharArrayToString(result);
      Print("✅ Server health check response: ", response);
      return true;
   }
   else
   {
      int error = GetLastError();
      Print("❌ Connection test failed. Error: ", error);
      return false;
   }
}

// اختبار الخادم بإرسال بيانات تجريبية
void TestServerCommunication()
{
   Print("🧪 Testing server communication...");
   
   // اختبار /test endpoint
   string testUrl = PythonServerURL + "/test";
   string testData = "{\"test\":\"data\",\"value\":123}";
   
   char postData[], resultData[];
   string resultHeaders;
   StringToCharArray(testData, postData);
   
   string headers = "Content-Type: application/json\r\n";
   
   int res = WebRequest("POST", testUrl, headers, 5000, postData, resultData, resultHeaders);
   
   if(res != -1)
   {
      Print("✅ Test endpoint response: ", CharArrayToString(resultData));
   }
   else
   {
      Print("❌ Test endpoint failed");
   }
   
   // اختبار get_signal
   string signal = GetSignalFromServer("EURUSDm", 1.1000);
   if(signal != "")
   {
      Print("✅ Signal test successful");
   }
   else
   {
      Print("❌ Signal test failed");
   }
}

// إرسال تأكيد تنفيذ الصفقة
void SendTradeConfirmation(string symbol, string action, double lot)
{
   string url = PythonServerURL + "/confirm_trade";
   string headers = "Content-Type: application/json\r\n";
   
   string jsonData = StringFormat(
      "{\"symbol\":\"%s\",\"action\":\"%s\",\"lot\":%.2f,\"timestamp\":\"%s\"}",
      symbol, action, lot, TimeToString(TimeCurrent())
   );
   
   char postData[], resultData[];
   StringToCharArray(jsonData, postData);
   
   WebRequest("POST", url, headers, 5000, postData, resultData, headers);
}