//+------------------------------------------------------------------+
//| تحديث دالة ParseResponse لاستقبال SL/TP من السيرفر            |
//+------------------------------------------------------------------+

// استبدل دالة ParseResponse القديمة بهذه:

void ParseResponse(string response, string &signal, double &confidence, 
                   double &serverSL, double &serverTP)
{
    signal = "NONE";
    confidence = 0;
    serverSL = 0;
    serverTP = 0;
    
    if(response == "") return;
    
    // استخراج الإشارة
    int signalPos = StringFind(response, "\"signal\":");
    if(signalPos >= 0) {
        int signalStart = signalPos + 10;
        int signalEnd = StringFind(response, "\"", signalStart + 1);
        if(signalEnd > signalStart) {
            signal = StringSubstr(response, signalStart + 1, signalEnd - signalStart - 1);
        }
    }
    
    // استخراج الثقة
    int confPos = StringFind(response, "\"confidence\":");
    if(confPos >= 0) {
        int confStart = confPos + 13;
        int confEnd = StringFind(response, ",", confStart);
        if(confEnd < 0) confEnd = StringFind(response, "}", confStart);
        
        if(confEnd > confStart) {
            string confStr = StringSubstr(response, confStart, confEnd - confStart);
            confidence = StringToDouble(confStr);
        }
    }
    
    // استخراج SL/TP إذا كان متاحاً
    int slPos = StringFind(response, "\"stop_loss\":");
    if(slPos >= 0) {
        int slStart = slPos + 12;
        int slEnd = StringFind(response, ",", slStart);
        if(slEnd > slStart) {
            string slStr = StringSubstr(response, slStart, slEnd - slStart);
            serverSL = StringToDouble(slStr);
        }
    }
    
    int tpPos = StringFind(response, "\"take_profit\":");
    if(tpPos >= 0) {
        int tpStart = tpPos + 14;
        int tpEnd = StringFind(response, ",", tpStart);
        if(tpEnd < 0) tpEnd = StringFind(response, "}", tpStart);
        
        if(tpEnd > tpStart) {
            string tpStr = StringSubstr(response, tpStart, tpEnd - tpStart);
            serverTP = StringToDouble(tpStr);
        }
    }
}

//+------------------------------------------------------------------+
//| تحديث دالة CalculateSLTP لاستخدام قيم السيرفر إذا كانت متاحة   |
//+------------------------------------------------------------------+

void CalculateSLTP(string symbol, string signal, double price, 
                   double &sl, double &tp, double serverSL, double serverTP)
{
    InstrumentInfo info = GetInstrumentInfo(symbol);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    // إذا كان السيرفر أرسل SL/TP، استخدمها
    if(serverSL > 0 && serverTP > 0) {
        sl = serverSL;
        tp = serverTP;
        
        // التحقق من صحة القيم
        double slDistance = MathAbs(price - sl) / (point * 10);
        double tpDistance = MathAbs(tp - price) / (point * 10);
        
        // إذا كانت القيم غير منطقية، احسب محلياً
        if(slDistance < 5 || slDistance > 500 || tpDistance < 5 || tpDistance > 1000) {
            Print("⚠️ Server SL/TP values seem incorrect, using local calculation");
            // استخدم الحساب المحلي
            CalculateLocalSLTP(symbol, signal, price, sl, tp);
        } else {
            Print("✅ Using ML-optimized SL/TP from server");
        }
    }
    else {
        // احسب محلياً إذا لم يرسل السيرفر قيم
        CalculateLocalSLTP(symbol, signal, price, sl, tp);
    }
    
    // التحقق النهائي من الحدود
    ValidateSLTPLimits(symbol, signal, price, sl, tp, info);
}

//+------------------------------------------------------------------+
//| دالة الحساب المحلي (الطريقة الحالية)                          |
//+------------------------------------------------------------------+
void CalculateLocalSLTP(string symbol, string signal, double price, 
                        double &sl, double &tp)
{
    // نقل الكود الحالي لحساب SL/TP هنا
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
            // ... الكود الحالي ...
            break;
    }
}

//+------------------------------------------------------------------+
//| تحديث ProcessPair لاستقبال SL/TP                               |
//+------------------------------------------------------------------+
void ProcessPair(int pairIndex)
{
    string symbol = activePairs[pairIndex];
    
    // متغيرات جديدة لـ SL/TP من السيرفر
    double serverSL = 0, serverTP = 0;
    
    // ... كود جمع البيانات ...
    
    // إرسال البيانات للخادم
    string response = SendToServer(jsonData);
    
    // تحديث: استقبال SL/TP أيضاً
    ParseResponse(response, tfSignals[tf], tfConfidences[tf], serverSL, serverTP);
    
    // ... باقي الكود ...
    
    // عند التداول، مرر SL/TP من السيرفر
    if(EnableTrading && combinedConf >= MinCombinedConfidence) {
        if(combinedSignal == "BUY" || combinedSignal == "SELL") {
            ExecuteTradeWithServerSLTP(symbol, combinedSignal, combinedConf, serverSL, serverTP);
        }
    }
}