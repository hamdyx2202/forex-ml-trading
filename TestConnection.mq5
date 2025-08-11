//+------------------------------------------------------------------+
//|                                              TestConnection.mq5  |
//|                                  اختبار الاتصال بالخادم فقط     |
//+------------------------------------------------------------------+
#property copyright "Test Connection"
#property version   "1.00"

input string ServerURL = "http://69.62.121.53:5000";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("Testing connection to: ", ServerURL);
    Print("========================================");
    
    // اختبار بسيط
    TestSimpleConnection();
    
    // اختبار health endpoint
    TestHealthEndpoint();
    
    // اختبار مع API
    TestAPIEndpoint();
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| اختبار بسيط                                                     |
//+------------------------------------------------------------------+
void TestSimpleConnection()
{
    Print("\n1. Simple Connection Test:");
    
    string url = ServerURL;
    string headers = "";
    char post[], result[];
    
    ResetLastError();
    int res = WebRequest("GET", url, headers, 5000, post, result, headers);
    
    Print("Response code: ", res);
    Print("Error code: ", GetLastError());
    
    if(res > 0)
    {
        string response = CharArrayToString(result);
        Print("Response: ", response);
    }
}

//+------------------------------------------------------------------+
//| اختبار health endpoint                                          |
//+------------------------------------------------------------------+
void TestHealthEndpoint()
{
    Print("\n2. Health Endpoint Test:");
    
    string url = ServerURL + "/health";
    string headers = "";
    char post[], result[];
    
    ResetLastError();
    int res = WebRequest("GET", url, headers, 5000, post, result, headers);
    
    Print("Response code: ", res);
    Print("Error code: ", GetLastError());
    
    if(res > 0)
    {
        string response = CharArrayToString(result);
        Print("✅ SUCCESS! Response: ", response);
    }
    else
    {
        int error = GetLastError();
        if(error == 4060)
        {
            Print("❌ ERROR 4060: WebRequest not allowed!");
            Print("SOLUTION:");
            Print("1. Go to: Tools → Options → Expert Advisors");
            Print("2. Check: Allow WebRequest for listed URL addresses");
            Print("3. Add these URLs:");
            Print("   - ", ServerURL);
            Print("   - http://69.62.121.53");
            Print("4. Click OK");
            Print("5. Remove and re-add this EA");
        }
        else if(error == 5203)
        {
            Print("❌ ERROR 5203: HTTP request failed");
            Print("Possible causes:");
            Print("1. Server not running");
            Print("2. Wrong URL");
            Print("3. Firewall blocking");
        }
    }
}

//+------------------------------------------------------------------+
//| اختبار API endpoint                                             |
//+------------------------------------------------------------------+
void TestAPIEndpoint()
{
    Print("\n3. API Test Endpoint:");
    
    string url = ServerURL + "/api/test";
    string headers = "Content-Type: application/json\r\n";
    char post[], result[];
    
    string json = "{\"api_key\":\"test\",\"test\":true}";
    StringToCharArray(json, post);
    
    ResetLastError();
    int res = WebRequest("POST", url, headers, 5000, post, result, headers);
    
    Print("Response code: ", res);
    Print("Error code: ", GetLastError());
    
    if(res > 0)
    {
        string response = CharArrayToString(result);
        Print("✅ API SUCCESS! Response: ", response);
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Test completed");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // لا نحتاج شيء هنا
}