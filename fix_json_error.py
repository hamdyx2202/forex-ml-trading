#!/usr/bin/env python3
"""
إصلاح خطأ JSON في run_forex_ml_server.py
"""

# قراءة الملف الأصلي
with open('run_forex_ml_server.py', 'r') as f:
    content = f.read()

# إضافة معالجة أفضل للأخطاء في دالة predict
new_predict_function = '''@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with better error handling"""
    try:
        server_stats['total_requests'] += 1
        
        # Better JSON parsing
        try:
            if request.content_type != 'application/json':
                logger.warning(f"Received content type: {request.content_type}")
            
            # Get raw data for debugging
            raw_data = request.get_data(as_text=True)
            logger.info(f"Raw data length: {len(raw_data)} chars")
            
            # Try to parse JSON
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Error at position: {e.pos}")
                # Try to extract valid JSON up to error position
                if e.pos > 0:
                    try:
                        partial_data = raw_data[:e.pos]
                        data = json.loads(partial_data)
                        logger.warning("Using partial JSON data")
                    except:
                        return jsonify({
                            'error': 'Invalid JSON format',
                            'action': 'NONE',
                            'confidence': 0
                        }), 200
                else:
                    return jsonify({
                        'error': 'Invalid JSON format',
                        'action': 'NONE',
                        'confidence': 0
                    }), 200
                    
        except Exception as e:
            logger.error(f"Request parsing error: {str(e)}")
            return jsonify({
                'error': 'Invalid request format',
                'action': 'NONE',
                'confidence': 0
            }), 200
        
        # Extract required fields
        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        
        logger.info(f"\\n📊 Prediction request: {symbol} {timeframe}")
        logger.info(f"   Received {len(candles)} candles")
        
        # Validate candles
        if not candles or len(candles) < 20:
            logger.warning(f"Not enough candles: {len(candles)}")
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'action': 'NONE',
                'confidence': 0,
                'error': f'Need at least 20 candles, received {len(candles)}'
            }), 200
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(candles)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            df = df.dropna()
            
        except Exception as e:
            logger.error(f"DataFrame conversion error: {str(e)}")
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'action': 'NONE',
                'confidence': 0,
                'error': 'Invalid candle data format'
            }), 200
        
        # Calculate features
        try:
            features = unified_system.calculate_features(df)
            latest_features = features.iloc[-1:].copy()
        except Exception as e:
            logger.error(f"Feature calculation error: {str(e)}")
            # Simple fallback prediction
            return simple_prediction_response(symbol, timeframe, df)
        
        # Make prediction
        try:
            prediction, confidence = unified_system.predict(symbol, timeframe, latest_features)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            prediction, confidence = 2, 0.5  # Hold
        
        # Generate signal
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
        
        # Calculate SL/TP
        try:
            atr = features.get('atr_14', pd.Series([50 * pip_value])).iloc[-1]
            sl_pips = max(min(atr / pip_value * 1.5, 100), 20)
        except:
            sl_pips = 50  # Default
            
        tp1_pips = sl_pips * 2.0
        tp2_pips = sl_pips * 3.0
        
        # Calculate prices
        if action == 'BUY':
            sl_price = current_price - (sl_pips * pip_value)
            tp1_price = current_price + (tp1_pips * pip_value)
            tp2_price = current_price + (tp2_pips * pip_value)
        elif action == 'SELL':
            sl_price = current_price + (sl_pips * pip_value)
            tp1_price = current_price - (tp1_pips * pip_value)
            tp2_price = current_price - (tp2_pips * pip_value)
        else:
            sl_price = tp1_price = tp2_price = current_price
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_pips),
            'tp1_pips': float(tp1_pips),
            'tp2_pips': float(tp2_pips),
            'risk_reward_ratio': float(tp1_pips / sl_pips) if sl_pips > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        if action != 'NONE':
            server_stats['total_signals'] += 1
        
        logger.info(f"   ✅ {action} signal with {confidence:.1%} confidence")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'action': 'NONE',
            'confidence': 0
        }), 200

def simple_prediction_response(symbol, timeframe, df):
    """Simple prediction response when features fail"""
    try:
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        # Simple MA strategy
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20
        
        if current_price > sma20 > sma50:
            action = 'BUY'
            confidence = 0.65
        elif current_price < sma20 < sma50:
            action = 'SELL'
            confidence = 0.65
        else:
            action = 'NONE'
            confidence = 0.5
            
        sl_pips = 50
        tp1_pips = 100
        tp2_pips = 150
        
        if action == 'BUY':
            sl_price = current_price - (sl_pips * pip_value)
            tp1_price = current_price + (tp1_pips * pip_value)
            tp2_price = current_price + (tp2_pips * pip_value)
        elif action == 'SELL':
            sl_price = current_price + (sl_pips * pip_value)
            tp1_price = current_price - (tp1_pips * pip_value)
            tp2_price = current_price - (tp2_pips * pip_value)
        else:
            sl_price = tp1_price = tp2_price = current_price
            
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': confidence,
            'current_price': current_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'sl_pips': sl_pips,
            'tp1_pips': tp1_pips,
            'tp2_pips': tp2_pips,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'action': 'NONE',
            'confidence': 0,
            'error': str(e)
        }), 200'''

# البحث عن دالة predict الأصلية وإستبدالها
import re

# البحث عن بداية دالة predict
pattern = r'@app\.route\(\'/predict\'.*?\n(?:def predict.*?(?=@app\.route|def main|$))'
content = re.sub(pattern, new_predict_function + '\n\n', content, flags=re.DOTALL)

# حفظ الملف المحدث
with open('run_forex_ml_server_fixed.py', 'w') as f:
    f.write(content)

print("✅ تم إنشاء run_forex_ml_server_fixed.py مع معالجة أفضل للأخطاء")