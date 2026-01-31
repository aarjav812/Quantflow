"""
Stock Return Forecasting Platform API
Focused on practical, actionable predictions
"""
import os
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Import forecaster
from models.forecaster import StockForecaster, forecast_stock


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


app = Flask(__name__, static_folder='static')
app.json_encoder = NumpyEncoder
CORS(app)


# Data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

# Cache for loaded data
_stock_data_cache = {}
_fundamentals_cache = None


def load_stock_data(symbol):
    """Load stock OHLCV data from CSV"""
    if symbol in _stock_data_cache:
        return _stock_data_cache[symbol]
    
    print(f"Loading data for symbol: {symbol}")
    
    # Try different file patterns
    file_patterns = [
        f"{symbol}.csv",
        f"{symbol}.NS.csv",
        f"{symbol.replace('.NS', '')}.csv",
        f"{symbol.upper()}.csv",
        f"{symbol.upper()}.NS.csv",
    ]
    
    for pattern in file_patterns:
        filepath = os.path.join(DATA_DIR, pattern)
        if os.path.exists(filepath):
            print(f"Found file: {filepath}")
            try:
                df = pd.read_csv(filepath)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                _stock_data_cache[symbol] = df
                return df
            except Exception as e:
                print(f"Error reading CSV {filepath}: {e}")
                continue
    
    print(f"No file found for {symbol} in {DATA_DIR}")
    return None


def load_fundamentals():
    """Load fundamentals data"""
    global _fundamentals_cache
    if _fundamentals_cache is not None:
        return _fundamentals_cache
    
    filepath = os.path.join(DATA_DIR, 'nifty50_fundamentals.csv')
    if os.path.exists(filepath):
        try:
            _fundamentals_cache = pd.read_csv(filepath)
            return _fundamentals_cache
        except Exception as e:
            print(f"Error loading fundamentals: {e}")
            return None
    return None



# ============== API Routes ==============

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/stocks')
def get_stocks():
    """Get list of all available stocks"""
    fundamentals = load_fundamentals()
    stocks = []
    
    if fundamentals is not None:
        for _, row in fundamentals.iterrows():
            symbol = str(row.get('symbol', '')).replace('.NS', '') if pd.notna(row.get('symbol')) else ''
            if symbol:
                stocks.append({
                    'symbol': symbol,
                    'name': row.get('name', symbol) if pd.notna(row.get('name')) else symbol,
                    'sector': row.get('sector', 'Unknown') if pd.notna(row.get('sector')) else 'Unknown',
                })
    
    # Fallback: scan data directory
    if not stocks:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv') and not filename.startswith(('nifty50', 'training', 'enhanced')):
                symbol = filename.replace('.csv', '').replace('.NS', '')
                if not symbol.endswith('_info'):
                    stocks.append({'symbol': symbol, 'name': symbol, 'sector': 'Unknown'})
    
    return jsonify({'stocks': stocks})


@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """Get OHLCV data for a specific stock"""
    df = load_stock_data(symbol)
    if df is None:
        return jsonify({'error': f'Stock {symbol} not found'}), 404
    
    days = request.args.get('days', 365, type=int)
    df = df.tail(days)
    
    return jsonify({
        'symbol': symbol,
        'data': {
            'dates': df['Date'].astype(str).tolist() if 'Date' in df.columns else [],
            'open': df['Open'].tolist() if 'Open' in df.columns else [],
            'high': df['High'].tolist() if 'High' in df.columns else [],
            'low': df['Low'].tolist() if 'Low' in df.columns else [],
            'close': df['Close'].tolist() if 'Close' in df.columns else [],
            'volume': df['Volume'].tolist() if 'Volume' in df.columns else [],
        },
        'current_price': float(df['Close'].iloc[-1]) if len(df) > 0 and 'Close' in df.columns else 0,
    })


@app.route('/api/forecast/<symbol>')
def get_forecast(symbol):
    """
    Get complete stock forecast with recommendation
    
    This is the MAIN endpoint that powers the UI
    """
    df = load_stock_data(symbol)
    if df is None:
        return jsonify({'error': f'Stock {symbol} not found'}), 404
    
    forecast_days = request.args.get('days', 30, type=int)
    
    def convert_np(obj):
        import numpy as np
        import pandas as pd
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_np(i) for i in obj)
        elif isinstance(obj, set):
            return {convert_np(i) for i in obj}
        elif pd.isna(obj):
            return None
        else:
            return obj
    try:
        # Get prices and dates
        prices = df['Close'].dropna()
        dates = df['Date'] if 'Date' in df.columns else None
        
        # Run the forecaster
        result = forecast_stock(prices, dates, forecast_days)
        
        return jsonify(convert_np(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Keep old endpoints for compatibility
@app.route('/api/arima/<symbol>')
def get_arima_forecast(symbol):
    """Get ARIMA forecast (legacy endpoint)"""
    return get_forecast(symbol)


@app.route('/api/risk/<symbol>')
def get_risk_metrics(symbol):
    """Get risk metrics (simplified)"""
    df = load_stock_data(symbol)
    if df is None:
        return jsonify({'error': f'Stock {symbol} not found'}), 404
    
    try:
        prices = df['Close'].dropna()
        returns = prices.pct_change().dropna()
        
        return jsonify({
            'symbol': symbol,
            'annual_return': float(returns.mean() * 252 * 100),
            'annual_volatility': float(returns.std() * np.sqrt(252) * 100),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': float((prices / prices.expanding().max() - 1).min() * 100),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  ForeQuant - AI-Powered Stock Forecasts")
    print("=" * 60)
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Server: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
