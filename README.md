# ForeQuant 📈

**AI-Powered Stock Return Forecaster for Nifty 50**

A quantitative finance platform that predicts stock returns using historical trends, momentum analysis, and seasonal patterns. Built for the Ideathon 2025.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- **Return Forecasting** - Predict stock returns for 7, 14, 30, 60, or 90 days
- **BUY/HOLD/SELL Recommendations** - Clear actionable signals with confidence scores
- **Trend Analysis** - Moving averages, momentum indicators, trend direction
- **Seasonal Patterns** - Historical monthly performance analysis
- **Modern UI** - Premium dark theme trading terminal interface

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ideathon-2025/Team-Encoders.git
cd Team-Encoders
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
Navigate to **http://localhost:5000**

---

## 📊 How It Works

### Forecasting Methodology

The prediction engine combines multiple factors:

1. **Historical Average Returns** - Baseline expected return
2. **Recent Momentum** - 10-day and 30-day price momentum
3. **Moving Average Signals** - Position relative to 20/50/200-day MAs
4. **Trend Direction** - Upward/Downward trend identification
5. **Seasonal Patterns** - Monthly historical performance

### Recommendation Logic

| Score | Recommendation | Description |
|-------|----------------|-------------|
| ≥ 3   | STRONG BUY     | Multiple bullish signals |
| 1-2   | BUY            | Positive outlook |
| -1 to 0 | HOLD         | Mixed signals |
| -3 to -2 | SELL        | Bearish indicators |
| < -3  | STRONG SELL    | High risk |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Statistical analysis

### Frontend
- **HTML5 / CSS3** - Modern UI with glassmorphism
- **JavaScript** - Dynamic interactions
- **Chart.js** - Interactive visualizations

---

## 📁 Project Structure

```
ForeQuant/
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── models/
│   ├── forecaster.py      # Main prediction engine
│   ├── arima.py           # ARIMA time series model
│   ├── garch.py           # GARCH volatility model
│   ├── factors.py         # Fama-French factor analysis
│   ├── portfolio.py       # Portfolio optimization
│   ├── risk.py            # Risk metrics
│   └── backtest.py        # Backtesting engine
├── static/
│   ├── index.html         # Main UI
│   ├── css/style.css      # Styling
│   └── js/app.js          # Frontend logic
└── data/
    ├── *.csv              # Stock OHLCV data
    └── *_info.json        # Stock information
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List all available stocks |
| GET | `/api/stock/<symbol>` | Get OHLCV data |
| GET | `/api/forecast/<symbol>?days=30` | Get forecast & recommendation |
| GET | `/api/risk/<symbol>` | Get risk metrics |

### Example Response
```json
{
  "recommendation": "BUY",
  "action": "Good entry point for investors",
  "forecast": {
    "expected_return_pct": 5.2,
    "target_price": 2584.50,
    "confidence_score": 72
  },
  "trends": {
    "trend_direction": "UPWARD",
    "momentum_10d": 3.2
  }
}
```

---

## 👥 Team Encoders

Built for **Ideathon 2025**

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only. The predictions are based on historical patterns and statistical analysis. Past performance does not guarantee future results. This is not financial advice.
