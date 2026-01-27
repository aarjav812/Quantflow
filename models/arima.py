"""
ARIMA Model for Time Series Forecasting
Auto-selects optimal (p,d,q) parameters using AIC/BIC
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def find_optimal_order(series, max_p=5, max_d=2, max_q=5):
    """
    Find optimal ARIMA order using AIC
    """
    best_aic = float('inf')
    best_order = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order, best_aic


def fit_arima(prices, order=None, forecast_days=30):
    """
    Fit ARIMA model and generate forecasts
    
    Parameters:
    -----------
    prices : pd.Series or np.array
        Historical price data
    order : tuple (p,d,q) or None
        ARIMA order, if None will auto-select
    forecast_days : int
        Number of days to forecast
        
    Returns:
    --------
    dict with forecasts, confidence intervals, and model info
    """
    # Ensure we have a series
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Use returns for better stationarity
    returns = prices.pct_change().dropna()
    
    # Auto-select order if not provided
    if order is None:
        # Use simplified grid search for speed
        order, aic = find_optimal_order(returns, max_p=3, max_d=1, max_q=3)
    
    # Fit the model
    model = ARIMA(returns, order=order)
    fitted = model.fit()
    
    # Generate forecasts
    forecast_result = fitted.get_forecast(steps=forecast_days)
    forecast_returns = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    
    # Convert returns to prices
    last_price = prices.iloc[-1]
    forecast_prices = [last_price]
    lower_prices = [last_price]
    upper_prices = [last_price]
    
    for i, ret in enumerate(forecast_returns):
        forecast_prices.append(forecast_prices[-1] * (1 + ret))
        lower_prices.append(lower_prices[-1] * (1 + conf_int.iloc[i, 0]))
        upper_prices.append(upper_prices[-1] * (1 + conf_int.iloc[i, 1]))
    
    # Remove the seed value
    forecast_prices = forecast_prices[1:]
    lower_prices = lower_prices[1:]
    upper_prices = upper_prices[1:]
    
    # Model diagnostics
    residuals = fitted.resid
    
    return {
        'order': order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'forecast_prices': forecast_prices,
        'lower_bound': lower_prices,
        'upper_bound': upper_prices,
        'forecast_returns': forecast_returns.tolist(),
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'last_price': float(last_price),
        'stationarity': check_stationarity(returns)
    }


def rolling_forecast(prices, window=252, horizon=5):
    """
    Perform rolling window ARIMA forecasts for backtesting
    
    Parameters:
    -----------
    prices : pd.Series
        Historical prices
    window : int
        Training window size
    horizon : int
        Forecast horizon
        
    Returns:
    --------
    DataFrame with forecasts vs actuals
    """
    results = []
    
    for i in range(window, len(prices) - horizon):
        train = prices.iloc[i-window:i]
        actual = prices.iloc[i:i+horizon]
        
        try:
            forecast_result = fit_arima(train, forecast_days=horizon)
            forecast_prices = forecast_result['forecast_prices']
            
            results.append({
                'date': prices.index[i] if hasattr(prices, 'index') else i,
                'actual_1d': actual.iloc[0] if len(actual) > 0 else None,
                'forecast_1d': forecast_prices[0] if len(forecast_prices) > 0 else None,
                'actual_5d': actual.iloc[-1] if len(actual) >= horizon else None,
                'forecast_5d': forecast_prices[-1] if len(forecast_prices) >= horizon else None,
            })
        except:
            continue
    
    return pd.DataFrame(results)
