"""
GARCH Model for Volatility Estimation
Captures volatility clustering in financial time series
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def fit_garch(prices, p=1, q=1, forecast_horizon=30, vol_target='Std'):
    """
    Fit GARCH(p,q) model and forecast volatility
    
    Parameters:
    -----------
    prices : pd.Series or np.array
        Historical price data
    p : int
        GARCH lag order
    q : int
        ARCH lag order
    forecast_horizon : int
        Number of periods to forecast
    vol_target : str
        'Std' for standard deviation, 'Var' for variance
        
    Returns:
    --------
    dict with volatility forecasts and model info
    """
    # Ensure we have a series
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate returns (multiply by 100 for numerical stability)
    returns = prices.pct_change().dropna() * 100
    
    # Fit GARCH model
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
    fitted = model.fit(disp='off')
    
    # Generate volatility forecasts
    forecast = fitted.forecast(horizon=forecast_horizon)
    
    # Extract variance forecasts and convert to annualized volatility
    variance_forecast = forecast.variance.iloc[-1].values
    
    if vol_target == 'Std':
        vol_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100  # Annualized
    else:
        vol_forecast = variance_forecast * 252 / 10000  # Annualized variance
    
    # Historical conditional volatility
    conditional_vol = fitted.conditional_volatility * np.sqrt(252) / 100
    
    # Model parameters
    params = fitted.params
    
    # Calculate VIX-style volatility (30-day forward vol)
    vix_style = np.sqrt(np.mean(variance_forecast[:min(30, len(variance_forecast))])) * np.sqrt(252) / 100
    
    return {
        'model_type': f'GARCH({p},{q})',
        'omega': float(params.get('omega', 0)),
        'alpha': float(params.get('alpha[1]', 0)),
        'beta': float(params.get('beta[1]', 0)),
        'persistence': float(params.get('alpha[1]', 0) + params.get('beta[1]', 0)),
        'unconditional_vol': float(np.sqrt(params.get('omega', 0) / (1 - params.get('alpha[1]', 0) - params.get('beta[1]', 0) + 1e-10)) * np.sqrt(252) / 100),
        'current_vol': float(conditional_vol.iloc[-1]),
        'vix_style_vol': float(vix_style * 100),  # As percentage
        'forecast_volatility': vol_forecast.tolist(),
        'historical_volatility': conditional_vol.tolist()[-252:],  # Last year
        'aic': float(fitted.aic),
        'bic': float(fitted.bic),
        'log_likelihood': float(fitted.loglikelihood),
    }


def fit_egarch(prices, p=1, q=1, forecast_horizon=30):
    """
    Fit EGARCH model (captures asymmetric volatility - leverage effect)
    
    Returns:
    --------
    dict with volatility forecasts and asymmetry parameter
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = prices.pct_change().dropna() * 100
    
    model = arch_model(returns, vol='EGARCH', p=p, q=q, dist='normal')
    fitted = model.fit(disp='off')
    
    forecast = fitted.forecast(horizon=forecast_horizon)
    variance_forecast = forecast.variance.iloc[-1].values
    vol_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100
    
    conditional_vol = fitted.conditional_volatility * np.sqrt(252) / 100
    
    params = fitted.params
    
    return {
        'model_type': f'EGARCH({p},{q})',
        'omega': float(params.get('omega', 0)),
        'alpha': float(params.get('alpha[1]', 0)),
        'gamma': float(params.get('gamma[1]', 0)),  # Asymmetry parameter
        'beta': float(params.get('beta[1]', 0)),
        'leverage_effect': float(params.get('gamma[1]', 0)) < 0,  # Negative gamma = leverage
        'current_vol': float(conditional_vol.iloc[-1]),
        'forecast_volatility': vol_forecast.tolist(),
        'historical_volatility': conditional_vol.tolist()[-252:],
        'aic': float(fitted.aic),
    }


def volatility_term_structure(prices, horizons=[1, 5, 10, 21, 63, 126, 252]):
    """
    Calculate volatility term structure (like VIX term structure)
    
    Parameters:
    -----------
    prices : pd.Series
        Historical prices
    horizons : list
        List of forecast horizons in days
        
    Returns:
    --------
    dict with term structure data
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = prices.pct_change().dropna() * 100
    
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')
    
    max_horizon = max(horizons)
    forecast = fitted.forecast(horizon=max_horizon)
    variance_forecast = forecast.variance.iloc[-1].values
    
    term_structure = {}
    for h in horizons:
        avg_var = np.mean(variance_forecast[:h])
        annualized_vol = np.sqrt(avg_var) * np.sqrt(252) / 100 * 100  # As percentage
        term_structure[f'{h}d'] = float(annualized_vol)
    
    return {
        'term_structure': term_structure,
        'horizons_days': horizons,
        'is_contango': term_structure.get('252d', 0) > term_structure.get('21d', 0),
        'slope': float((term_structure.get('252d', 0) - term_structure.get('21d', 0)) / term_structure.get('21d', 1) * 100)
    }


def realized_vs_implied(prices, window=21):
    """
    Compare realized volatility with GARCH implied volatility
    
    Returns:
    --------
    DataFrame with realized and implied vol comparison
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = prices.pct_change().dropna()
    
    # Realized volatility (rolling)
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    # GARCH implied volatility
    returns_scaled = returns * 100
    model = arch_model(returns_scaled, vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')
    implied_vol = fitted.conditional_volatility * np.sqrt(252) / 100 * 100
    
    # Align indices
    realized_vol = realized_vol.iloc[window:]
    implied_vol = implied_vol.iloc[window:]
    
    # Volatility risk premium (VRP)
    vrp = implied_vol.values - realized_vol.values
    
    return {
        'realized_vol': realized_vol.tolist()[-252:],
        'implied_vol': implied_vol.tolist()[-252:],
        'vrp_mean': float(np.nanmean(vrp)),
        'vrp_current': float(vrp[-1]) if len(vrp) > 0 else 0,
        'correlation': float(np.corrcoef(realized_vol.dropna(), implied_vol.iloc[:len(realized_vol.dropna())].dropna())[0, 1]) if len(realized_vol.dropna()) > 0 else 0
    }
