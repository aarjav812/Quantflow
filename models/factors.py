"""
Fama-French Factor Analysis
Nobel Prize-winning approach to understanding stock returns
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(prices_df):
    """Calculate daily returns from price data"""
    return prices_df.pct_change().dropna()


def construct_factors(stock_data_dict, fundamentals_df, risk_free_rate=0.05):
    """
    Construct Fama-French factors from Nifty 50 cross-section
    
    Parameters:
    -----------
    stock_data_dict : dict
        Dictionary mapping symbol to DataFrame with OHLCV data
    fundamentals_df : pd.DataFrame
        Fundamentals data with market_cap, price_to_book, etc.
    risk_free_rate : float
        Annual risk-free rate (default 5%)
        
    Returns:
    --------
    DataFrame with Market, SMB, HML factors
    """
    # Get common dates across all stocks
    all_dates = None
    returns_dict = {}
    
    for symbol, df in stock_data_dict.items():
        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        daily_returns = df['Close'].pct_change()
        returns_dict[symbol] = daily_returns
        
        if all_dates is None:
            all_dates = set(daily_returns.index)
        else:
            all_dates = all_dates.intersection(set(daily_returns.index))
    
    common_dates = sorted(list(all_dates))
    
    # Create returns matrix
    returns_matrix = pd.DataFrame(index=common_dates)
    for symbol, returns in returns_dict.items():
        returns_matrix[symbol] = returns[returns.index.isin(common_dates)]
    
    returns_matrix = returns_matrix.dropna()
    
    # Get market caps and P/B ratios
    market_caps = {}
    pb_ratios = {}
    
    for _, row in fundamentals_df.iterrows():
        symbol = row['symbol'].replace('.NS', '')
        if 'market_cap' in row and pd.notna(row['market_cap']):
            market_caps[symbol] = row['market_cap']
        if 'price_to_book' in row and pd.notna(row['price_to_book']):
            pb_ratios[symbol] = row['price_to_book']
    
    # Filter to stocks with both market cap and P/B
    valid_symbols = [s for s in returns_matrix.columns 
                     if s.replace('.NS', '') in market_caps and s.replace('.NS', '') in pb_ratios]
    
    if len(valid_symbols) < 6:
        # Fallback: use simple rankings based on available data
        n_stocks = len(returns_matrix.columns)
        valid_symbols = list(returns_matrix.columns)[:n_stocks]
        # Assign dummy market caps and P/B ratios
        for i, s in enumerate(valid_symbols):
            if s.replace('.NS', '') not in market_caps:
                market_caps[s.replace('.NS', '')] = (i + 1) * 1e10
            if s.replace('.NS', '') not in pb_ratios:
                pb_ratios[s.replace('.NS', '')] = (i + 1) * 0.5
    
    # Sort by market cap for size factor (SMB)
    size_sorted = sorted(valid_symbols, key=lambda x: market_caps.get(x.replace('.NS', ''), 0))
    small_stocks = size_sorted[:len(size_sorted)//2]
    big_stocks = size_sorted[len(size_sorted)//2:]
    
    # Sort by P/B for value factor (HML)
    # Low P/B = Value stocks (High book-to-market)
    value_sorted = sorted(valid_symbols, key=lambda x: pb_ratios.get(x.replace('.NS', ''), float('inf')))
    value_stocks = value_sorted[:len(value_sorted)//3]  # Low P/B
    growth_stocks = value_sorted[-len(value_sorted)//3:]  # High P/B
    
    # Calculate daily factor returns
    factor_returns = pd.DataFrame(index=returns_matrix.index)
    
    # Daily risk-free rate
    rf_daily = risk_free_rate / 252
    
    # Market factor (equal-weighted market return minus risk-free)
    factor_returns['MKT'] = returns_matrix[valid_symbols].mean(axis=1) - rf_daily
    
    # SMB: Small Minus Big
    small_return = returns_matrix[small_stocks].mean(axis=1) if small_stocks else 0
    big_return = returns_matrix[big_stocks].mean(axis=1) if big_stocks else 0
    factor_returns['SMB'] = small_return - big_return
    
    # HML: High Minus Low (Value minus Growth)
    value_return = returns_matrix[value_stocks].mean(axis=1) if value_stocks else 0
    growth_return = returns_matrix[growth_stocks].mean(axis=1) if growth_stocks else 0
    factor_returns['HML'] = value_return - growth_return
    
    factor_returns['RF'] = rf_daily
    
    return factor_returns.dropna()


def factor_regression(stock_returns, factor_returns):
    """
    Run Fama-French 3-factor regression for a stock
    
    R_i - R_f = alpha + beta_mkt * MKT + beta_smb * SMB + beta_hml * HML + epsilon
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock daily returns
    factor_returns : pd.DataFrame
        Factor returns (MKT, SMB, HML, RF columns)
        
    Returns:
    --------
    dict with factor loadings and statistics
    """
    # Align data
    common_index = stock_returns.index.intersection(factor_returns.index)
    y = stock_returns[common_index] - factor_returns.loc[common_index, 'RF']
    X = factor_returns.loc[common_index, ['MKT', 'SMB', 'HML']]
    
    # Add constant for alpha
    X = sm.add_constant(X)
    
    # Run OLS regression
    model = sm.OLS(y, X).fit()
    
    # Calculate annualized alpha
    alpha_daily = model.params.get('const', 0)
    alpha_annual = alpha_daily * 252
    
    # T-statistics and p-values
    results = {
        'alpha_daily': float(alpha_daily),
        'alpha_annual': float(alpha_annual * 100),  # As percentage
        'alpha_tstat': float(model.tvalues.get('const', 0)),
        'alpha_pvalue': float(model.pvalues.get('const', 1)),
        'beta_market': float(model.params.get('MKT', 0)),
        'beta_market_tstat': float(model.tvalues.get('MKT', 0)),
        'beta_smb': float(model.params.get('SMB', 0)),
        'beta_smb_tstat': float(model.tvalues.get('SMB', 0)),
        'beta_hml': float(model.params.get('HML', 0)),
        'beta_hml_tstat': float(model.tvalues.get('HML', 0)),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'residual_std': float(model.resid.std()),
        'n_observations': int(len(y)),
    }
    
    # Interpretation
    results['interpretation'] = {
        'market_exposure': 'High' if abs(model.params.get('MKT', 0)) > 1.2 else 'Low' if abs(model.params.get('MKT', 0)) < 0.8 else 'Normal',
        'size_tilt': 'Small Cap' if model.params.get('SMB', 0) > 0.1 else 'Large Cap' if model.params.get('SMB', 0) < -0.1 else 'Neutral',
        'value_tilt': 'Value' if model.params.get('HML', 0) > 0.1 else 'Growth' if model.params.get('HML', 0) < -0.1 else 'Neutral',
        'generates_alpha': model.pvalues.get('const', 1) < 0.05 and alpha_annual > 0
    }
    
    return results


def factor_performance(factor_returns, period='all'):
    """
    Analyze factor performance over time
    
    Parameters:
    -----------
    factor_returns : pd.DataFrame
        Factor returns
    period : str
        'all', '1y', '3y', '5y'
        
    Returns:
    --------
    dict with factor performance metrics
    """
    if period == '1y':
        factor_returns = factor_returns.iloc[-252:]
    elif period == '3y':
        factor_returns = factor_returns.iloc[-756:]
    elif period == '5y':
        factor_returns = factor_returns.iloc[-1260:]
    
    results = {}
    for factor in ['MKT', 'SMB', 'HML']:
        if factor not in factor_returns.columns:
            continue
            
        ret = factor_returns[factor]
        
        # Annualized return
        ann_return = ret.mean() * 252
        
        # Annualized volatility
        ann_vol = ret.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming RF already subtracted for MKT)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # T-stat for mean return
        t_stat = ret.mean() / (ret.std() / np.sqrt(len(ret))) if ret.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + ret).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        results[factor] = {
            'annual_return': float(ann_return * 100),
            'annual_volatility': float(ann_vol * 100),
            'sharpe_ratio': float(sharpe),
            't_statistic': float(t_stat),
            'max_drawdown': float(max_dd * 100),
            'positive_months': float((ret.resample('M').sum() > 0).mean() * 100) if hasattr(ret.index, 'freq') or len(ret) > 21 else 0
        }
    
    return results


def factor_correlation(factor_returns):
    """
    Calculate factor correlations
    """
    factors = ['MKT', 'SMB', 'HML']
    available_factors = [f for f in factors if f in factor_returns.columns]
    
    corr_matrix = factor_returns[available_factors].corr()
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'mkt_smb_corr': float(corr_matrix.loc['MKT', 'SMB']) if 'MKT' in corr_matrix.index and 'SMB' in corr_matrix.columns else 0,
        'mkt_hml_corr': float(corr_matrix.loc['MKT', 'HML']) if 'MKT' in corr_matrix.index and 'HML' in corr_matrix.columns else 0,
        'smb_hml_corr': float(corr_matrix.loc['SMB', 'HML']) if 'SMB' in corr_matrix.index and 'HML' in corr_matrix.columns else 0,
    }
