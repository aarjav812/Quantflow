"""
Backtesting Engine
Event-driven backtesting with transaction costs and slippage
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy evaluation
    """
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001, slippage=0.0005):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        transaction_cost : float
            Transaction cost as fraction of trade value
        slippage : float
            Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.reset()
    
    def reset(self):
        """Reset engine state"""
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_values = []
        self.dates = []
    
    def get_portfolio_value(self, current_prices):
        """Calculate current portfolio value"""
        positions_value = sum(
            self.positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + positions_value
    
    def execute_trade(self, symbol, quantity, price, date):
        """
        Execute a trade with transaction costs and slippage
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
        quantity : int
            Positive for buy, negative for sell
        price : float
            Current market price
        date : datetime
            Trade date
        """
        if quantity == 0:
            return
        
        # Apply slippage
        if quantity > 0:  # Buy - price goes up
            exec_price = price * (1 + self.slippage)
        else:  # Sell - price goes down
            exec_price = price * (1 - self.slippage)
        
        trade_value = abs(quantity * exec_price)
        cost = trade_value * self.transaction_cost
        
        if quantity > 0:  # Buy
            total_cost = trade_value + cost
            if total_cost > self.cash:
                # Adjust quantity to available cash
                quantity = int((self.cash - cost) / exec_price)
                trade_value = quantity * exec_price
                cost = trade_value * self.transaction_cost
                total_cost = trade_value + cost
            
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        else:  # Sell
            current_position = self.positions.get(symbol, 0)
            quantity = max(quantity, -current_position)  # Can't sell more than owned
            
            if quantity < 0:
                self.cash += trade_value - cost
                self.positions[symbol] = current_position + quantity
        
        if quantity != 0:
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'quantity': quantity,
                'price': exec_price,
                'cost': cost,
                'type': 'BUY' if quantity > 0 else 'SELL'
            })
    
    def run_backtest(self, prices_dict, signals_func, start_date=None, end_date=None):
        """
        Run backtest with given signal generation function
        
        Parameters:
        -----------
        prices_dict : dict
            Dictionary mapping symbol to DataFrame with OHLCV data
        signals_func : callable
            Function that generates signals given current state
            signals_func(prices_dict, date, positions, cash) -> dict of {symbol: quantity}
        start_date : datetime, optional
            Backtest start date
        end_date : datetime, optional
            Backtest end date
            
        Returns:
        --------
        dict with backtest results
        """
        self.reset()
        
        # Get all unique dates
        all_dates = set()
        for symbol, df in prices_dict.items():
            dates = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
            all_dates.update(dates)
        
        all_dates = sorted(all_dates)
        
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date)]
        
        # Run simulation
        for date in all_dates:
            # Get current prices
            current_prices = {}
            for symbol, df in prices_dict.items():
                df_date = df[pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index) == date]
                if len(df_date) > 0:
                    current_prices[symbol] = float(df_date['Close'].iloc[0])
            
            if not current_prices:
                continue
            
            # Generate signals
            signals = signals_func(prices_dict, date, self.positions.copy(), self.cash)
            
            # Execute trades
            for symbol, quantity in signals.items():
                if symbol in current_prices:
                    self.execute_trade(symbol, quantity, current_prices[symbol], date)
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_prices)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        return self.calculate_performance()
    
    def calculate_performance(self):
        """Calculate backtest performance metrics"""
        if len(self.portfolio_values) < 2:
            return {'error': 'Insufficient data for backtest'}
        
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        
        # Total return
        total_return = (portfolio_series.iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        n_days = len(returns)
        ann_return = ((1 + total_return/100) ** (252/n_days) - 1) * 100 if n_days > 0 else 0
        
        # Volatility
        ann_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cumulative = portfolio_series / portfolio_series.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Win rate
        winning_trades = [t for t in self.trades if t['type'] == 'SELL']
        # Simplified win rate calculation
        positive_days = (returns > 0).sum()
        win_rate = positive_days / len(returns) * 100 if len(returns) > 0 else 0
        
        # Total costs
        total_costs = sum(t['cost'] for t in self.trades)
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': float(portfolio_series.iloc[-1]),
            'total_return': float(total_return),
            'annualized_return': float(ann_return),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'total_trades': len(self.trades),
            'total_transaction_costs': float(total_costs),
            'portfolio_values': self.portfolio_values,
            'dates': [str(d) for d in self.dates],
            'trades': self.trades[-50:]  # Last 50 trades
        }


# Pre-built strategies

def momentum_strategy(prices_dict, date, positions, cash, lookback=20, top_n=5):
    """
    Momentum strategy: Buy top performing stocks over lookback period
    """
    signals = {}
    returns = {}
    
    for symbol, df in prices_dict.items():
        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] <= date]
        
        if len(df) >= lookback:
            ret = df['Close'].iloc[-1] / df['Close'].iloc[-lookback] - 1
            returns[symbol] = ret
    
    # Get top N performers
    sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, r in sorted_returns[:top_n] if r > 0]
    
    # Equal weight allocation
    if top_symbols:
        weight = 1.0 / len(top_symbols)
        capital_per_stock = cash * weight * 0.95  # Keep some cash buffer
        
        for symbol in top_symbols:
            if symbol in prices_dict:
                df = prices_dict[symbol]
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[df['Date'] <= date]
                if len(df) > 0:
                    price = df['Close'].iloc[-1]
                    current_value = positions.get(symbol, 0) * price
                    target_quantity = int(capital_per_stock / price)
                    signals[symbol] = target_quantity - positions.get(symbol, 0)
    
    # Sell positions not in top N
    for symbol in positions:
        if symbol not in top_symbols and positions[symbol] > 0:
            signals[symbol] = -positions[symbol]
    
    return signals


def mean_reversion_strategy(prices_dict, date, positions, cash, lookback=20, z_threshold=2):
    """
    Mean reversion strategy: Buy oversold stocks, sell overbought
    """
    signals = {}
    
    for symbol, df in prices_dict.items():
        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] <= date]
        
        if len(df) >= lookback:
            prices = df['Close'].iloc[-lookback:]
            mean = prices.mean()
            std = prices.std()
            current = prices.iloc[-1]
            
            if std > 0:
                z_score = (current - mean) / std
                
                current_pos = positions.get(symbol, 0)
                price = current
                
                if z_score < -z_threshold and current_pos == 0:
                    # Oversold - buy
                    quantity = int(cash * 0.1 / price)  # 10% of cash per stock
                    signals[symbol] = quantity
                
                elif z_score > z_threshold and current_pos > 0:
                    # Overbought - sell
                    signals[symbol] = -current_pos
                
                elif abs(z_score) < 0.5 and current_pos > 0:
                    # Return to mean - close position
                    signals[symbol] = -current_pos
    
    return signals


def buy_and_hold_strategy(prices_dict, date, positions, cash, symbols=None):
    """
    Buy and hold strategy: Equal weight across specified symbols
    """
    signals = {}
    
    if symbols is None:
        symbols = list(prices_dict.keys())[:10]  # Top 10 by default
    
    if not positions:  # Only buy on first day
        weight = 1.0 / len(symbols)
        capital_per_stock = cash * weight * 0.95
        
        for symbol in symbols:
            if symbol in prices_dict:
                df = prices_dict[symbol]
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df[df['Date'] <= date]
                if len(df) > 0:
                    price = df['Close'].iloc[-1]
                    quantity = int(capital_per_stock / price)
                    signals[symbol] = quantity
    
    return signals


def run_strategy_backtest(prices_dict, strategy='momentum', **kwargs):
    """
    Run backtest with predefined strategy
    
    Parameters:
    -----------
    prices_dict : dict
        Dictionary mapping symbol to DataFrame
    strategy : str
        'momentum', 'mean_reversion', or 'buy_and_hold'
    **kwargs : dict
        Additional parameters for the strategy
        
    Returns:
    --------
    dict with backtest results
    """
    engine = BacktestEngine(
        initial_capital=kwargs.get('initial_capital', 1000000),
        transaction_cost=kwargs.get('transaction_cost', 0.001),
        slippage=kwargs.get('slippage', 0.0005)
    )
    
    if strategy == 'momentum':
        lookback = kwargs.get('lookback', 20)
        top_n = kwargs.get('top_n', 5)
        signal_func = lambda pd, d, p, c: momentum_strategy(pd, d, p, c, lookback, top_n)
    
    elif strategy == 'mean_reversion':
        lookback = kwargs.get('lookback', 20)
        z_threshold = kwargs.get('z_threshold', 2)
        signal_func = lambda pd, d, p, c: mean_reversion_strategy(pd, d, p, c, lookback, z_threshold)
    
    elif strategy == 'buy_and_hold':
        symbols = kwargs.get('symbols', None)
        signal_func = lambda pd, d, p, c: buy_and_hold_strategy(pd, d, p, c, symbols)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return engine.run_backtest(
        prices_dict, 
        signal_func,
        start_date=kwargs.get('start_date'),
        end_date=kwargs.get('end_date')
    )
