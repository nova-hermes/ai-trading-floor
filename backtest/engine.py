"""
AI Trading Floor v2 — Backtesting Engine
Supports multiple strategies, walk-forward validation, and Monte Carlo simulation
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

import sys
sys.path.insert(0, '/home/doug/ai-trading-floor')
from config.settings import BACKTEST_CONFIG

console = Console()


@dataclass
class Trade:
    """Represents a single trade in backtest"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    status: str = "open"  # "open", "closed", "stopped"


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary string"""
        return (
            f"Strategy: {self.strategy_name}\n"
            f"Symbol: {self.symbol} ({self.timeframe})\n"
            f"Period: {self.start_date.date()} to {self.end_date.date()}\n"
            f"Total Return: {self.total_return:.2%}\n"
            f"Annual Return: {self.annual_return:.2%}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Win Rate: {self.win_rate:.1%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Total Trades: {self.total_trades}\n"
        )


class Strategy:
    """Base class for trading strategies"""

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Series of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        raise NotImplementedError("Subclass must implement generate_signals")


class RSIMeanReversion(Strategy):
    """RSI Mean Reversion Strategy"""

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Mean Reversion", {
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
        })
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Buy when RSI < oversold, sell when RSI > overbought"""
        close = df['close']

        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[rsi < self.oversold] = 1   # Buy
        signals[rsi > self.overbought] = -1  # Sell

        return signals


class MACrossover(Strategy):
    """Moving Average Crossover Strategy"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("MA Crossover", {
            "fast_period": fast_period,
            "slow_period": slow_period,
        })
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Buy when fast MA crosses above slow MA, sell when crosses below"""
        close = df['close']

        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        # Generate signals on crossover
        signals = pd.Series(0, index=df.index)
        signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1   # Buy
        signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1  # Sell

        return signals


class BollingerBreakout(Strategy):
    """Bollinger Bands Breakout Strategy"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Breakout", {
            "period": period,
            "std_dev": std_dev,
        })
        self.period = period
        self.std_dev = std_dev

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Buy when price breaks above upper band, sell when breaks below lower"""
        close = df['close']

        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = sma + (std * self.std_dev)
        lower = sma - (std * self.std_dev)

        signals = pd.Series(0, index=df.index)
        signals[close > upper] = 1   # Breakout buy
        signals[close < lower] = -1  # Breakdown sell

        return signals


class MACDStrategy(Strategy):
    """MACD Crossover Strategy"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Crossover", {
            "fast": fast,
            "slow": slow,
            "signal": signal,
        })
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Buy on MACD bullish crossover, sell on bearish"""
        close = df['close']

        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period).mean()

        signals = pd.Series(0, index=df.index)
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

        return signals


class MultiIndicatorStrategy(Strategy):
    """Combined strategy using multiple indicators"""

    def __init__(self):
        super().__init__("Multi-Indicator", {})
        self.rsi_strat = RSIMeanReversion()
        self.macd_strat = MACDStrategy()
        self.bb_strat = BollingerBreakout()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Require 2 out of 3 indicators to agree"""
        rsi_signals = self.rsi_strat.generate_signals(df)
        macd_signals = self.macd_strat.generate_signals(df)
        bb_signals = self.bb_strat.generate_signals(df)

        combined = rsi_signals + macd_signals + bb_signals

        signals = pd.Series(0, index=df.index)
        signals[combined >= 2] = 1    # 2+ buy signals
        signals[combined <= -2] = -1  # 2+ sell signals

        return signals


class BacktestEngine:
    """
    Core backtesting engine
    Runs strategies against historical data and calculates performance metrics
    """

    def __init__(self, initial_capital: float = None, commission: float = None, slippage: float = None):
        self.initial_capital = initial_capital or BACKTEST_CONFIG["initial_capital"]
        self.commission = commission or BACKTEST_CONFIG["commission"]
        self.slippage = slippage or BACKTEST_CONFIG["slippage"]

    def run(self, strategy: Strategy, df: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """
        Run a backtest
        
        Args:
            strategy: Strategy instance to test
            df: OHLCV DataFrame
            symbol: Symbol name for reporting
        
        Returns:
            BacktestResult with all metrics
        """
        if len(df) < 50:
            raise ValueError("Not enough data (need at least 50 candles)")

        console.print(f"\n[bold blue]📊 Running Backtest: {strategy.name}[/bold blue]")
        console.print(f"Symbol: {symbol} | Period: {df.index[0]} to {df.index[-1]}")

        # Generate signals
        signals = strategy.generate_signals(df)

        # Simulate trading
        trades = self._simulate_trades(df, signals, symbol)

        # Calculate metrics
        result = self._calculate_metrics(strategy.name, symbol, df, trades)

        # Display results
        self._display_results(result)

        return result

    def _simulate_trades(self, df: pd.DataFrame, signals: pd.Series, symbol: str) -> List[Trade]:
        """Simulate trades based on signals"""
        trades = []
        current_trade = None
        cash = self.initial_capital
        position_qty = 0
        position_entry = 0

        for i in range(len(df)):
            signal = signals.iloc[i]
            price = df['close'].iloc[i]
            timestamp = df.index[i]

            # Open new position
            if signal == 1 and current_trade is None:
                entry_price = price * (1 + self.slippage)
                
                # Use 95% of cash
                trade_cash = cash * 0.95
                quantity = trade_cash / entry_price
                
                # Commission
                commission = trade_cash * self.commission
                cash = cash * 0.05  # Keep 5% as reserve
                
                position_qty = quantity
                position_entry = entry_price

                current_trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    symbol=symbol,
                    side="long",
                    entry_price=entry_price,
                    quantity=quantity,
                    fees=commission,
                )

            # Close position
            elif signal == -1 and current_trade is not None:
                exit_price = price * (1 - self.slippage)
                
                # P&L
                pnl = (exit_price - position_entry) * position_qty
                exit_value = position_qty * exit_price
                commission = exit_value * self.commission
                pnl -= commission
                
                # Update trade
                current_trade.exit_time = timestamp
                current_trade.exit_price = exit_price
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl / (position_entry * position_qty)
                current_trade.fees += commission
                current_trade.status = "closed"

                # Return to cash
                cash = cash + exit_value - commission
                position_qty = 0
                position_entry = 0

                trades.append(current_trade)
                current_trade = None

        # Close any open position at end
        if current_trade is not None:
            exit_price = df['close'].iloc[-1]
            pnl = (exit_price - position_entry) * position_qty
            exit_value = position_qty * exit_price
            commission = exit_value * self.commission
            pnl -= commission
            
            current_trade.exit_time = df.index[-1]
            current_trade.exit_price = exit_price
            current_trade.pnl = pnl
            current_trade.pnl_pct = pnl / (position_entry * position_qty)
            current_trade.fees += commission
            current_trade.status = "closed"
            
            cash = cash + exit_value - commission
            trades.append(current_trade)

        return trades

    def _calculate_metrics(self, strategy_name: str, symbol: str, df: pd.DataFrame, trades: List[Trade]) -> BacktestResult:
        """Calculate performance metrics"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe="1h",
                start_date=df.index[0],
                end_date=df.index[-1],
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0,
                annual_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_trade_duration=0,
                trades=[],
                equity_curve=[self.initial_capital],
                drawdown_curve=[0],
            )

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])

        # P&L metrics
        pnls = [t.pnl for t in trades]
        total_pnl = sum(pnls)
        final_capital = self.initial_capital + total_pnl

        # Win/loss metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Return metrics
        total_return = total_pnl / self.initial_capital
        days = (df.index[-1] - df.index[0]).days
        annual_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0

        # Equity curve
        equity = [self.initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)

        # Drawdown
        peak = equity[0]
        drawdown = [0]
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            drawdown.append(dd)

        max_drawdown = max(drawdown)

        # Sharpe ratio (simplified)
        returns = pd.Series(pnls) / self.initial_capital
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0

        # Average trade duration
        durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        avg_trade_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe="1h",
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            trades=trades,
            equity_curve=equity,
            drawdown_curve=drawdown,
        )

    def _display_results(self, result: BacktestResult):
        """Display backtest results in a nice format"""
        # Summary table
        table = Table(title=f"📊 Backtest Results: {result.strategy_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Symbol", result.symbol)
        table.add_row("Period", f"{result.start_date.date()} to {result.end_date.date()}")
        table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
        table.add_row("Final Capital", f"${result.final_capital:,.2f}")

        # Color code returns
        return_color = "green" if result.total_return > 0 else "red"
        table.add_row("Total Return", f"[{return_color}]{result.total_return:.2%}[/{return_color}]")
        table.add_row("Annual Return", f"[{return_color}]{result.annual_return:.2%}[/{return_color}]")

        table.add_row("Max Drawdown", f"[red]{result.max_drawdown:.2%}[/red]")
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
        table.add_row("Total Trades", str(result.total_trades))
        table.add_row("Winning Trades", f"[green]{result.winning_trades}[/green]")
        table.add_row("Losing Trades", f"[red]{result.losing_trades}[/red]")
        table.add_row("Avg Win", f"[green]${result.avg_win:,.2f}[/green]")
        table.add_row("Avg Loss", f"[red]${result.avg_loss:,.2f}[/red]")
        table.add_row("Largest Win", f"[green]${result.largest_win:,.2f}[/green]")
        table.add_row("Largest Loss", f"[red]${result.largest_loss:,.2f}[/red]")
        table.add_row("Avg Trade Duration", f"{result.avg_trade_duration:.1f} hours")

        console.print(table)

        # Trade summary
        if result.trades:
            console.print(f"\n[bold]📝 Trade History (last 10):[/bold]")
            trade_table = Table()
            trade_table.add_column("Entry Time")
            trade_table.add_column("Side")
            trade_table.add_column("Entry Price")
            trade_table.add_column("Exit Price")
            trade_table.add_column("P&L")

            for trade in result.trades[-10:]:
                pnl_color = "green" if trade.pnl > 0 else "red"
                trade_table.add_row(
                    str(trade.entry_time),
                    trade.side,
                    f"${trade.entry_price:,.2f}",
                    f"${trade.exit_price:,.2f}",
                    f"[{pnl_color}]${trade.pnl:,.2f}[/{pnl_color}]",
                )

            console.print(trade_table)


class WalkForwardValidator:
    """
    Walk-forward validation for strategies
    Splits data into train/test windows and evaluates out-of-sample performance
    """

    def __init__(self, train_pct: float = 0.7, n_splits: int = 5):
        self.train_pct = train_pct
        self.n_splits = n_splits
        self.engine = BacktestEngine()

    def validate(self, strategy: Strategy, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Run walk-forward validation
        
        Args:
            strategy: Strategy to validate
            df: Full OHLCV DataFrame
            symbol: Symbol name
        
        Returns:
            Dictionary with validation results
        """
        console.print(f"\n[bold blue]🔄 Walk-Forward Validation: {strategy.name}[/bold blue]")
        console.print(f"Total data: {len(df)} candles")
        console.print(f"Train/Test split: {self.train_pct:.0%}/{1-self.train_pct:.0%}")
        console.print(f"Number of splits: {self.n_splits}")

        # Calculate window sizes
        total_len = len(df)
        window_size = total_len // self.n_splits
        train_size = int(window_size * self.train_pct)
        test_size = window_size - train_size

        results = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Running splits...", total=self.n_splits)

            for i in range(self.n_splits):
                # Calculate window boundaries
                start = i * window_size
                train_end = start + train_size
                test_end = min(train_end + test_size, total_len)

                if test_end > total_len:
                    break

                # Split data
                train_data = df.iloc[start:train_end]
                test_data = df.iloc[train_end:test_end]

                # Run backtest on test data (out-of-sample)
                try:
                    result = self.engine.run(strategy, test_data, symbol)
                    results.append({
                        "split": i + 1,
                        "train_period": f"{train_data.index[0].date()} to {train_data.index[-1].date()}",
                        "test_period": f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "total_trades": result.total_trades,
                    })
                except Exception as e:
                    console.print(f"[yellow]Split {i+1} failed: {e}[/yellow]")

                progress.update(task, advance=1)

        # Aggregate results
        if results:
            avg_return = np.mean([r["total_return"] for r in results])
            avg_sharpe = np.mean([r["sharpe_ratio"] for r in results])
            avg_drawdown = np.mean([r["max_drawdown"] for r in results])
            avg_win_rate = np.mean([r["win_rate"] for r in results])

            # Display results
            table = Table(title="📊 Walk-Forward Validation Results")
            table.add_column("Split")
            table.add_column("Test Period")
            table.add_column("Return")
            table.add_column("Sharpe")
            table.add_column("Max DD")
            table.add_column("Win Rate")
            table.add_column("Trades")

            for r in results:
                ret_color = "green" if r["total_return"] > 0 else "red"
                table.add_row(
                    str(r["split"]),
                    r["test_period"],
                    f"[{ret_color}]{r['total_return']:.2%}[/{ret_color}]",
                    f"{r['sharpe_ratio']:.2f}",
                    f"[red]{r['max_drawdown']:.2%}[/red]",
                    f"{r['win_rate']:.1%}",
                    str(r["total_trades"]),
                )

            console.print(table)

            # Summary
            console.print(Panel(
                f"[bold]Average Results Across {len(results)} Splits:[/bold]\n\n"
                f"Return: [{'green' if avg_return > 0 else 'red'}]{avg_return:.2%}[/{'green' if avg_return > 0 else 'red'}]\n"
                f"Sharpe Ratio: {avg_sharpe:.2f}\n"
                f"Max Drawdown: [red]{avg_drawdown:.2%}[/red]\n"
                f"Win Rate: {avg_win_rate:.1%}\n\n"
                f"{'✅ Strategy shows consistent out-of-sample performance' if avg_return > 0 and avg_sharpe > 0.5 else '⚠️ Strategy may be overfitting or underperforming'}",
                title="Walk-Forward Summary",
            ))

            return {
                "splits": results,
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_drawdown": avg_drawdown,
                "avg_win_rate": avg_win_rate,
                "consistent": avg_return > 0 and avg_sharpe > 0.5,
            }

        return {"splits": [], "consistent": False}


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing
    Randomizes trade order to estimate confidence intervals
    """

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def simulate(self, result: BacktestResult) -> Dict:
        """
        Run Monte Carlo simulation on backtest results
        
        Args:
            result: BacktestResult from a previous backtest
        
        Returns:
            Dictionary with simulation results and confidence intervals
        """
        console.print(f"\n[bold blue]🎲 Monte Carlo Simulation ({self.n_simulations} runs)[/bold blue]")

        if not result.trades:
            console.print("[yellow]No trades to simulate[/yellow]")
            return {}

        # Extract trade P&Ls
        pnls = np.array([t.pnl for t in result.trades])
        initial_capital = result.initial_capital

        # Run simulations
        final_capitals = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Randomly shuffle trade order
            shuffled_pnls = np.random.permutation(pnls)

            # Calculate equity curve
            equity = [initial_capital]
            for pnl in shuffled_pnls:
                equity.append(equity[-1] + pnl)

            final_capitals.append(equity[-1])

            # Calculate max drawdown
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)

        # Calculate statistics
        final_capitals = np.array(final_capitals)
        max_drawdowns = np.array(max_drawdowns)

        returns = (final_capitals - initial_capital) / initial_capital

        # Confidence intervals
        ci_5 = np.percentile(returns, 5)
        ci_25 = np.percentile(returns, 25)
        ci_50 = np.percentile(returns, 50)
        ci_75 = np.percentile(returns, 75)
        ci_95 = np.percentile(returns, 95)

        dd_ci_5 = np.percentile(max_drawdowns, 5)
        dd_ci_50 = np.percentile(max_drawdowns, 50)
        dd_ci_95 = np.percentile(max_drawdowns, 95)

        # Display results
        console.print(Panel(
            f"[bold]Monte Carlo Results ({self.n_simulations} simulations):[/bold]\n\n"
            f"[bold]Return Distribution:[/bold]\n"
            f"  5th percentile: [{'green' if ci_5 > 0 else 'red'}]{ci_5:.2%}[/{'green' if ci_5 > 0 else 'red'}]\n"
            f"  25th percentile: [{'green' if ci_25 > 0 else 'red'}]{ci_25:.2%}[/{'green' if ci_25 > 0 else 'red'}]\n"
            f"  Median (50th): [{'green' if ci_50 > 0 else 'red'}]{ci_50:.2%}[/{'green' if ci_50 > 0 else 'red'}]\n"
            f"  75th percentile: [{'green' if ci_75 > 0 else 'red'}]{ci_75:.2%}[/{'green' if ci_75 > 0 else 'red'}]\n"
            f"  95th percentile: [{'green' if ci_95 > 0 else 'red'}]{ci_95:.2%}[/{'green' if ci_95 > 0 else 'red'}]\n\n"
            f"[bold]Max Drawdown Distribution:[/bold]\n"
            f"  5th percentile (best): [green]{dd_ci_5:.2%}[/green]\n"
            f"  Median: [yellow]{dd_ci_50:.2%}[/yellow]\n"
            f"  95th percentile (worst): [red]{dd_ci_95:.2%}[/red]\n\n"
            f"[bold]Probability Metrics:[/bold]\n"
            f"  Prob of profit: {(returns > 0).mean():.1%}\n"
            f"  Prob of >10% return: {(returns > 0.10).mean():.1%}\n"
            f"  Prob of >20% drawdown: {(max_drawdowns > 0.20).mean():.1%}\n"
            f"  Prob of ruin (>50% DD): {(max_drawdowns > 0.50).mean():.1%}\n",
            title="Monte Carlo Simulation",
        ))

        return {
            "n_simulations": self.n_simulations,
            "return_ci": {
                "5%": ci_5,
                "25%": ci_25,
                "50%": ci_50,
                "75%": ci_75,
                "95%": ci_95,
            },
            "drawdown_ci": {
                "5%": dd_ci_5,
                "50%": dd_ci_50,
                "95%": dd_ci_95,
            },
            "prob_profit": float((returns > 0).mean()),
            "prob_10pct": float((returns > 0.10).mean()),
            "prob_20pct_dd": float((max_drawdowns > 0.20).mean()),
            "prob_ruin": float((max_drawdowns > 0.50).mean()),
        }


# Convenience function
def run_backtest(strategy_name: str, symbol: str, days: int = 90, timeframe: str = "1h"):
    """
    Convenience function to run a complete backtest
    
    Args:
        strategy_name: Name of strategy to test
        symbol: Trading pair or stock ticker
        days: Number of days of historical data
        timeframe: Data timeframe
    """
    from data.provider import data_provider

    # Get data
    is_crypto = "/" in symbol
    if is_crypto:
        df = data_provider.get_crypto_data(symbol, days=days, timeframe=timeframe)
    else:
        df = data_provider.get_stock_data(symbol, days=days, interval=timeframe)

    if df is None or df.empty:
        console.print(f"[red]Failed to get data for {symbol}[/red]")
        return None

    # Select strategy
    strategies = {
        "rsi": RSIMeanReversion(),
        "ma_crossover": MACrossover(),
        "bollinger": BollingerBreakout(),
        "macd": MACDStrategy(),
        "multi": MultiIndicatorStrategy(),
    }

    strategy = strategies.get(strategy_name)
    if not strategy:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        console.print(f"Available: {', '.join(strategies.keys())}")
        return None

    # Run backtest
    engine = BacktestEngine()
    result = engine.run(strategy, df, symbol)

    return result


# Singleton instances
backtest_engine = BacktestEngine()
walk_forward = WalkForwardValidator()
monte_carlo = MonteCarloSimulator()
