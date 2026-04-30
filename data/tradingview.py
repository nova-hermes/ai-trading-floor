"""
AI Trading Floor v2 — TradingView Integration
Uses TradingView's technical analysis API for real-time indicators
No TradingView subscription needed for basic data
"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class TradingViewSignal:
    """TradingView recommendation signal"""
    symbol: str
    exchange: str
    screener: str
    interval: str
    recommendation: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    buy_count: int
    sell_count: int
    neutral_count: int
    indicators: Dict[str, str]


class TradingViewAnalyzer:
    """
    TradingView Technical Analysis integration
    Uses tradingview_ta library for real-time indicators
    """

    def __init__(self):
        self._ta = None
        self._initialize()

    def _initialize(self):
        """Initialize TradingView TA"""
        try:
            from tradingview_ta import TA_Handler, Interval
            self._ta = TA_Handler
            self._Interval = Interval
            console.print("[green]✅ TradingView TA initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ TradingView TA failed: {e}[/yellow]")

    def get_analysis(
        self,
        symbol: str,
        exchange: str = "BINANCE",
        screener: str = "crypto",
        interval: str = "1h"
    ) -> Optional[TradingViewSignal]:
        """
        Get TradingView technical analysis for a symbol
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT") or stock ticker (e.g., "AAPL")
            exchange: Exchange name (BINANCE, COINBASE, NASDAQ, NYSE, etc.)
            screener: "crypto" or "america" (for US stocks)
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1W, 1M)
        
        Returns:
            TradingViewSignal with recommendation
        """
        if not self._ta:
            console.print("[red]TradingView TA not initialized[/red]")
            return None

        try:
            # Map interval strings
            interval_map = {
                "1m": self._Interval.INTERVAL_1_MINUTE,
                "5m": self._Interval.INTERVAL_5_MINUTES,
                "15m": self._Interval.INTERVAL_15_MINUTES,
                "1h": self._Interval.INTERVAL_1_HOUR,
                "4h": self._Interval.INTERVAL_4_HOURS,
                "1d": self._Interval.INTERVAL_1_DAY,
                "1W": self._Interval.INTERVAL_1_WEEK,
                "1M": self._Interval.INTERVAL_1_MONTH,
            }

            tv_interval = interval_map.get(interval, self._Interval.INTERVAL_1_HOUR)

            # Create handler
            handler = self._ta(
                symbol=symbol,
                exchange=exchange,
                screener=screener,
                interval=tv_interval,
            )

            # Get analysis
            analysis = handler.get_analysis()

            # Parse recommendation
            summary = analysis.summary
            recommendation = summary.get("RECOMMENDATION", "NEUTRAL")
            buy_count = summary.get("BUY", 0)
            sell_count = summary.get("SELL", 0)
            neutral_count = summary.get("NEUTRAL", 0)

            # Parse individual indicators
            indicators = {}
            if hasattr(analysis, 'indicators'):
                for name, value in analysis.indicators.items():
                    if value is not None:
                        indicators[name] = str(value)

            return TradingViewSignal(
                symbol=symbol,
                exchange=exchange,
                screener=screener,
                interval=interval,
                recommendation=recommendation,
                buy_count=buy_count,
                sell_count=sell_count,
                neutral_count=neutral_count,
                indicators=indicators,
            )

        except Exception as e:
            console.print(f"[red]TradingView analysis failed for {symbol}: {e}[/red]")
            return None

    def get_crypto_analysis(self, symbol: str = "BTCUSDT", interval: str = "1h") -> Optional[TradingViewSignal]:
        """
        Get analysis for a crypto pair on Binance
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Timeframe
        """
        return self.get_analysis(
            symbol=symbol,
            exchange="BINANCE",
            screener="crypto",
            interval=interval,
        )

    def get_stock_analysis(self, symbol: str = "AAPL", interval: str = "1h") -> Optional[TradingViewSignal]:
        """
        Get analysis for a US stock
        
        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT", "NVDA")
            interval: Timeframe
        """
        return self.get_analysis(
            symbol=symbol,
            exchange="NASDAQ",
            screener="america",
            interval=interval,
        )

    def scan_multiple(self, symbols: List[Dict], interval: str = "1h") -> List[TradingViewSignal]:
        """
        Scan multiple symbols for signals
        
        Args:
            symbols: List of dicts with 'symbol', 'exchange', 'screener'
            interval: Timeframe
        
        Returns:
            List of TradingViewSignal objects
        """
        signals = []

        for sym_info in symbols:
            signal = self.get_analysis(
                symbol=sym_info["symbol"],
                exchange=sym_info.get("exchange", "BINANCE"),
                screener=sym_info.get("screener", "crypto"),
                interval=interval,
            )
            if signal:
                signals.append(signal)

        return signals

    def display_signal(self, signal: TradingViewSignal):
        """Display a TradingView signal in a nice format"""
        # Color code recommendation
        rec_colors = {
            "STRONG_BUY": "green",
            "BUY": "green",
            "NEUTRAL": "yellow",
            "SELL": "red",
            "STRONG_SELL": "red",
        }
        color = rec_colors.get(signal.recommendation, "white")

        # Create summary table
        table = Table(title=f"📊 TradingView Analysis: {signal.symbol}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Symbol", signal.symbol)
        table.add_row("Exchange", signal.exchange)
        table.add_row("Interval", signal.interval)
        table.add_row("Recommendation", f"[{color}]{signal.recommendation}[/{color}]")
        table.add_row("Buy Signals", f"[green]{signal.buy_count}[/green]")
        table.add_row("Sell Signals", f"[red]{signal.sell_count}[/red]")
        table.add_row("Neutral Signals", f"[yellow]{signal.neutral_count}[/yellow]")

        console.print(table)

        # Display key indicators if available
        if signal.indicators:
            console.print(f"\n[bold]Key Indicators:[/bold]")
            indicator_table = Table()
            indicator_table.add_column("Indicator")
            indicator_table.add_column("Value")

            # Show important indicators
            important_indicators = [
                "RSI", "MACD.macd", "MACD.signal",
                "BB.upper", "BB.lower", "BB.middle",
                "SMA20", "SMA50", "SMA200",
                "EMA20", "EMA50",
                "Stoch.K", "Stoch.D",
                "ADX", "ATR",
            ]

            for ind in important_indicators:
                if ind in signal.indicators:
                    indicator_table.add_row(ind, signal.indicators[ind])

            console.print(indicator_table)

    def get_recommendation_score(self, signal: TradingViewSignal) -> float:
        """
        Convert recommendation to a numeric score (-1 to 1)
        
        Args:
            signal: TradingViewSignal
        
        Returns:
            Score from -1 (strong sell) to 1 (strong buy)
        """
        score_map = {
            "STRONG_BUY": 1.0,
            "BUY": 0.5,
            "NEUTRAL": 0.0,
            "SELL": -0.5,
            "STRONG_SELL": -1.0,
        }
        return score_map.get(signal.recommendation, 0.0)


# Singleton
tradingview_analyzer = TradingViewAnalyzer()


if __name__ == "__main__":
    # Test the TradingView analyzer
    tv = TradingViewAnalyzer()

    # Test crypto analysis
    print("\n=== Testing Crypto Analysis (BTCUSDT) ===")
    btc_signal = tv.get_crypto_analysis("BTCUSDT", "1h")
    if btc_signal:
        tv.display_signal(btc_signal)

    # Test stock analysis
    print("\n=== Testing Stock Analysis (AAPL) ===")
    aapl_signal = tv.get_stock_analysis("AAPL", "1d")
    if aapl_signal:
        tv.display_signal(aapl_signal)
