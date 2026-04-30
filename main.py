#!/usr/bin/env python3
"""
AI Trading Floor v2 — CLI
Usage:
    python main.py analyze BTC/USDT
    python main.py analyze AAPL --execute
    python main.py scan
    python main.py backtest BTC/USDT --strategy rsi --days 90
    python main.py ml BTC/USDT --days 180
    python main.py portfolio
    python main.py dashboard
"""
import sys
sys.path.insert(0, '/home/doug/ai-trading-floor')

import click
from rich.console import Console
from rich.prompt import Confirm

console = Console()


@click.group()
def cli():
    """AI Trading Floor v2 — AI-Powered Trading System"""
    pass


@cli.command()
@click.argument("symbol")
@click.option("--execute", "-e", is_flag=True, help="Execute the trade (paper)")
def analyze(symbol, execute):
    """Analyze a single symbol with 5 AI agents"""
    from data.provider import data_provider
    from data.technical import tech_analyzer
    from agents.trading_floor import trading_floor

    is_crypto = "/" in symbol
    if is_crypto:
        df = data_provider.get_crypto_data(symbol, days=30, timeframe="1h")
    else:
        df = data_provider.get_stock_data(symbol, days=30, interval="1h")

    if df is None or df.empty:
        console.print(f"[red]No data for {symbol}[/red]")
        return

    technicals = tech_analyzer.analyze(df)
    context = {
        "symbol": symbol,
        "current_price": technicals["price"]["current"],
        "technicals": technicals,
        "recent_performance": technicals["price"],
        "fundamentals": {},
        "news": [],
    }

    decision = trading_floor.analyze_and_decide(context)

    if execute and decision.action != "HOLD":
        if Confirm.ask(f"Execute {decision.action} {symbol}?"):
            from execution.engine import execution_engine
            position = execution_engine.execute_decision(decision)
            if position:
                console.print(f"[green]✅ Position: {position.id}[/green]")


@cli.command()
def scan():
    """Scan market for trading opportunities"""
    from data.provider import data_provider
    from data.technical import tech_analyzer
    from agents.trading_floor import trading_floor

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    for sym in symbols:
        try:
            df = data_provider.get_crypto_data(sym, days=30, timeframe="1h")
            if df is None:
                continue
            technicals = tech_analyzer.analyze(df)
            context = {"symbol": sym, "current_price": technicals["price"]["current"],
                       "technicals": technicals, "recent_performance": technicals["price"],
                       "fundamentals": {}, "news": []}
            decision = trading_floor.analyze_and_decide(context)
            console.print(f"\n{sym}: {decision.action} ({decision.confidence:.0%})")
        except Exception as e:
            console.print(f"[red]{sym}: Error — {e}[/red]")


@cli.command()
@click.argument("symbol")
@click.option("--strategy", "-s", default="rsi",
              type=click.Choice(["rsi", "ma_crossover", "bollinger", "macd", "multi"]))
@click.option("--days", "-d", default=90)
def backtest(symbol, strategy, days):
    """Backtest a strategy on historical data"""
    from backtest.engine import (BacktestEngine, RSIMeanReversion, MACrossover,
                                  BollingerBreakout, MACDStrategy, MultiIndicatorStrategy)
    from data.provider import data_provider

    strategies = {"rsi": RSIMeanReversion, "ma_crossover": MACrossover,
                  "bollinger": BollingerBreakout, "macd": MACDStrategy,
                  "multi": MultiIndicatorStrategy}

    is_crypto = "/" in symbol
    if is_crypto:
        df = data_provider.get_crypto_data(symbol, days=days, timeframe="1h")
    else:
        df = data_provider.get_stock_data(symbol, days=days, interval="1h")

    if df is None or df.empty:
        console.print(f"[red]No data for {symbol}[/red]")
        return

    engine = BacktestEngine()
    engine.run(strategies[strategy](), df, symbol)


@cli.command()
@click.argument("symbol")
@click.option("--days", "-d", default=180)
def ml(symbol, days):
    """Train ML models and get prediction"""
    from data.ml_alpha import ml_engine
    from data.provider import data_provider

    is_crypto = "/" in symbol
    if is_crypto:
        df = data_provider.get_crypto_data(symbol, days=days, timeframe="1h")
    else:
        df = data_provider.get_stock_data(symbol, days=days, interval="1h")

    if df is None or df.empty:
        console.print(f"[red]No data for {symbol}[/red]")
        return

    ml_engine.train(df, symbol)
    signal = ml_engine.predict(df, symbol)
    if signal:
        ml_engine.display_prediction(signal)


@cli.command()
def portfolio():
    """Show portfolio summary"""
    from execution.engine import execution_engine
    execution_engine.display_portfolio()


@cli.command()
def dashboard():
    """Start web dashboard"""
    from dashboard.app import app
    console.print("[bold blue]🏛️ Starting AI Trading Floor Dashboard...[/bold blue]")
    console.print("[bold]Open http://localhost:5000 in your browser[/bold]")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    cli()
