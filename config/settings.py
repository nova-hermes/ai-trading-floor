# AI Trading Floor v2 — Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# === LLM Configuration ===
LLM_CONFIG = {
    "provider": "xiaomi",
    "base_url": os.getenv("XIAOMI_BASE_URL", "https://api.xiaomimimo.com/v1"),
    "api_key": os.getenv("XIAOMI_API_KEY"),
    "model": "mimo-v2.5",  # Free with Xiaomi credits
    "temperature": 0.7,
    "max_tokens": 2000,
}

# === Exchange Configuration ===
EXCHANGE_CONFIG = {
    "exchange": "binance",
    "api_key": os.getenv("BINANCE_API_KEY", ""),
    "secret": os.getenv("BINANCE_SECRET", ""),
    "sandbox": True,  # Paper trading by default
    "options": {
        "defaultType": "spot",
        "adjustForTimeDifference": True,
    }
}

# === Data Provider Configuration ===
DATA_CONFIG = {
    "openbb": {
        "enabled": True,
        # OpenBB is free, no API key needed for basic usage
    },
    "yfinance": {
        "enabled": True,  # Backup for stocks
    },
    "ccxt": {
        "enabled": True,  # For crypto
    }
}

# === Trading Configuration ===
TRADING_CONFIG = {
    "default_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "stock_symbols": ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
    "timeframe": "1h",
    "lookback_days": 30,
    "risk_per_trade": 0.02,  # 2% of portfolio
    "max_open_positions": 5,
    "stop_loss_pct": 0.05,  # 5% stop loss
    "take_profit_pct": 0.10,  # 10% take profit
}

# === Agent Configuration ===
AGENT_CONFIG = {
    "num_rounds": 3,  # Debate rounds
    "consensus_threshold": 0.6,  # 60% agreement needed
    "personas": [
        {
            "name": "Warren Buffett",
            "style": "value_investing",
            "focus": "fundamentals, moats, long-term value",
            "risk_tolerance": "low",
        },
        {
            "name": "Ray Dalio",
            "style": "macro",
            "focus": "economic cycles, all-weather portfolio, risk parity",
            "risk_tolerance": "medium",
        },
        {
            "name": "Cathie Wood",
            "style": "growth",
            "focus": "disruptive innovation, tech, exponential growth",
            "risk_tolerance": "high",
        },
        {
            "name": "Technical Analyst",
            "style": "technical",
            "focus": "chart patterns, support/resistance, volume, momentum",
            "risk_tolerance": "medium",
        },
        {
            "name": "Quant Analyst",
            "style": "quantitative",
            "focus": "statistical arbitrage, mean reversion, factor models",
            "risk_tolerance": "medium",
        },
    ]
}

# === Backtest Configuration ===
BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "commission": 0.001,  # 0.1%
    "slippage": 0.0005,  # 0.05%
}

# === Database ===
DB_PATH = ROOT_DIR / "data" / "trades.db"
