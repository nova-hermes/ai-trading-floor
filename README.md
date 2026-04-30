# 🏛️ AI Trading Floor v2

AI-powered trading system with multi-agent debate, real-time data, and automated execution.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AI TRADING FLOOR v2                    │
│                                                         │
│  DATA LAYER                                             │
│  ├── OpenBB (90+ providers: stocks, crypto, forex, macro)│
│  ├── CCXT (20+ exchanges: Binance, Coinbase, etc.)      │
│  └── yfinance (backup for stocks)                       │
│                                                         │
│  ANALYSIS LAYER                                         │
│  ├── 30+ Technical Indicators (RSI, MACD, BB, etc.)    │
│  ├── Pattern Recognition (Engulfing, Hammer, etc.)     │
│  └── Fundamental Data (P/E, Market Cap, etc.)          │
│                                                         │
│  AI AGENT LAYER                                         │
│  ├── Warren Buffett (Value Investing)                   │
│  ├── Ray Dalio (Macro/Risk Parity)                      │
│  ├── Cathie Wood (Growth/Disruption)                    │
│  ├── Technical Analyst (Chart Patterns)                 │
│  └── Quant Analyst (Statistical Arbitrage)              │
│                                                         │
│  EXECUTION LAYER                                        │
│  ├── CCXT (20+ exchanges)                               │
│  ├── Paper Trading Engine                               │
│  ├── Position Tracking (SQLite)                         │
│  └── Risk Management (Stop Loss, Take Profit)           │
│                                                         │
│  BACKTEST LAYER (Coming Soon)                           │
│  ├── Walk-Forward Validation                            │
│  ├── Monte Carlo Simulation                             │
│  └── Historical Performance Tracking                    │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd /home/doug/ai-trading-floor

# Analyze a symbol
python3 main.py analyze BTC/USDT
python3 main.py analyze AAPL

# Scan market for opportunities
python3 main.py scan

# Check portfolio
python3 main.py portfolio
```

## How It Works

1. **Data Collection**: Fetches OHLCV data from OpenBB (stocks), CCXT (crypto), or yfinance (backup)
2. **Technical Analysis**: Computes 30+ indicators (RSI, MACD, Bollinger Bands, etc.) and detects patterns
3. **AI Debate**: 5 AI agents with distinct investor personas debate the trade opportunity
4. **Consensus**: Weighted vote based on agent confidence levels
5. **Execution**: Paper trades by default, with stop loss and take profit management
6. **Monitoring**: Tracks positions and auto-closes on SL/TP triggers

## Configuration

Edit `config/settings.py` to customize:

- **LLM_CONFIG**: Model provider settings (default: Xiaomi mimo-v2.5)
- **TRADING_CONFIG**: Default symbols, risk parameters, timeframes
- **AGENT_CONFIG**: Agent personas and debate rounds
- **BACKTEST_CONFIG**: Initial capital, commission, slippage

## API Keys

Copy `.env.example` to `.env` and fill in:

```bash
# Required (for AI agents)
XIAOMI_API_KEY=your_key_here

# Optional (for live trading)
BINANCE_API_KEY=your_key_here
BINANCE_SECRET=your_secret_here
```

## Project Structure

```
ai-trading-floor/
├── config/
│   └── settings.py          # Configuration
├── data/
│   ├── provider.py           # Data provider (OpenBB/CCXT/yfinance)
│   ├── technical.py          # Technical analysis engine
│   └── trades.db             # SQLite database
├── agents/
│   └── trading_floor.py      # AI agent debate system
├── execution/
│   └── engine.py             # Trade execution engine
├── backtest/                 # Coming soon
├── strategies/               # Custom strategies
├── tests/                    # Unit tests
├── main.py                   # CLI entry point
└── requirements.txt          # Dependencies
```

## Key Differences from v1

| Feature | v1 (stock-analysis) | v2 (ai-trading-floor) |
|---------|--------------------|-----------------------|
| Data Sources | yfinance only | OpenBB (90+) + CCXT (20+) |
| Technical Analysis | Basic (5 indicators) | Comprehensive (30+ indicators) |
| AI Agents | 1 LLM roleplaying 7 characters | 5 separate LLM calls with distinct personas |
| Execution | Paper only, basic | Paper + Live, with risk management |
| Backtesting | Single strategy | 7 engines (coming soon) |
| Database | None | SQLite for trade history |

## Upcoming Features

- [ ] TradingView MCP integration (live chart reading)
- [ ] Qlib integration (ML-based alpha discovery)
- [ ] Backtesting engine (walk-forward, Monte Carlo)
- [ ] Telegram alerts for trade signals
- [ ] Web dashboard for portfolio monitoring
- [ ] Dune Analytics MCP (on-chain data)
- [ ] Polymarket MCP (prediction market odds)

## Credits

Built with:
- [OpenBB](https://openbb.co/) — Open-source Bloomberg alternative
- [CCXT](https://ccxt.com/) — Unified crypto exchange API
- [OpenAI](https://openai.com/) — LLM API for AI agents
