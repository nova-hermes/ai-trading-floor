"""
AI Trading Floor v2 — Trade Execution Engine
Handles order placement, position tracking, and risk management via CCXT
"""
import json
import time
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, '/home/doug/ai-trading-floor')
from config.settings import EXCHANGE_CONFIG, TRADING_CONFIG, DB_PATH

console = Console()


@dataclass
class Position:
    """Represents an open trading position"""
    id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    status: str  # "open", "closed", "stopped"
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: str = ""


class ExecutionEngine:
    """
    Handles trade execution across exchanges via CCXT
    Supports paper trading and live trading
    """

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.exchange = None
        self.db_path = DB_PATH
        self._init_exchange()
        self._init_database()

    def _init_exchange(self):
        """Initialize CCXT exchange connection"""
        try:
            import ccxt

            exchange_id = EXCHANGE_CONFIG["exchange"]
            exchange_class = getattr(ccxt, exchange_id)

            config = {
                'enableRateLimit': True,
                'options': EXCHANGE_CONFIG.get("options", {}),
            }

            # Add API keys if available (not needed for paper trading)
            if not self.paper_trading and EXCHANGE_CONFIG.get("api_key"):
                config['apiKey'] = EXCHANGE_CONFIG["api_key"]
                config['secret'] = EXCHANGE_CONFIG["secret"]

            if EXCHANGE_CONFIG.get("sandbox"):
                config['sandbox'] = True

            self.exchange = exchange_class(config)

            mode = "📝 PAPER" if self.paper_trading else "💰 LIVE"
            console.print(f"[green]{mode} Trading — {exchange_id} initialized[/green]")

        except Exception as e:
            console.print(f"[red]Exchange init failed: {e}[/red]")
            self.exchange = None

    def _init_database(self):
        """Initialize SQLite database for trade tracking"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                entry_time TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                pnl REAL DEFAULT 0,
                exit_price REAL DEFAULT 0,
                exit_time TEXT,
                decision_data TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                quantity REAL,
                details TEXT
            )
        ''')

        conn.commit()
        conn.close()
        console.print(f"[green]✅ Database initialized: {self.db_path}[/green]")

    def execute_decision(self, decision) -> Optional[Position]:
        """
        Execute a trading decision from the TradingFloor
        
        Args:
            decision: TradeDecision from agents
        
        Returns:
            Position if trade was executed, None otherwise
        """
        if decision.action == "HOLD":
            console.print("[yellow]⏸️ Decision is HOLD — no trade executed[/yellow]")
            return None

        symbol = decision.symbol
        side = "long" if decision.action == "BUY" else "short"

        console.print(f"\n[bold]⚡ Executing {decision.action} {symbol}[/bold]")
        console.print(f"  Entry: ${decision.entry_price:,.2f}")
        console.print(f"  Stop Loss: ${decision.stop_loss:,.2f}")
        console.print(f"  Take Profit: ${decision.take_profit:,.2f}")
        console.print(f"  Position Size: {decision.position_size_pct:.1%}")

        # Check risk limits
        if not self._check_risk_limits(symbol, decision.position_size_pct):
            console.print("[red]❌ Risk limits exceeded — trade rejected[/red]")
            return None

        # Get current price
        current_price = self._get_current_price(symbol)
        if not current_price:
            console.print("[red]❌ Could not get current price[/red]")
            return None

        # Calculate quantity
        portfolio_value = self._get_portfolio_value()
        position_value = portfolio_value * decision.position_size_pct
        quantity = position_value / current_price

        # Execute the order
        if self.paper_trading:
            order_result = self._paper_order(symbol, side, quantity, current_price)
        else:
            order_result = self._live_order(symbol, side, quantity, current_price)

        if order_result:
            # Create position
            position = Position(
                id=f"{symbol}_{int(time.time())}",
                symbol=symbol,
                side=side,
                entry_price=current_price,
                quantity=quantity,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                entry_time=datetime.now().isoformat(),
                status="open",
            )

            # Save to database
            self._save_position(position, decision)

            console.print(f"[green]✅ {decision.action} order executed![/green]")
            console.print(f"  Quantity: {quantity:.6f}")
            console.print(f"  Value: ${position_value:,.2f}")

            return position

        return None

    def _paper_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Simulate order execution for paper trading"""
        console.print(f"[cyan]📝 Paper {side} order: {quantity:.6f} {symbol} @ ${price:,.2f}[/cyan]")

        # Simulate slight slippage
        slippage = price * 0.0005  # 0.05% slippage
        fill_price = price + slippage if side == "long" else price - slippage

        return {
            "id": f"paper_{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": fill_price,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
        }

    def _live_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[Dict]:
        """Execute real order on exchange"""
        if not self.exchange:
            console.print("[red]No exchange connection[/red]")
            return None

        try:
            # Convert symbol format for CCXT
            # BTC/USDT stays as-is for CCXT

            order_type = 'market'  # Use market orders for immediate execution

            if side == "long":
                result = self.exchange.create_market_buy_order(symbol, quantity)
            else:
                result = self.exchange.create_market_sell_order(symbol, quantity)

            console.print(f"[green]✅ Live order executed: {result['id']}[/green]")
            return result

        except Exception as e:
            console.print(f"[red]Live order failed: {e}[/red]")
            return None

    def check_stop_loss_take_profit(self) -> List[Dict]:
        """
        Check all open positions for stop loss / take profit triggers
        
        Returns:
            List of triggered positions
        """
        triggered = []
        open_positions = self._get_open_positions()

        for pos in open_positions:
            current_price = self._get_current_price(pos.symbol)
            if not current_price:
                continue

            should_close = False
            reason = ""

            if pos.side == "long":
                if current_price <= pos.stop_loss:
                    should_close = True
                    reason = "STOP LOSS"
                elif current_price >= pos.take_profit:
                    should_close = True
                    reason = "TAKE PROFIT"
            else:  # short
                if current_price >= pos.stop_loss:
                    should_close = True
                    reason = "STOP LOSS"
                elif current_price <= pos.take_profit:
                    should_close = True
                    reason = "TAKE PROFIT"

            if should_close:
                # Close the position
                pnl = self._calculate_pnl(pos, current_price)
                self._close_position(pos, current_price, reason)

                triggered.append({
                    "position": pos,
                    "reason": reason,
                    "pnl": pnl,
                    "exit_price": current_price,
                })

                emoji = "🟢" if pnl > 0 else "🔴"
                console.print(f"{emoji} {reason}: {pos.symbol} | P&L: ${pnl:,.2f}")

        return triggered

    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate profit/loss for a position"""
        if position.side == "long":
            return (exit_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - exit_price) * position.quantity

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position and update database"""
        position.status = "closed"
        position.exit_price = exit_price
        position.exit_time = datetime.now().isoformat()
        position.pnl = self._calculate_pnl(position, exit_price)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE positions 
            SET status = ?, exit_price = ?, exit_time = ?, pnl = ?
            WHERE id = ?
        ''', (position.status, position.exit_price, position.exit_time, position.pnl, position.id))

        # Log the trade
        cursor.execute('''
            INSERT INTO trades_log (timestamp, action, symbol, price, quantity, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            f"CLOSE ({reason})",
            position.symbol,
            exit_price,
            position.quantity,
            json.dumps({"pnl": position.pnl, "reason": reason})
        ))

        conn.commit()
        conn.close()

    def _save_position(self, position: Position, decision):
        """Save position to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO positions (id, symbol, side, entry_price, quantity, stop_loss, take_profit, entry_time, status, decision_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.id, position.symbol, position.side,
            position.entry_price, position.quantity,
            position.stop_loss, position.take_profit,
            position.entry_time, position.status,
            json.dumps({
                "confidence": decision.confidence,
                "consensus_score": decision.consensus_score,
                "reasoning": decision.reasoning[:200],
            })
        ))

        cursor.execute('''
            INSERT INTO trades_log (timestamp, action, symbol, price, quantity, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            f"OPEN ({position.side.upper()})",
            position.symbol,
            position.entry_price,
            position.quantity,
            json.dumps({"stop_loss": position.stop_loss, "take_profit": position.take_profit})
        ))

        conn.commit()
        conn.close()

    def _get_open_positions(self) -> List[Position]:
        """Get all open positions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM positions WHERE status = "open"')
        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            positions.append(Position(
                id=row[0], symbol=row[1], side=row[2],
                entry_price=row[3], quantity=row[4],
                stop_loss=row[5], take_profit=row[6],
                entry_time=row[7], status=row[8],
                pnl=row[9], exit_price=row[10], exit_time=row[11] or ""
            ))

        return positions

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from exchange"""
        if not self.exchange:
            return None

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            console.print(f"[yellow]Price fetch failed for {symbol}: {e}[/yellow]")
            return None

    def _get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        if self.paper_trading:
            # Paper trading: use initial capital from config
            from config.settings import BACKTEST_CONFIG
            return BACKTEST_CONFIG["initial_capital"]
        else:
            # Live: fetch from exchange
            try:
                balance = self.exchange.fetch_balance()
                return balance['total']['USDT'] if 'USDT' in balance['total'] else 10000
            except:
                return 10000

    def _check_risk_limits(self, symbol: str, position_size_pct: float) -> bool:
        """Check if trade is within risk limits"""
        # Check max position size
        if position_size_pct > TRADING_CONFIG["risk_per_trade"] * 2.5:
            console.print(f"[yellow]Position size {position_size_pct:.1%} exceeds limit[/yellow]")
            return False

        # Check max open positions
        open_positions = self._get_open_positions()
        if len(open_positions) >= TRADING_CONFIG["max_open_positions"]:
            console.print(f"[yellow]Max open positions ({TRADING_CONFIG['max_open_positions']}) reached[/yellow]")
            return False

        # Check if already have position in this symbol
        for pos in open_positions:
            if pos.symbol == symbol:
                console.print(f"[yellow]Already have open position in {symbol}[/yellow]")
                return False

        return True

    def get_portfolio_summary(self) -> Dict:
        """Get summary of all positions and P&L"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Open positions
        cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "open"')
        open_count = cursor.fetchone()[0]

        # Total P&L
        cursor.execute('SELECT SUM(pnl) FROM positions WHERE status = "closed"')
        total_pnl = cursor.fetchone()[0] or 0

        # Win rate
        cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "closed" AND pnl > 0')
        wins = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "closed"')
        total_closed = cursor.fetchone()[0]

        # Best and worst trades
        cursor.execute('SELECT MAX(pnl) FROM positions WHERE status = "closed"')
        best_trade = cursor.fetchone()[0] or 0
        cursor.execute('SELECT MIN(pnl) FROM positions WHERE status = "closed"')
        worst_trade = cursor.fetchone()[0] or 0

        conn.close()

        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0

        return {
            "open_positions": open_count,
            "total_trades": total_closed,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        }

    def display_portfolio(self):
        """Display portfolio summary in a nice table"""
        summary = self.get_portfolio_summary()
        open_positions = self._get_open_positions()

        # Summary table
        table = Table(title="📊 Portfolio Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Open Positions", str(summary["open_positions"]))
        table.add_row("Total Trades", str(summary["total_trades"]))
        table.add_row("Total P&L", f"${summary['total_pnl']:,.2f}")
        table.add_row("Win Rate", f"{summary['win_rate']:.1f}%")
        table.add_row("Best Trade", f"${summary['best_trade']:,.2f}")
        table.add_row("Worst Trade", f"${summary['worst_trade']:,.2f}")

        console.print(table)

        # Open positions
        if open_positions:
            pos_table = Table(title="📈 Open Positions")
            pos_table.add_column("Symbol")
            pos_table.add_column("Side")
            pos_table.add_column("Entry")
            pos_table.add_column("Stop Loss")
            pos_table.add_column("Take Profit")
            pos_table.add_column("Quantity")

            for pos in open_positions:
                side_color = "green" if pos.side == "long" else "red"
                pos_table.add_row(
                    pos.symbol,
                    f"[{side_color}]{pos.side.upper()}[/{side_color}]",
                    f"${pos.entry_price:,.2f}",
                    f"${pos.stop_loss:,.2f}",
                    f"${pos.take_profit:,.2f}",
                    f"{pos.quantity:.6f}",
                )

            console.print(pos_table)


# Singleton
execution_engine = ExecutionEngine(paper_trading=True)


if __name__ == "__main__":
    # Test the execution engine
    engine = ExecutionEngine(paper_trading=True)
    engine.display_portfolio()
