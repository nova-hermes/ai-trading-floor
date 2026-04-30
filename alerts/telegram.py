"""
AI Trading Floor v2 — Telegram Alerts Module
Sends trade signals and portfolio updates to Telegram
"""
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import os

from rich.console import Console

console = Console()

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")


class TelegramAlerts:
    """
    Send trading alerts to Telegram
    """

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_HOME_CHANNEL", "233082513")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

        if not self.bot_token:
            console.print("[yellow]⚠️ No TELEGRAM_BOT_TOKEN found in .env[/yellow]")

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram
        
        Args:
            message: Message text
            parse_mode: "HTML" or "Markdown"
        
        Returns:
            True if sent successfully
        """
        if not self.bot_token:
            console.print("[yellow]Telegram not configured[/yellow]")
            return False

        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                return True
            else:
                console.print(f"[red]Telegram error: {response.text}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]Telegram send failed: {e}[/red]")
            return False

    def send_trade_signal(self, decision) -> bool:
        """
        Send a trade signal alert
        
        Args:
            decision: TradeDecision object
        
        Returns:
            True if sent successfully
        """
        # Emoji based on action
        emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
        emoji = emoji_map.get(decision.action, "❓")

        # Color for confidence
        if decision.confidence >= 0.7:
            conf_emoji = "🔥"
        elif decision.confidence >= 0.5:
            conf_emoji = "⚡"
        else:
            conf_emoji = "💤"

        message = f"""
<b>{emoji} Trade Signal: {decision.symbol}</b>

<b>Action:</b> {decision.action}
<b>Confidence:</b> {conf_emoji} {decision.confidence:.0%}
<b>Entry Price:</b> ${decision.entry_price:,.2f}
<b>Stop Loss:</b> ${decision.stop_loss:,.2f}
<b>Take Profit:</b> ${decision.take_profit:,.2f}
<b>Position Size:</b> {decision.position_size_pct:.1%}

<b>Consensus:</b> {decision.consensus_score:.0%}

<b>Reasoning:</b>
<i>{decision.reasoning[:200]}...</i>

<b>Agent Votes:</b>
"""

        # Add agent votes
        for vote in decision.agent_votes:
            vote_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(vote.action, "❓")
            message += f"  {vote_emoji} {vote.agent_name}: {vote.action} ({vote.confidence:.0%})\n"

        message += f"\n<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"

        return self.send_message(message)

    def send_ml_prediction(self, signal) -> bool:
        """
        Send ML prediction alert
        
        Args:
            signal: AlphaSignal object
        
        Returns:
            True if sent successfully
        """
        emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
        emoji = emoji_map.get(signal.prediction, "❓")

        message = f"""
<b>🤖 ML Prediction: {signal.symbol}</b>

<b>Prediction:</b> {emoji} {signal.prediction}
<b>Confidence:</b> {signal.confidence:.0%}

<b>Probabilities:</b>
  📈 BUY: {signal.probability.get('BUY', 0):.0%}
  ⏸️ HOLD: {signal.probability.get('HOLD', 0):.0%}
  📉 SELL: {signal.probability.get('SELL', 0):.0%}

<b>Top Features:</b>
"""

        for feat, imp in list(signal.feature_importance.items())[:5]:
            message += f"  • {feat}: {imp:.4f}\n"

        message += f"\n<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"

        return self.send_message(message)

    def send_portfolio_update(self, summary: Dict) -> bool:
        """
        Send portfolio summary alert
        
        Args:
            summary: Portfolio summary dictionary
        
        Returns:
            True if sent successfully
        """
        pnl_emoji = "🟢" if summary.get("total_pnl", 0) > 0 else "🔴"

        message = f"""
<b>📊 Portfolio Update</b>

<b>Open Positions:</b> {summary.get('open_positions', 0)}
<b>Total Trades:</b> {summary.get('total_trades', 0)}
<b>Total P&L:</b> {pnl_emoji} ${summary.get('total_pnl', 0):,.2f}
<b>Win Rate:</b> {summary.get('win_rate', 0):.1f}%

<b>Best Trade:</b> 🟢 ${summary.get('best_trade', 0):,.2f}
<b>Worst Trade:</b> 🔴 ${summary.get('worst_trade', 0):,.2f}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

        return self.send_message(message)

    def send_price_alert(self, symbol: str, current_price: float, target_price: float, direction: str) -> bool:
        """
        Send price alert when target is reached
        
        Args:
            symbol: Symbol name
            current_price: Current price
            target_price: Target price
            direction: "above" or "below"
        
        Returns:
            True if sent successfully
        """
        emoji = "🔔" if direction == "above" else "🔕"

        message = f"""
<b>{emoji} Price Alert: {symbol}</b>

<b>Current Price:</b> ${current_price:,.2f}
<b>Target:</b> ${target_price:,.2f}
<b>Direction:</b> Price is {direction} target

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

        return self.send_message(message)

    def send_scan_results(self, decisions: List) -> bool:
        """
        Send market scan results
        
        Args:
            decisions: List of TradeDecision objects
        
        Returns:
            True if sent successfully
        """
        if not decisions:
            return True

        message = f"""
<b>🔍 Market Scan Results</b>

<b>Symbols Scanned:</b> {len(decisions)}
<b>Actionable Signals:</b> {len([d for d in decisions if d.action != 'HOLD'])}

"""

        # Add top signals
        for d in decisions[:5]:
            emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(d.action, "❓")
            message += f"{emoji} <b>{d.symbol}</b>: {d.action} ({d.confidence:.0%})\n"

        message += f"\n<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"

        return self.send_message(message)

    def send_error(self, error_msg: str) -> bool:
        """
        Send error alert
        
        Args:
            error_msg: Error message
        
        Returns:
            True if sent successfully
        """
        message = f"""
<b>⚠️ Error Alert</b>

<b>Error:</b> {error_msg}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

        return self.send_message(message)


# Singleton
telegram_alerts = TelegramAlerts()


if __name__ == "__main__":
    # Test Telegram alerts
    alerts = TelegramAlerts()

    # Test sending a message
    print("Testing Telegram alerts...")
    result = alerts.send_message("🤖 <b>AI Trading Floor v2</b> — Test message!")

    if result:
        print("✅ Telegram message sent successfully!")
    else:
        print("❌ Failed to send Telegram message")
