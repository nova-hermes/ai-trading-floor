"""
AI Trading Floor v2 — AI Agent System
Multi-agent debate with distinct investor personas
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.insert(0, '/home/doug/ai-trading-floor')
from config.settings import LLM_CONFIG, AGENT_CONFIG

console = Console()


@dataclass
class AgentOpinion:
    """An agent's opinion on a trade"""
    agent_name: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "short-term"


@dataclass
class TradeDecision:
    """Final aggregated trade decision"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    reasoning: str
    agent_votes: List[AgentOpinion] = field(default_factory=list)
    consensus_score: float = 0.0


class TradingAgent:
    """A single AI trading agent with a distinct persona"""

    def __init__(self, persona: Dict):
        self.name = persona["name"]
        self.style = persona["style"]
        self.focus = persona["focus"]
        self.risk_tolerance = persona["risk_tolerance"]
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )

    def analyze(self, context: Dict) -> AgentOpinion:
        """
        Analyze market data and form an opinion
        
        Args:
            context: Dictionary with market data, technicals, news, etc.
        
        Returns:
            AgentOpinion with the agent's assessment
        """
        prompt = self._build_prompt(context)

        try:
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            console.print(f"[red]Agent {self.name} failed: {e}[/red]")
            return AgentOpinion(
                agent_name=self.name,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Agent error: {str(e)}"
            )

    def _system_prompt(self) -> str:
        """System prompt defining the agent's persona"""
        return f"""You are {self.name}, a legendary investor.

Your investment style: {self.style}
Your focus areas: {self.focus}
Your risk tolerance: {self.risk_tolerance}

You MUST respond in this EXACT JSON format (no other text):
{{
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Your detailed analysis in 2-3 sentences",
    "price_target": null or number,
    "stop_loss": null or number,
    "timeframe": "short-term" or "medium-term" or "long-term"
}}

IMPORTANT: Be decisive. Pick BUY, SELL, or HOLD. Don't hedge with "it depends."
Base your decision on the data provided, not general principles."""

    def _build_prompt(self, context: Dict) -> str:
        """Build the analysis prompt with market data"""
        symbol = context.get("symbol", "UNKNOWN")
        current_price = context.get("current_price", 0)
        technicals = context.get("technicals", {})
        fundamentals = context.get("fundamentals", {})
        news = context.get("news", [])
        recent_performance = context.get("recent_performance", {})

        prompt = f"""=== TRADE ANALYSIS: {symbol} ===

CURRENT PRICE: ${current_price:,.2f}

RECENT PERFORMANCE:
- 1-day change: {recent_performance.get('change_1d', 'N/A')}%
- 5-day change: {recent_performance.get('change_5d', 'N/A')}%
- 20-day high: ${recent_performance.get('high_20d', 'N/A')}
- 20-day low: ${recent_performance.get('low_20d', 'N/A')}

TECHNICAL INDICATORS:
- RSI (14): {technicals.get('momentum', {}).get('rsi_14', 'N/A')}
- MACD: {technicals.get('momentum', {}).get('macd_line', 'N/A')}
- MACD Signal: {technicals.get('momentum', {}).get('macd_signal', 'N/A')}
- Bollinger Position: {technicals.get('volatility', {}).get('bb_position', 'N/A')}
- SMA 20: {technicals.get('moving_averages', {}).get('sma_20', 'N/A')}
- Volume Ratio (vs 20d avg): {technicals.get('volume_analysis', {}).get('volume_ratio', 'N/A')}
- ATR: {technicals.get('volatility', {}).get('atr_14', 'N/A')}

PATTERNS DETECTED:
{json.dumps(technicals.get('patterns', {}), indent=2)}

TECHNICAL SIGNALS:
{self._format_signals(technicals.get('signals', []))}
"""

        # Add fundamentals if available (for stocks)
        if fundamentals and fundamentals.get('pe_ratio'):
            prompt += f"""
FUNDAMENTALS:
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
- P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}
- Market Cap: ${fundamentals.get('market_cap', 0):,.0f}
- Sector: {fundamentals.get('sector', 'N/A')}
- Beta: {fundamentals.get('beta', 'N/A')}
"""

        # Add news if available
        if news:
            prompt += "\nRECENT NEWS:\n"
            for item in news[:3]:
                prompt += f"- {item.get('title', '')}\n"

        prompt += f"""
Based on YOUR investment style ({self.style}) and focus ({self.focus}),
what is your recommendation: BUY, SELL, or HOLD?

Respond ONLY with the JSON object. No other text."""

        return prompt

    def _format_signals(self, signals: list) -> str:
        """Format technical signals for the prompt"""
        if not signals:
            return "No signals generated"

        lines = []
        for s in signals:
            if hasattr(s, 'indicator'):
                lines.append(f"- {s.indicator}: {s.signal} (confidence: {s.strength:.0%}) - {s.description}")
            elif isinstance(s, dict):
                lines.append(f"- {s.get('indicator', '?')}: {s.get('signal', '?')} - {s.get('description', '')}")
        return "\n".join(lines) if lines else "No signals"

    def _parse_response(self, content: str) -> AgentOpinion:
        """Parse the agent's JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            return AgentOpinion(
                agent_name=self.name,
                action=data.get("action", "HOLD").upper(),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                price_target=data.get("price_target"),
                stop_loss=data.get("stop_loss"),
                timeframe=data.get("timeframe", "short-term"),
            )
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"[yellow]Failed to parse {self.name}'s response: {e}[/yellow]")
            # Try to extract action from text
            content_upper = content.upper()
            if "BUY" in content_upper:
                action = "BUY"
            elif "SELL" in content_upper:
                action = "SELL"
            else:
                action = "HOLD"

            return AgentOpinion(
                agent_name=self.name,
                action=action,
                confidence=0.3,
                reasoning=content[:200],
            )


class TradingFloor:
    """
    Orchestrates multiple trading agents and reaches consensus
    Like a real trading floor — experts debate, then decide
    """

    def __init__(self):
        self.agents = [
            TradingAgent(persona) for persona in AGENT_CONFIG["personas"]
        ]
        self.consensus_threshold = AGENT_CONFIG["consensus_threshold"]

    def analyze_and_decide(self, context: Dict) -> TradeDecision:
        """
        Have all agents analyze a trade opportunity and reach consensus
        
        Args:
            context: Market data context from DataProvider + TechnicalAnalyzer
        
        Returns:
            TradeDecision with consensus recommendation
        """
        symbol = context.get("symbol", "UNKNOWN")
        current_price = context.get("current_price", 0)

        console.print(f"\n[bold blue]🏛️ TRADING FLOOR SESSION: {symbol} @ ${current_price:,.2f}[/bold blue]")
        console.print("=" * 60)

        # Phase 1: Each agent analyzes independently
        opinions: List[AgentOpinion] = []
        for agent in self.agents:
            console.print(f"\n[bold]{agent.name}[/bold] ({agent.style}) is analyzing...")
            opinion = agent.analyze(context)
            opinions.append(opinion)
            self._display_opinion(opinion)

        # Phase 2: Calculate consensus
        consensus = self._calculate_consensus(opinions, current_price, context)

        # Phase 3: Display final decision
        self._display_decision(consensus)

        return consensus

    def _calculate_consensus(self, opinions: List[AgentOpinion], current_price: float, context: Dict) -> TradeDecision:
        """Calculate weighted consensus from all agent opinions"""
        # Count votes with confidence weighting
        buy_score = 0
        sell_score = 0
        hold_score = 0
        total_confidence = 0

        for op in opinions:
            weight = op.confidence
            total_confidence += weight

            if op.action == "BUY":
                buy_score += weight
            elif op.action == "SELL":
                sell_score += weight
            else:
                hold_score += weight

        # Normalize scores
        if total_confidence > 0:
            buy_pct = buy_score / total_confidence
            sell_pct = sell_score / total_confidence
            hold_pct = hold_score / total_confidence
        else:
            buy_pct = sell_pct = hold_pct = 0.33

        # Determine action
        if buy_pct >= self.consensus_threshold:
            action = "BUY"
            confidence = buy_pct
        elif sell_pct >= self.consensus_threshold:
            action = "SELL"
            confidence = sell_pct
        elif buy_pct > sell_pct and buy_pct > hold_pct:
            action = "BUY"
            confidence = buy_pct
        elif sell_pct > buy_pct and sell_pct > hold_pct:
            action = "SELL"
            confidence = sell_pct
        else:
            action = "HOLD"
            confidence = hold_pct

        # Calculate risk management levels
        technicals = context.get("technicals", {})
        atr = technicals.get("volatility", {}).get("atr_14", current_price * 0.02)

        if action == "BUY":
            stop_loss = current_price - (atr * 2) if atr else current_price * 0.95
            take_profit = current_price + (atr * 3) if atr else current_price * 1.10
        elif action == "SELL":
            stop_loss = current_price + (atr * 2) if atr else current_price * 1.05
            take_profit = current_price - (atr * 3) if atr else current_price * 0.90
        else:
            stop_loss = 0
            take_profit = 0

        # Aggregate reasoning
        reasoning_parts = []
        for op in opinions:
            if op.action == action:
                reasoning_parts.append(f"{op.agent_name}: {op.reasoning[:100]}")

        return TradeDecision(
            symbol=context.get("symbol", "UNKNOWN"),
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=min(0.05, confidence * 0.10),  # Max 5% of portfolio
            reasoning=" | ".join(reasoning_parts[:3]),
            agent_votes=opinions,
            consensus_score=max(buy_pct, sell_pct, hold_pct),
        )

    def _display_opinion(self, opinion: AgentOpinion):
        """Display an agent's opinion in a nice format"""
        color_map = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
        emoji_map = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}

        color = color_map.get(opinion.action, "white")
        emoji = emoji_map.get(opinion.action, "❓")

        console.print(Panel(
            f"[bold]{emoji} {opinion.action}[/bold] (Confidence: {opinion.confidence:.0%})\n\n"
            f"{opinion.reasoning}\n\n"
            f"{'Price Target: $' + f'{opinion.price_target:,.2f}' if opinion.price_target else ''}"
            f"{' | Stop Loss: $' + f'{opinion.stop_loss:,.2f}' if opinion.stop_loss else ''}"
            f"{' | Timeframe: ' + opinion.timeframe if opinion.timeframe else ''}",
            title=f"[bold]{opinion.agent_name}[/bold]",
            border_style=color,
        ))

    def _display_decision(self, decision: TradeDecision):
        """Display the final trading decision"""
        color_map = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
        emoji_map = {"BUY": "🚀", "SELL": "🔴", "HOLD": "⏸️"}

        color = color_map.get(decision.action, "white")
        emoji = emoji_map.get(decision.action, "❓")

        # Vote summary table
        table = Table(title="Agent Votes")
        table.add_column("Agent", style="cyan")
        table.add_column("Vote", style="bold")
        table.add_column("Confidence", style="magenta")
        table.add_column("Reasoning")

        for vote in decision.agent_votes:
            vote_color = color_map.get(vote.action, "white")
            table.add_row(
                vote.agent_name,
                f"[{vote_color}]{vote.action}[/{vote_color}]",
                f"{vote.confidence:.0%}",
                vote.reasoning[:80] + "..." if len(vote.reasoning) > 80 else vote.reasoning,
            )

        console.print(table)

        # Final decision
        console.print(Panel(
            f"[bold {color}]{emoji} {decision.action} {decision.symbol}[/bold {color}]\n\n"
            f"Entry: ${decision.entry_price:,.2f}\n"
            f"Stop Loss: ${decision.stop_loss:,.2f}\n"
            f"Take Profit: ${decision.take_profit:,.2f}\n"
            f"Position Size: {decision.position_size_pct:.1%} of portfolio\n"
            f"Consensus Score: {decision.consensus_score:.0%}\n\n"
            f"[dim]{decision.reasoning}[/dim]",
            title="[bold]🏛️ FLOOR DECISION[/bold]",
            border_style=color,
        ))

        return decision


# Singleton
trading_floor = TradingFloor()


if __name__ == "__main__":
    from data.provider import data_provider
    from data.technical import tech_analyzer

    # Test with BTC
    symbol = "BTC/USDT"
    btc_data = data_provider.get_crypto_data(symbol, days=30, timeframe="1h")

    if btc_data is not None:
        # Run technical analysis
        technicals = tech_analyzer.analyze(btc_data)

        # Build context for agents
        context = {
            "symbol": symbol,
            "current_price": technicals["price"]["current"],
            "technicals": technicals,
            "recent_performance": technicals["price"],
            "fundamentals": {},
            "news": [],
        }

        # Run the trading floor
        decision = trading_floor.analyze_and_decide(context)
        print(f"\nFinal Decision: {decision.action} with {decision.confidence:.0%} confidence")


def analyze_with_tradingview(self, context: Dict) -> TradeDecision:
    """
    Enhanced analysis that includes TradingView signals
    """
    from data.tradingview import tradingview_analyzer
    
    symbol = context.get("symbol", "UNKNOWN")
    current_price = context.get("current_price", 0)
    
    # Get TradingView signal
    is_crypto = "/" in symbol
    if is_crypto:
        # Convert BTC/USDT to BTCUSDT
        tv_symbol = symbol.replace("/", "")
        tv_signal = tradingview_analyzer.get_crypto_analysis(tv_symbol, "1h")
    else:
        tv_signal = tradingview_analyzer.get_stock_analysis(symbol, "1d")
    
    if tv_signal:
        # Add TradingView data to context
        context["tradingview"] = {
            "recommendation": tv_signal.recommendation,
            "buy_count": tv_signal.buy_count,
            "sell_count": tv_signal.sell_count,
            "neutral_count": tv_signal.neutral_count,
            "score": tradingview_analyzer.get_recommendation_score(tv_signal),
        }
    
    # Run normal analysis
    return self.analyze_and_decide(context)
