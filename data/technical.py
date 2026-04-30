"""
AI Trading Floor v2 — Technical Analysis Engine
Computes 30+ indicators and pattern recognition
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TechnicalSignal:
    """A technical analysis signal"""
    indicator: str
    value: float
    signal: str  # "BUY", "SELL", "NEUTRAL"
    strength: float  # 0.0 to 1.0
    description: str


class TechnicalAnalyzer:
    """Computes technical indicators and generates signals"""

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run full technical analysis on OHLCV data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            Dictionary with all indicators and signals
        """
        if len(df) < 20:
            return {"error": "Not enough data (need at least 20 candles)"}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        results = {
            "price": {
                "current": float(close.iloc[-1]),
                "change_1d": float(close.pct_change().iloc[-1] * 100),
                "change_5d": float(close.pct_change(5).iloc[-1] * 100) if len(df) > 5 else 0,
                "high_20d": float(high.rolling(20).max().iloc[-1]),
                "low_20d": float(low.rolling(20).min().iloc[-1]),
            },
            "moving_averages": self._moving_averages(close),
            "momentum": self._momentum(close, high, low),
            "volume_analysis": self._volume_analysis(close, volume),
            "volatility": self._volatility(close, high, low),
            "patterns": self._pattern_recognition(df),
            "signals": [],
            "overall_signal": "NEUTRAL",
            "confidence": 0.0,
        }

        # Generate signals
        signals = self._generate_signals(results)
        results["signals"] = signals

        # Calculate overall signal
        buy_count = sum(1 for s in signals if s.signal == "BUY")
        sell_count = sum(1 for s in signals if s.signal == "SELL")
        total = len(signals) if signals else 1

        if buy_count > sell_count * 1.5:
            results["overall_signal"] = "BUY"
            results["confidence"] = buy_count / total
        elif sell_count > buy_count * 1.5:
            results["overall_signal"] = "SELL"
            results["confidence"] = sell_count / total
        else:
            results["overall_signal"] = "NEUTRAL"
            results["confidence"] = max(buy_count, sell_count) / total

        return results

    def _moving_averages(self, close: pd.Series) -> Dict:
        """Calculate moving averages"""
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean() if len(close) >= 50 else pd.Series([np.nan])
        ma_200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan])

        # EMA
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()

        current = close.iloc[-1]

        return {
            "sma_5": float(ma_5.iloc[-1]) if not pd.isna(ma_5.iloc[-1]) else None,
            "sma_10": float(ma_10.iloc[-1]) if not pd.isna(ma_10.iloc[-1]) else None,
            "sma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else None,
            "sma_50": float(ma_50.iloc[-1]) if len(close) >= 50 and not pd.isna(ma_50.iloc[-1]) else None,
            "sma_200": float(ma_200.iloc[-1]) if len(close) >= 200 and not pd.isna(ma_200.iloc[-1]) else None,
            "ema_12": float(ema_12.iloc[-1]),
            "ema_26": float(ema_26.iloc[-1]),
            "price_vs_sma20": "above" if current > ma_20.iloc[-1] else "below",
            "golden_cross": bool(ma_5.iloc[-1] > ma_20.iloc[-1]) if not pd.isna(ma_5.iloc[-1]) and not pd.isna(ma_20.iloc[-1]) else None,
        }

    def _momentum(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """Calculate momentum indicators"""
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = 100 * ((close - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(3).mean()

        # Williams %R
        williams_r = -100 * ((high_14 - close) / (high_14 - low_14))

        # Rate of Change (ROC)
        roc = close.pct_change(10) * 100

        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        return {
            "rsi_14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            "rsi_signal": self._rsi_signal(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else "N/A",
            "macd_line": float(macd_line.iloc[-1]),
            "macd_signal": float(signal_line.iloc[-1]),
            "macd_histogram": float(macd_histogram.iloc[-1]),
            "macd_crossover": "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2] else
                            "bearish" if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2] else "none",
            "stoch_k": float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else None,
            "stoch_d": float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else None,
            "williams_r": float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None,
            "roc_10": float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else None,
            "cci_20": float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None,
        }

    def _volume_analysis(self, close: pd.Series, volume: pd.Series) -> Dict:
        """Analyze volume patterns"""
        vol_sma_20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_sma_20.iloc[-1] if vol_sma_20.iloc[-1] > 0 else 1

        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).cumsum()

        # Volume Price Trend (VPT)
        vpt = (volume * close.pct_change()).cumsum()

        return {
            "current_volume": float(volume.iloc[-1]),
            "avg_volume_20d": float(vol_sma_20.iloc[-1]) if not pd.isna(vol_sma_20.iloc[-1]) else None,
            "volume_ratio": float(vol_ratio),
            "volume_signal": "high" if vol_ratio > 1.5 else "low" if vol_ratio < 0.5 else "normal",
            "obv": float(obv.iloc[-1]),
            "obv_trend": "up" if obv.iloc[-1] > obv.iloc[-5] else "down",
            "vpt": float(vpt.iloc[-1]),
        }

    def _volatility(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """Calculate volatility metrics"""
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        bb_width = (bb_upper - bb_lower) / sma_20

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Historical Volatility
        returns = close.pct_change()
        hist_vol = returns.rolling(20).std() * np.sqrt(252) * 100

        current = close.iloc[-1]

        return {
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_middle": float(sma_20.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "bb_width": float(bb_width.iloc[-1]),
            "bb_position": "above_upper" if current > bb_upper.iloc[-1] else
                          "below_lower" if current < bb_lower.iloc[-1] else "within",
            "atr_14": float(atr.iloc[-1]),
            "atr_pct": float(atr.iloc[-1] / current * 100),
            "hist_volatility_20d": float(hist_vol.iloc[-1]) if not pd.isna(hist_vol.iloc[-1]) else None,
        }

    def _pattern_recognition(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        o, h, l, c = df['open'], df['high'], df['low'], df['close']

        patterns = {}

        # Doji (small body, long wicks)
        body = abs(c.iloc[-1] - o.iloc[-1])
        range_ = h.iloc[-1] - l.iloc[-1]
        patterns["doji"] = bool(body < range_ * 0.1 and range_ > 0)

        # Hammer (small body at top, long lower shadow)
        lower_shadow = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
        upper_shadow = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
        patterns["hammer"] = bool(lower_shadow > body * 2 and upper_shadow < body * 0.5)

        # Shooting Star (small body at bottom, long upper shadow)
        patterns["shooting_star"] = bool(upper_shadow > body * 2 and lower_shadow < body * 0.5)

        # Engulfing patterns
        if len(df) >= 2:
            prev_o, prev_c = o.iloc[-2], c.iloc[-2]
            curr_o, curr_c = o.iloc[-1], c.iloc[-1]
            patterns["bullish_engulfing"] = bool(
                prev_c < prev_o and  # Previous red
                curr_c > curr_o and  # Current green
                curr_o < prev_c and  # Opens below prev close
                curr_c > prev_o       # Closes above prev open
            )
            patterns["bearish_engulfing"] = bool(
                prev_c > prev_o and  # Previous green
                curr_c < curr_o and  # Current red
                curr_o > prev_c and  # Opens above prev close
                curr_c < prev_o       # Closes below prev open
            )

        # Three consecutive up/down candles
        if len(df) >= 3:
            last_3_green = all(c.iloc[-i] > o.iloc[-i] for i in range(1, 4))
            last_3_red = all(c.iloc[-i] < o.iloc[-i] for i in range(1, 4))
            patterns["three_green"] = bool(last_3_green)
            patterns["three_red"] = bool(last_3_red)

        return patterns

    def _rsi_signal(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi < 30:
            return "oversold"
        elif rsi > 70:
            return "overbought"
        return "neutral"

    def _generate_signals(self, results: Dict) -> list:
        """Generate trading signals from analysis"""
        signals = []

        # RSI signals
        rsi = results['momentum'].get('rsi_14')
        if rsi:
            if rsi < 30:
                signals.append(TechnicalSignal("RSI", rsi, "BUY", 0.8, f"RSI oversold at {rsi:.1f}"))
            elif rsi > 70:
                signals.append(TechnicalSignal("RSI", rsi, "SELL", 0.8, f"RSI overbought at {rsi:.1f}"))
            else:
                signals.append(TechnicalSignal("RSI", rsi, "NEUTRAL", 0.3, f"RSI neutral at {rsi:.1f}"))

        # MACD crossover
        macd_cross = results['momentum'].get('macd_crossover')
        if macd_cross == "bullish":
            signals.append(TechnicalSignal("MACD", 0, "BUY", 0.7, "MACD bullish crossover"))
        elif macd_cross == "bearish":
            signals.append(TechnicalSignal("MACD", 0, "SELL", 0.7, "MACD bearish crossover"))

        # Bollinger Bands
        bb_pos = results['volatility'].get('bb_position')
        if bb_pos == "below_lower":
            signals.append(TechnicalSignal("BB", 0, "BUY", 0.6, "Price below lower Bollinger Band"))
        elif bb_pos == "above_upper":
            signals.append(TechnicalSignal("BB", 0, "SELL", 0.6, "Price above upper Bollinger Band"))

        # Moving Average signals
        ma = results['moving_averages']
        if ma.get('golden_cross'):
            signals.append(TechnicalSignal("MA", 0, "BUY", 0.5, "Price above SMA20 (golden cross zone)"))
        elif ma.get('price_vs_sma20') == "below":
            signals.append(TechnicalSignal("MA", 0, "SELL", 0.4, "Price below SMA20"))

        # Volume confirmation
        vol = results['volume_analysis']
        if vol.get('volume_ratio', 1) > 1.5:
            # High volume confirms the move
            last_signal = signals[-1] if signals else None
            if last_signal:
                last_signal.strength = min(1.0, last_signal.strength + 0.1)

        # Pattern signals
        patterns = results['patterns']
        if patterns.get('bullish_engulfing'):
            signals.append(TechnicalSignal("PATTERN", 0, "BUY", 0.7, "Bullish engulfing pattern"))
        if patterns.get('bearish_engulfing'):
            signals.append(TechnicalSignal("PATTERN", 0, "SELL", 0.7, "Bearish engulfing pattern"))
        if patterns.get('hammer'):
            signals.append(TechnicalSignal("PATTERN", 0, "BUY", 0.6, "Hammer pattern (potential reversal)"))
        if patterns.get('shooting_star'):
            signals.append(TechnicalSignal("PATTERN", 0, "SELL", 0.6, "Shooting star pattern"))

        return signals


# Singleton
tech_analyzer = TechnicalAnalyzer()


if __name__ == "__main__":
    # Test with sample data
    from data.provider import data_provider

    print("=== Technical Analysis Test ===\n")

    # Get BTC data
    btc = data_provider.get_crypto_data("BTC/USDT", days=30, timeframe="1h")
    if btc is not None:
        results = tech_analyzer.analyze(btc)

        from rich import print as rprint
        rprint(results)
