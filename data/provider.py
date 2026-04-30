"""
AI Trading Floor v2 — Data Provider Layer
Uses OpenBB for unified access to 90+ data providers
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from rich.console import Console

console = Console()

class DataProvider:
    """Unified data provider using OpenBB + yfinance fallback"""

    def __init__(self):
        self._obb = None
        self._yf = None
        self._initialize()

    def _initialize(self):
        """Initialize OpenBB and yfinance"""
        try:
            from openbb import obb
            self._obb = obb
            console.print("[green]✅ OpenBB initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ OpenBB failed: {e}[/yellow]")

        try:
            import yfinance as yf
            self._yf = yf
            console.print("[green]✅ yfinance initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ yfinance failed: {e}[/yellow]")

    def get_stock_data(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get stock market data
        
        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT")
            days: Number of days of history
            interval: Data interval ("1d", "1h", "5m")
        
        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Try OpenBB first (best data quality)
        if self._obb:
            try:
                result = self._obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    interval=interval,
                )
                df = result.to_df()
                if not df.empty:
                    console.print(f"[green]📊 Got {symbol} from OpenBB ({len(df)} candles)[/green]")
                    return self._standardize_dataframe(df)
            except Exception as e:
                console.print(f"[yellow]OpenBB failed for {symbol}: {e}[/yellow]")

        # Fallback to yfinance
        if self._yf:
            try:
                ticker = self._yf.Ticker(symbol)
                df = ticker.history(period=f"{days}d", interval=interval)
                if not df.empty:
                    console.print(f"[green]📊 Got {symbol} from yfinance ({len(df)} candles)[/green]")
                    return self._standardize_dataframe(df)
            except Exception as e:
                console.print(f"[yellow]yfinance failed for {symbol}: {e}[/yellow]")

        console.print(f"[red]❌ Failed to get data for {symbol}[/red]")
        return None

    def get_crypto_data(
        self,
        symbol: str = "BTC/USDT",
        exchange: str = "binance",
        days: int = 30,
        timeframe: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Get cryptocurrency OHLCV data via CCXT
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH/USDT")
            exchange: Exchange name (default: binance)
            days: Number of days of history
            timeframe: Candle timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import ccxt

            # Initialize exchange (no API key needed for public data)
            ex = getattr(ccxt, exchange)({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            # Calculate timestamps
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Fetch OHLCV
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

            if ohlcv:
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                console.print(f"[green]📊 Got {symbol} from {exchange} ({len(df)} candles)[/green]")
                return df

        except Exception as e:
            console.print(f"[red]CCXT failed for {symbol}: {e}[/red]")

        return None

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a stock
        
        Args:
            symbol: Stock ticker
        
        Returns:
            Dictionary with fundamental metrics
        """
        if self._obb:
            try:
                # Get company overview
                overview = self._obb.equity.profile(symbol=symbol)
                df = overview.to_df()

                if not df.empty:
                    row = df.iloc[0]
                    return {
                        "symbol": symbol,
                        "name": row.get("name", "N/A"),
                        "sector": row.get("sector", "N/A"),
                        "industry": row.get("industry", "N/A"),
                        "market_cap": row.get("market_cap", 0),
                        "pe_ratio": row.get("pe_ratio", 0),
                        "pb_ratio": row.get("pb_ratio", 0),
                        "dividend_yield": row.get("dividend_yield", 0),
                        "52_week_high": row.get("fifty_two_week_high", 0),
                        "52_week_low": row.get("fifty_two_week_low", 0),
                        "beta": row.get("beta", 0),
                        "description": row.get("description", ""),
                    }
            except Exception as e:
                console.print(f"[yellow]OpenBB fundamentals failed: {e}[/yellow]")

        # Fallback to yfinance
        if self._yf:
            try:
                ticker = self._yf.Ticker(symbol)
                info = ticker.info
                return {
                    "symbol": symbol,
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "pb_ratio": info.get("priceToBook", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                    "52_week_low": info.get("fiftyTwoWeekLow", 0),
                    "beta": info.get("beta", 0),
                    "description": info.get("longBusinessSummary", ""),
                }
            except Exception as e:
                console.print(f"[yellow]yfinance fundamentals failed: {e}[/yellow]")

        return None

    def get_macro_data(self, indicator: str = "GDP") -> Optional[pd.DataFrame]:
        """
        Get macroeconomic data via OpenBB
        
        Args:
            indicator: Economic indicator ("GDP", "CPI", "UNEMPLOYMENT", etc.)
        
        Returns:
            DataFrame with macro data
        """
        if self._obb:
            try:
                result = self._obb.economy.macro_indicator(
                    indicator=indicator,
                    country="united_states"
                )
                df = result.to_df()
                if not df.empty:
                    console.print(f"[green]📊 Got {indicator} macro data[/green]")
                    return df
            except Exception as e:
                console.print(f"[yellow]Macro data failed: {e}[/yellow]")

        return None

    def get_market_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """
        Get market news via OpenBB
        
        Args:
            symbol: Optional stock ticker for company-specific news
            limit: Number of news items
        
        Returns:
            List of news dictionaries
        """
        news_items = []

        if self._obb:
            try:
                if symbol:
                    result = self._obb.news.company(symbol=symbol, limit=limit)
                else:
                    result = self._obb.news.market(limit=limit)

                df = result.to_df()
                for _, row in df.iterrows():
                    news_items.append({
                        "title": row.get("title", ""),
                        "summary": row.get("summary", ""),
                        "url": row.get("url", ""),
                        "source": row.get("source", ""),
                        "date": str(row.get("date", "")),
                    })
            except Exception as e:
                console.print(f"[yellow]News fetch failed: {e}[/yellow]")

        return news_items

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase OHLCV format"""
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume',
        }

        # Rename columns
        df = df.rename(columns={
            k: v for k, v in column_mapping.items()
            if k in df.columns
        })

        # Ensure we have OHLCV columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # Try case-insensitive match
                for c in df.columns:
                    if c.lower() == col:
                        df = df.rename(columns={c: col})
                        break

        return df


# Singleton instance
data_provider = DataProvider()


if __name__ == "__main__":
    # Test the data provider
    from rich import print as rprint

    dp = DataProvider()

    # Test stock data
    print("\n=== Testing Stock Data ===")
    aapl = dp.get_stock_data("AAPL", days=5)
    if aapl is not None:
        print(aapl.tail())

    # Test crypto data
    print("\n=== Testing Crypto Data ===")
    btc = dp.get_crypto_data("BTC/USDT", days=5, timeframe="1d")
    if btc is not None:
        print(btc.tail())

    # Test fundamentals
    print("\n=== Testing Fundamentals ===")
    fundamentals = dp.get_fundamentals("AAPL")
    if fundamentals:
        rprint(fundamentals)
