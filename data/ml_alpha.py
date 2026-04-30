"""
AI Trading Floor v2 — ML Alpha Discovery Engine
Simplified version of Qlib using scikit-learn for ML-based trading signals
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class AlphaSignal:
    """ML-generated trading signal"""
    symbol: str
    prediction: str  # "BUY", "SELL", "HOLD"
    confidence: float
    probability: Dict[str, float]  # Class probabilities
    feature_importance: Dict[str, float]
    model_accuracy: float
    timeframe: str


class FeatureEngineer:
    """Create ML features from OHLCV data"""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features for ML models
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Price-based features
        features['returns_1d'] = close.pct_change(1)
        features['returns_5d'] = close.pct_change(5)
        features['returns_10d'] = close.pct_change(10)
        features['returns_20d'] = close.pct_change(20)

        # Volatility
        features['volatility_5d'] = close.pct_change().rolling(5).std()
        features['volatility_20d'] = close.pct_change().rolling(20).std()

        # Moving average features
        for period in [5, 10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}'] = close / ma  # Price relative to MA
            features[f'ma_{period}_slope'] = ma.pct_change(5)  # MA slope

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = (close - (sma20 + 2 * std20)) / close
        features['bb_lower'] = (close - (sma20 - 2 * std20)) / close
        features['bb_width'] = (4 * std20) / sma20

        # Volume features
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['volume_change'] = volume.pct_change()

        # Price patterns
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)

        # Candle features
        body = abs(close - df['open'])
        range_ = high - low
        features['body_size'] = body / range_
        features['upper_shadow'] = (high - df[['open', 'close']].max(axis=1)) / range_
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - low) / range_

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean() / close

        return features

    def create_labels(self, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        Create classification labels (BUY/SELL/HOLD)
        
        Args:
            df: OHLCV DataFrame
            forward_periods: How many periods to look ahead
            threshold: Return threshold for BUY/SELL classification
        
        Returns:
            Series of labels: 2 (BUY), 0 (SELL), 1 (HOLD)
        """
        forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)

        labels = pd.Series(1, index=df.index)  # Default HOLD
        labels[forward_returns > threshold] = 2   # BUY
        labels[forward_returns < -threshold] = 0  # SELL

        return labels


class MLAlphaEngine:
    """
    ML-based alpha discovery engine
    Trains models to predict BUY/SELL/HOLD signals
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
        }
        self.scaler = StandardScaler()
        self.trained_models = {}

    def train(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Train ML models on historical data
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name
        
        Returns:
            Dictionary with training results
        """
        console.print(f"\n[bold blue]🤖 Training ML Models: {symbol}[/bold blue]")

        # Create features and labels
        features = self.feature_engineer.create_features(df)
        labels = self.feature_engineer.create_labels(df)

        # Drop NaN rows
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 100:
            console.print("[red]Not enough data for training (need 100+ rows)[/red]")
            return {}

        console.print(f"Training data: {len(X)} samples, {len(X.columns)} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train each model with time series cross-validation
        results = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            console.print(f"\nTraining {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')

            # Train on full data
            model.fit(X_scaled, y)
            self.trained_models[name] = model

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X.columns, model.feature_importances_))
                # Sort by importance
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            else:
                importance = {}

            results[name] = {
                "cv_accuracy": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "feature_importance": importance,
            }

            console.print(f"  CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

        # Display results
        self._display_training_results(results, symbol)

        return results

    def predict(self, df: pd.DataFrame, symbol: str = "UNKNOWN", model_name: str = "gradient_boosting") -> AlphaSignal:
        """
        Generate trading signal using trained model
        
        Args:
            df: Recent OHLCV DataFrame
            symbol: Symbol name
            model_name: Which model to use
        
        Returns:
            AlphaSignal with prediction
        """
        if model_name not in self.trained_models:
            console.print(f"[red]Model {model_name} not trained yet[/red]")
            return None

        # Create features for latest data
        features = self.feature_engineer.create_features(df)

        # Get latest row (with features)
        latest = features.dropna().iloc[-1:].values

        if len(latest) == 0:
            console.print("[red]No valid features for prediction[/red]")
            return None

        # Scale
        latest_scaled = self.scaler.transform(latest)

        # Predict
        model = self.trained_models[model_name]
        prediction = model.predict(latest_scaled)[0]
        probabilities = model.predict_proba(latest_scaled)[0]

        # Map prediction to label
        label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        predicted_label = label_map.get(prediction, "HOLD")

        # Get class probabilities
        prob_dict = {}
        for i, label in enumerate(label_map.values()):
            if i < len(probabilities):
                prob_dict[label] = float(probabilities[i])

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(features.columns, model.feature_importances_))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        else:
            importance = {}

        return AlphaSignal(
            symbol=symbol,
            prediction=predicted_label,
            confidence=float(max(probabilities)),
            probability=prob_dict,
            feature_importance=importance,
            model_accuracy=self.trained_models[model_name].score(latest_scaled, [prediction]),
            timeframe="1h",
        )

    def backtest_ml(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
        """
        Backtest ML model with walk-forward validation
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name
        
        Returns:
            Dictionary with backtest results
        """
        console.print(f"\n[bold blue]📊 ML Walk-Forward Backtest: {symbol}[/bold blue]")

        features = self.feature_engineer.create_features(df)
        labels = self.feature_engineer.create_labels(df)

        # Drop NaN
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 200:
            console.print("[red]Not enough data for backtest[/red]")
            return {}

        # Walk-forward validation
        n_splits = 5
        window_size = len(X) // n_splits
        train_size = int(window_size * 0.7)
        test_size = window_size - train_size

        results = []
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

        for i in range(n_splits):
            start = i * window_size
            train_end = start + train_size
            test_end = min(train_end + test_size, len(X))

            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate accuracy
            acc = accuracy_score(y_test, y_pred)

            results.append({
                "split": i + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": acc,
            })

        # Display results
        table = Table(title="📊 ML Walk-Forward Results")
        table.add_column("Split")
        table.add_column("Train Size")
        table.add_column("Test Size")
        table.add_column("Accuracy")

        for r in results:
            acc_color = "green" if r["accuracy"] > 0.5 else "red"
            table.add_row(
                str(r["split"]),
                str(r["train_size"]),
                str(r["test_size"]),
                f"[{acc_color}]{r['accuracy']:.2%}[/{acc_color}]",
            )

        console.print(table)

        avg_acc = np.mean([r["accuracy"] for r in results])
        console.print(f"\n[bold]Average Accuracy: {avg_acc:.2%}[/bold]")

        return {
            "splits": results,
            "avg_accuracy": avg_acc,
            "consistent": avg_acc > 0.5,
        }

    def _display_training_results(self, results: Dict, symbol: str):
        """Display training results"""
        table = Table(title=f"🤖 ML Training Results: {symbol}")
        table.add_column("Model")
        table.add_column("CV Accuracy")
        table.add_column("CV Std")
        table.add_column("Top Features")

        for name, res in results.items():
            top_features = list(res["feature_importance"].keys())[:3]
            top_str = ", ".join(top_features) if top_features else "N/A"

            acc_color = "green" if res["cv_accuracy"] > 0.5 else "red"
            table.add_row(
                name,
                f"[{acc_color}]{res['cv_accuracy']:.2%}[/{acc_color}]",
                f"{res['cv_std']:.2%}",
                top_str,
            )

        console.print(table)

    def display_prediction(self, signal: AlphaSignal):
        """Display ML prediction"""
        pred_colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}
        color = pred_colors.get(signal.prediction, "white")

        console.print(Panel(
            f"[bold {color}]{signal.prediction}[/bold {color}] (Confidence: {signal.confidence:.0%})\n\n"
            f"Probabilities:\n"
            f"  BUY: {signal.probability.get('BUY', 0):.0%}\n"
            f"  HOLD: {signal.probability.get('HOLD', 0):.0%}\n"
            f"  SELL: {signal.probability.get('SELL', 0):.0%}\n\n"
            f"Top Features:\n" +
            "\n".join(f"  {k}: {v:.4f}" for k, v in list(signal.feature_importance.items())[:5]),
            title=f"ML Prediction: {signal.symbol}",
            border_style=color,
        ))


# Singleton
ml_engine = MLAlphaEngine()


if __name__ == "__main__":
    # Test ML engine
    from data.provider import data_provider

    print("=== Testing ML Alpha Engine ===")

    # Get data
    btc = data_provider.get_crypto_data("BTC/USDT", days=180, timeframe="1h")

    if btc is not None:
        # Train models
        results = ml_engine.train(btc, "BTC/USDT")

        # Make prediction
        signal = ml_engine.predict(btc, "BTC/USDT")
        if signal:
            ml_engine.display_prediction(signal)

        # Backtest ML
        ml_engine.backtest_ml(btc, "BTC/USDT")
