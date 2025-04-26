# stock_analyzer/analysis.py
import numpy as np
import pandas as pd
from .models import Stock, StockData, AnalysisResult


class TechnicalAnalyzer:
    def __init__(self, stock_symbol, days=365):
        self.stock = Stock.objects.get(symbol=stock_symbol)
        # Historische Daten abrufen und in ein DataFrame umwandeln
        data = StockData.objects.filter(stock=self.stock).order_by('date')[:days]
        self.df = pd.DataFrame(list(data.values()))

        if not self.df.empty:
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close', 'volume']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(float)

    def calculate_indicators(self):
        """Berechnet alle technischen Indikatoren"""
        if self.df.empty:
            return None

        # RSI (Relative Strength Index)
        self._calculate_rsi()

        # Gleitende Durchschnitte (Simple Moving Averages)
        self._calculate_sma()

        # MACD (Moving Average Convergence Divergence)
        self._calculate_macd()

        # Bollinger Bänder
        self._calculate_bollinger_bands()

        # Stochastik Oszillator
        self._calculate_stochastic()

        # Average Directional Index (ADX)
        self._calculate_adx()

        # Ichimoku Cloud
        self._calculate_ichimoku()

        # OBV (On-Balance Volume)
        self._calculate_obv()

        # ATR (Average True Range)
        self._calculate_atr()

        return self.df

    def _calculate_rsi(self, period=14):
        """Berechnet den RSI-Indikator"""
        delta = self.df['close_price'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

    def _calculate_sma(self):
        """Berechnet verschiedene gleitende Durchschnitte"""
        self.df['sma_20'] = self.df['close_price'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close_price'].rolling(window=50).mean()
        self.df['sma_200'] = self.df['close_price'].rolling(window=200).mean()

        # Exponentieller gleitender Durchschnitt (EMA)
        self.df['ema_12'] = self.df['close_price'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close_price'].ewm(span=26, adjust=False).mean()

    def _calculate_macd(self):
        """Berechnet den MACD-Indikator"""
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

    def _calculate_bollinger_bands(self, period=20, std_dev=2):
        """Berechnet Bollinger Bänder"""
        self.df['bollinger_middle'] = self.df['close_price'].rolling(window=period).mean()
        std = self.df['close_price'].rolling(window=period).std()
        self.df['bollinger_upper'] = self.df['bollinger_middle'] + std_dev * std
        self.df['bollinger_lower'] = self.df['bollinger_middle'] - std_dev * std

    def _calculate_stochastic(self, k_period=14, d_period=3):
        """Berechnet Stochastik Oszillator"""
        low_min = self.df['low_price'].rolling(window=k_period).min()
        high_max = self.df['high_price'].rolling(window=k_period).max()

        self.df['stoch_k'] = 100 * ((self.df['close_price'] - low_min) / (high_max - low_min))
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()

    def _calculate_adx(self, period=14):
        """Berechnet den Average Directional Index (ADX)"""
        # True Range berechnen
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['close_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['close_price'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        # Directional Movement berechnen
        plus_dm = self.df['high_price'].diff()
        minus_dm = self.df['low_price'].diff(-1).abs()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Roh-Werte in ein DataFrame umwandeln
        data = pd.DataFrame({
            'tr': true_range,
            '+dm': plus_dm,
            '-dm': minus_dm
        })

        # Exponential Moving Average für die Berechnungen
        smoothed_tr = data['tr'].ewm(span=period, adjust=False).mean()
        smoothed_plus_dm = data['+dm'].ewm(span=period, adjust=False).mean()
        smoothed_minus_dm = data['-dm'].ewm(span=period, adjust=False).mean()

        # +DI und -DI berechnen
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr

        # ADX berechnen
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        self.df['adx'] = dx.ewm(span=period, adjust=False).mean()
        self.df['+di'] = plus_di
        self.df['-di'] = minus_di

    def _calculate_ichimoku(self):
        """Berechnet die Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): (9-Perioden-Hoch + 9-Perioden-Tief) / 2
        high_9 = self.df['high_price'].rolling(window=9).max()
        low_9 = self.df['low_price'].rolling(window=9).min()
        self.df['tenkan_sen'] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line): (26-Perioden-Hoch + 26-Perioden-Tief) / 2
        high_26 = self.df['high_price'].rolling(window=26).max()
        low_26 = self.df['low_price'].rolling(window=26).min()
        self.df['kijun_sen'] = (high_26 + low_26) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-Perioden-Hoch + 52-Perioden-Tief) / 2
        high_52 = self.df['high_price'].rolling(window=52).max()
        low_52 = self.df['low_price'].rolling(window=52).min()
        self.df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        # Chikou Span (Lagging Span): Schlusskurs 26 Perioden zurückversetzt
        self.df['chikou_span'] = self.df['close_price'].shift(-26)

    def _calculate_obv(self):
        """Berechnet das On-Balance Volume (OBV)"""
        price_diff = self.df['close_price'].diff()
        volume = self.df['volume']

        obv = pd.Series(0, index=self.df.index)

        for i in range(1, len(self.df)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        self.df['obv'] = obv

    def _calculate_atr(self, period=14):
        """Berechnet den Average True Range (ATR)"""
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['close_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['close_price'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        self.df['atr'] = true_range.rolling(window=period).mean()

    def calculate_technical_score(self):
        """Berechnet einen technischen Score basierend auf allen Indikatoren"""
        if self.df.empty or 'rsi' not in self.df.columns:
            self.calculate_indicators()

        latest = self.df.iloc[-1]
        score = 50  # Neutraler Ausgangspunkt
        signals = []

        # RSI Signale (0-100)
        if latest['rsi'] < 30:
            score += 10  # Überverkauft - Kaufsignal
            signals.append(("RSI", "BUY", f"Überverkauft ({latest['rsi']:.2f})"))
        elif latest['rsi'] > 70:
            score -= 10  # Überkauft - Verkaufssignal
            signals.append(("RSI", "SELL", f"Überkauft ({latest['rsi']:.2f})"))

        # MACD Signale
        if latest['macd'] > latest['macd_signal']:
            score += 7.5
            signals.append(("MACD", "BUY", "MACD über Signal-Linie"))
        else:
            score -= 7.5
            signals.append(("MACD", "SELL", "MACD unter Signal-Linie"))

        # MACD Histogram Trend
        if self.df['macd_histogram'].iloc[-1] > self.df['macd_histogram'].iloc[-2]:
            score += 2.5
            signals.append(("MACD Histogram", "BUY", "Aufwärtstrend"))
        else:
            score -= 2.5
            signals.append(("MACD Histogram", "SELL", "Abwärtstrend"))

        # SMA Signale
        if latest['close_price'] > latest['sma_20']:
            score += 5
            signals.append(("SMA 20", "BUY", "Preis über SMA 20"))
        else:
            score -= 5
            signals.append(("SMA 20", "SELL", "Preis unter SMA 20"))

        if latest['close_price'] > latest['sma_50']:
            score += 5
            signals.append(("SMA 50", "BUY", "Preis über SMA 50"))
        else:
            score -= 5
            signals.append(("SMA 50", "SELL", "Preis unter SMA 50"))

        if latest['close_price'] > latest['sma_200']:
            score += 10
            signals.append(("SMA 200", "BUY", "Preis über SMA 200"))
        else:
            score -= 10
            signals.append(("SMA 200", "SELL", "Preis unter SMA 200"))

        # Golden/Death Cross
        if latest['sma_50'] > latest['sma_200'] and self.df['sma_50'].iloc[-2] <= self.df['sma_200'].iloc[-2]:
            score += 15  # Golden Cross - starkes Kaufsignal
            signals.append(("Golden Cross", "BUY", "SMA 50 überquert SMA 200 aufwärts"))
        elif latest['sma_50'] < latest['sma_200'] and self.df['sma_50'].iloc[-2] >= self.df['sma_200'].iloc[-2]:
            score -= 15  # Death Cross - starkes Verkaufssignal
            signals.append(("Death Cross", "SELL", "SMA 50 überquert SMA 200 abwärts"))

        # Bollinger Bänder
        if latest['close_price'] > latest['bollinger_upper']:
            score -= 7.5
            signals.append(("Bollinger Bänder", "SELL", "Preis über oberem Band"))
        elif latest['close_price'] < latest['bollinger_lower']:
            score += 7.5
            signals.append(("Bollinger Bänder", "BUY", "Preis unter unterem Band"))

        # Stochastik
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            score += 5
            signals.append(("Stochastik", "BUY", "Überverkauft"))
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            score -= 5
            signals.append(("Stochastik", "SELL", "Überkauft"))

        # Stochastik Kreuzung
        if latest['stoch_k'] > latest['stoch_d'] and self.df['stoch_k'].iloc[-2] <= self.df['stoch_d'].iloc[-2]:
            score += 5
            signals.append(("Stochastik Kreuzung", "BUY", "K-Linie überquert D-Linie aufwärts"))
        elif latest['stoch_k'] < latest['stoch_d'] and self.df['stoch_k'].iloc[-2] >= self.df['stoch_d'].iloc[-2]:
            score -= 5
            signals.append(("Stochastik Kreuzung", "SELL", "K-Linie überquert D-Linie abwärts"))

        # ADX (Trendstärke)
        if latest['adx'] > 25:
            # Starker Trend - prüfen welche Richtung
            if latest['+di'] > latest['-di']:
                score += 10
                signals.append(("ADX", "BUY", f"Starker Aufwärtstrend (ADX: {latest['adx']:.2f})"))
            else:
                score -= 10
                signals.append(("ADX", "SELL", f"Starker Abwärtstrend (ADX: {latest['adx']:.2f})"))

        # Ichimoku Cloud
        if (latest['close_price'] > latest['senkou_span_a'] and
                latest['close_price'] > latest['senkou_span_b']):
            score += 10
            signals.append(("Ichimoku", "BUY", "Preis über der Cloud"))
        elif (latest['close_price'] < latest['senkou_span_a'] and
              latest['close_price'] < latest['senkou_span_b']):
            score -= 10
            signals.append(("Ichimoku", "SELL", "Preis unter der Cloud"))

        # Tenkan-sen / Kijun-sen Kreuzung
        if (latest['tenkan_sen'] > latest['kijun_sen'] and
                self.df['tenkan_sen'].iloc[-2] <= self.df['kijun_sen'].iloc[-2]):
            score += 7.5
            signals.append(("Ichimoku TK Cross", "BUY", "Tenkan-sen überquert Kijun-sen aufwärts"))
        elif (latest['tenkan_sen'] < latest['kijun_sen'] and
              self.df['tenkan_sen'].iloc[-2] >= self.df['kijun_sen'].iloc[-2]):
            score -= 7.5
            signals.append(("Ichimoku TK Cross", "SELL", "Tenkan-sen überquert Kijun-sen abwärts"))

        # OBV Trend mit Preisbewegung vergleichen
        price_change = self.df['close_price'].iloc[-5:].pct_change().sum()
        obv_change = (self.df['obv'].iloc[-1] - self.df['obv'].iloc[-5]) / abs(self.df['obv'].iloc[-5])

        if price_change > 0 and obv_change > 0:
            score += 5  # Preis und Volumen steigen - bullish
            signals.append(("OBV", "BUY", "Volumen bestätigt Preisanstieg"))
        elif price_change < 0 and obv_change < 0:
            score -= 5  # Preis und Volumen fallen - bearish
            signals.append(("OBV", "SELL", "Volumen bestätigt Preisrückgang"))
        elif price_change > 0 and obv_change < 0:
            score -= 2.5  # Preisanstieg ohne Volumenstärke - potentiell bearish
            signals.append(("OBV", "CAUTION", "Volumen bestätigt Preisanstieg nicht"))
        elif price_change < 0 and obv_change > 0:
            score += 2.5  # Preisrückgang mit steigendem Volumen - potentiell bullish
            signals.append(("OBV", "CAUTION", "Volumen zeigt mögliche Umkehr an"))

        # Score begrenzen
        score = max(0, min(100, score))

        # Empfehlung generieren
        if score >= 70:
            recommendation = "BUY"
        elif score <= 30:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return {
            'score': score,
            'recommendation': recommendation,
            'signals': signals,
            'details': {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'sma_20': latest['sma_20'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'bollinger_upper': latest['bollinger_upper'],
                'bollinger_lower': latest['bollinger_lower'],
                'stoch_k': latest['stoch_k'],
                'stoch_d': latest['stoch_d'],
                'adx': latest['adx'],
                '+di': latest['+di'],
                '-di': latest['-di']
            }
        }

    def save_analysis_result(self):
        """Speichert das Analyseergebnis in der Datenbank"""
        result = self.calculate_technical_score()
        latest_date = self.df['date'].iloc[-1]

        analysis_result, created = AnalysisResult.objects.update_or_create(
            stock=self.stock,
            date=latest_date,
            defaults={
                'technical_score': result['score'],
                'recommendation': result['recommendation'],
                'rsi_value': result['details']['rsi'],
                'macd_value': result['details']['macd'],
                'macd_signal': result['details']['macd_signal'],
                'sma_20': result['details']['sma_20'],
                'sma_50': result['details']['sma_50'],
                'sma_200': result['details']['sma_200'],
                'bollinger_upper': result['details']['bollinger_upper'],
                'bollinger_lower': result['details']['bollinger_lower']
            }
        )

        return analysis_result


def _calculate_indicators_for_dataframe(self, df):
    """Berechnet Indikatoren für ein bereitgestelltes DataFrame (für Backtesting)"""
    # Kopie erstellen, um das Originaldatenframe nicht zu verändern
    df_copy = df.copy()

    # Sicherstellen, dass die Spalten korrekte Datentypen haben
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(float)

    # RSI berechnen
    delta = df_copy['close_price'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))

    # Moving Averages
    df_copy['sma_20'] = df_copy['close_price'].rolling(window=20).mean()
    df_copy['sma_50'] = df_copy['close_price'].rolling(window=50).mean()
    df_copy['sma_200'] = df_copy['close_price'].rolling(window=200).mean()

    # EMA für MACD
    df_copy['ema_12'] = df_copy['close_price'].ewm(span=12, adjust=False).mean()
    df_copy['ema_26'] = df_copy['close_price'].ewm(span=26, adjust=False).mean()

    # MACD
    df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    df_copy['macd_histogram'] = df_copy['macd'] - df_copy['macd_signal']

    # Bollinger Bands
    df_copy['bollinger_middle'] = df_copy['close_price'].rolling(window=20).mean()
    std = df_copy['close_price'].rolling(window=20).std()
    df_copy['bollinger_upper'] = df_copy['bollinger_middle'] + 2 * std
    df_copy['bollinger_lower'] = df_copy['bollinger_middle'] - 2 * std

    return df_copy