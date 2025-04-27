# stock_analyzer/analysis.py
from datetime import timedelta

import numpy as np
import pandas as pd
from .models import Stock, StockData, AnalysisResult


class TechnicalAnalyzer:
    def __init__(self, stock_symbol, days=365):
        self.stock = Stock.objects.get(symbol=stock_symbol)

        # Das neueste Datum (letzte historische Daten)
        last_date = StockData.objects.filter(stock=self.stock).order_by('-date').first().date

        # Berechne das Startdatum
        start_date = last_date - timedelta(days=days)

        # Historische Daten für den Zeitraum ab dem Startdatum laden
        data = StockData.objects.filter(stock=self.stock, date__range=[start_date, last_date]).order_by('date')

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

        self._calculate_roc()

        self._calculate_psar()

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
        """Berechnet einen präziseren technischen Score basierend auf dynamischen Kriterien"""
        if self.df.empty or 'rsi' not in self.df.columns:
            self.calculate_indicators()

        latest = self.df.iloc[-1]
        score = 50  # Neutraler Startwert
        signals = []

        # --- Einzelne Indikatoren dynamisch bewerten ---

        # RSI dynamisch
        if latest['rsi'] < 30:
            bonus = (30 - latest['rsi']) * 0.5
            score += bonus
            signals.append(("RSI", "BUY", f"Überverkauft ({latest['rsi']:.2f}) → +{bonus:.1f}"))
        elif latest['rsi'] > 70:
            malus = (latest['rsi'] - 70) * 0.5
            score -= malus
            signals.append(("RSI", "SELL", f"Überkauft ({latest['rsi']:.2f}) → -{malus:.1f}"))

        # MACD Signal
        if latest['macd'] > latest['macd_signal']:
            score += 7.5
            signals.append(("MACD", "BUY", "MACD über Signal-Linie → +7.5"))
        else:
            score -= 7.5
            signals.append(("MACD", "SELL", "MACD unter Signal-Linie → -7.5"))

        # SMA Levels
        if latest['close_price'] > latest['sma_20']:
            score += 5
            signals.append(("SMA 20", "BUY", "Preis über SMA 20 → +5"))
        else:
            score -= 5
            signals.append(("SMA 20", "SELL", "Preis unter SMA 20 → -5"))

        if latest['close_price'] > latest['sma_50']:
            score += 5
            signals.append(("SMA 50", "BUY", "Preis über SMA 50 → +5"))
        else:
            score -= 5
            signals.append(("SMA 50", "SELL", "Preis unter SMA 50 → -5"))

        if latest['close_price'] > latest['sma_200']:
            score += 10
            signals.append(("SMA 200", "BUY", "Preis über SMA 200 → +10"))
        else:
            score -= 10
            signals.append(("SMA 200", "SELL", "Preis unter SMA 200 → -10"))

        # Bollinger Bänder
        if latest['close_price'] > latest['bollinger_upper']:
            score -= 7.5
            signals.append(("Bollinger Bänder", "SELL", "Preis über oberem Band → -7.5"))
        elif latest['close_price'] < latest['bollinger_lower']:
            score += 7.5
            signals.append(("Bollinger Bänder", "BUY", "Preis unter unterem Band → +7.5"))

        # Stochastik Oszillator
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            score += 5
            signals.append(("Stochastik", "BUY", "Stochastik überverkauft → +5"))
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            score -= 5
            signals.append(("Stochastik", "SELL", "Stochastik überkauft → -5"))

        # Ichimoku Cloud
        if latest['close_price'] > latest['senkou_span_a'] and latest['close_price'] > latest['senkou_span_b']:
            score += 10
            signals.append(("Ichimoku", "BUY", "Preis über Cloud → +10"))
        elif latest['close_price'] < latest['senkou_span_a'] and latest['close_price'] < latest['senkou_span_b']:
            score -= 10
            signals.append(("Ichimoku", "SELL", "Preis unter Cloud → -10"))

        # OBV Trendbestätigung
        if 'obv' in self.df.columns:
            price_change = self.df['close_price'].iloc[-5:].pct_change().sum()
            obv_change = (self.df['obv'].iloc[-1] - self.df['obv'].iloc[-5]) / abs(self.df['obv'].iloc[-5])

            if price_change > 0 and obv_change > 0:
                score += 5
                signals.append(("OBV", "BUY", "Volumen bestätigt Anstieg → +5"))
            elif price_change < 0 and obv_change < 0:
                score -= 5
                signals.append(("OBV", "SELL", "Volumen bestätigt Rückgang → -5"))

        # --- Bonus für Kombinationen ---
        if latest['rsi'] < 30 and latest['macd'] > latest['macd_signal'] and latest['close_price'] > latest['sma_20']:
            score += 10
            signals.append(("Kombi-Signal", "BUY", "RSI + MACD + SMA positiv → +10"))

        # --- Verstärkung durch Trendstärke (ADX) ---
        if latest['adx'] > 25:
            if latest['+di'] > latest['-di']:
                score *= 1.1
                signals.append(("ADX Boost", "BUY", "Starker Aufwärtstrend (Verstärkung Score)"))
            else:
                score *= 0.9
                signals.append(("ADX Dämpfung", "SELL", "Starker Abwärtstrend (Dämpfung Score)"))

        # Score begrenzen
        score = max(0, min(100, score))

        # --- Neue Empfehlungen (5-Stufen) ---
        if score >= 90:
            recommendation = "STRONG BUY"
        elif score >= 70:
            recommendation = "BUY"
        elif score >= 40:
            recommendation = "HOLD"
        elif score >= 20:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"

        return {
            'score': round(score, 2),
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

    def _calculate_psar(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Berechnet den Parabolic SAR Indikator"""
        high = self.df['high_price']
        low = self.df['low_price']
        close = self.df['close_price']

        psar = close.copy()
        bull = True  # Aufwärtstrend zu Beginn
        af = af_start  # Acceleration Factor

        # Extrempunkte
        ep = low[0]  # Extrempunkt, starten mit tiefem Wert
        hp = high[0]  # Höchster Punkt im aktuellen Trend
        lp = low[0]  # Niedrigster Punkt im aktuellen Trend

        for i in range(2, len(close)):
            # Vorherige Werte
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])

            # Trendumkehr überprüfen
            reverse = False

            if bull:
                # Aufwärtstrend
                if low[i] < psar[i]:
                    bull = False  # Zu Abwärtstrend wechseln
                    reverse = True
                    psar[i] = hp  # Setzen SAR auf höchsten Punkt
                    ep = low[i]  # Extrempunkt auf niedrigsten Punkt
                    af = af_start  # AF zurücksetzen
                else:
                    if high[i] > hp:
                        hp = high[i]  # Höchsten Punkt aktualisieren
                        ep = hp  # Extrempunkt aktualisieren
                        af = min(af + af_increment, af_max)  # AF erhöhen
            else:
                # Abwärtstrend
                if high[i] > psar[i]:
                    bull = True  # Zu Aufwärtstrend wechseln
                    reverse = True
                    psar[i] = lp  # Setzen SAR auf niedrigsten Punkt
                    ep = high[i]  # Extrempunkt auf höchsten Punkt
                    af = af_start  # AF zurücksetzen
                else:
                    if low[i] < lp:
                        lp = low[i]  # Niedrigsten Punkt aktualisieren
                        ep = lp  # Extrempunkt aktualisieren
                        af = min(af + af_increment, af_max)  # AF erhöhen

            # Begrenzung des SAR-Werts
            if bull:
                psar[i] = min(psar[i], low[i - 1], low[i - 2])
            else:
                psar[i] = max(psar[i], high[i - 1], high[i - 2])

        self.df['psar'] = psar

    def _calculate_roc(self, period=14):
        """Berechnet den Rate of Change (ROC) Indikator"""
        self.df['roc'] = self.df['close_price'].pct_change(period) * 100

    def save_analysis_result(self):
        result = self.calculate_technical_score()
        latest_date = self.df['date'].iloc[-1]

        details = result.get('details', {})

        analysis_result, created = AnalysisResult.objects.update_or_create(
            stock=self.stock,
            date=latest_date,
            defaults={
                'technical_score': result['score'],
                'recommendation': result['recommendation'],
                'rsi_value': details.get('rsi'),
                'macd_value': details.get('macd'),
                'macd_signal': details.get('macd_signal'),
                'sma_20': details.get('sma_20'),
                'sma_50': details.get('sma_50'),
                'sma_200': details.get('sma_200'),
                'bollinger_upper': details.get('bollinger_upper'),
                'bollinger_lower': details.get('bollinger_lower')
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


