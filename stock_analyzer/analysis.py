# stock_analyzer/analysis.py
from datetime import timedelta

import numpy as np
import pandas as pd
from .models import Stock, StockData, AnalysisResult
from scipy.signal import argrelextrema
from stockstats import StockDataFrame


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


class AdvancedIndicators:
    def __init__(self, df):
        """Initialisiert die erweiterten Indikatoren mit einem DataFrame"""
        self.df = df.copy()
        # Stellt sicher, dass das DataFrame die richtigen Spalten hat
        required_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame muss Spalte '{col}' enthalten")

        # Spaltennamen anpassen für stockstats (falls benötigt)
        column_map = {
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume': 'volume'
        }

        # Erstelle eine Kopie mit den angepassten Spaltennamen
        self.stock_df = self.df.copy()
        for old_col, new_col in column_map.items():
            if old_col in self.stock_df.columns:
                self.stock_df[new_col] = self.stock_df[old_col]

        # Konvertiere zu StockDataFrame
        self.stock_df = StockDataFrame.retype(self.stock_df)

    def calculate_heikin_ashi(self):
        """
        Berechnet Heikin-Ashi-Kerzen.
        Heikin-Ashi sind modifizierte Kerzen, die weniger Rauschen und klarere Trends zeigen.
        """
        ha_df = self.df.copy()

        # HA Eröffnung = (Vortag HA Eröffnung + Vortag HA Schluss) / 2
        ha_df['ha_open'] = (self.df['open_price'] + self.df['close_price']).shift(1) / 2

        # Für die erste Zeile (keine Vortageswerte)
        ha_df.loc[0, 'ha_open'] = self.df.loc[0, 'open_price']

        # HA Schluss = (Eröffnung + Hoch + Tief + Schluss) / 4
        ha_df['ha_close'] = (self.df['open_price'] + self.df['high_price'] +
                             self.df['low_price'] + self.df['close_price']) / 4

        # HA Hoch = max(Hoch, ha_open, ha_close)
        ha_df['ha_high'] = ha_df[['ha_open', 'ha_close']].max(axis=1).combine(self.df['high_price'], max)

        # HA Tief = min(Tief, ha_open, ha_close)
        ha_df['ha_low'] = ha_df[['ha_open', 'ha_close']].min(axis=1).combine(self.df['low_price'], min)

        # Füge die Heikin-Ashi-Kerzen zum originalen DataFrame hinzu
        for col in ['ha_open', 'ha_high', 'ha_low', 'ha_close']:
            self.df[col] = ha_df[col]

        # Bestimme Trend basierend auf Heikin-Ashi
        self.df['ha_trend'] = np.where(
            self.df['ha_close'] > self.df['ha_open'],
            1,  # Grüne Kerze = Aufwärtstrend
            -1  # Rote Kerze = Abwärtstrend
        )

        return self.df

    def calculate_fibonacci_levels(self, window=100):
        """
        Berechnet Fibonacci-Retracement-Levels basierend auf den letzten X Tagen.
        """
        # Bestimme lokale Hochs und Tiefs im Fenster
        self.df['local_max'] = self.df['high_price'].rolling(window=window, center=True).max()
        self.df['local_min'] = self.df['low_price'].rolling(window=window, center=True).min()

        # Fibonacci-Levels: 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

        for i, level in enumerate(fib_levels):
            # Retracement-Levels nach unten (von Hochs)
            self.df[f'fib_down_{int(level * 1000)}'] = (
                    self.df['local_max'] - (self.df['local_max'] - self.df['local_min']) * level
            )

            # Retracement-Levels nach oben (von Tiefs)
            self.df[f'fib_up_{int(level * 1000)}'] = (
                    self.df['local_min'] + (self.df['local_max'] - self.df['local_min']) * level
            )

        # Fibonacci-Extensions: 1.618, 2.618, 4.236
        ext_levels = [1.618, 2.618, 4.236]

        for level in ext_levels:
            # Extension-Levels nach oben
            self.df[f'fib_ext_up_{int(level * 1000)}'] = (
                    self.df['local_min'] + (self.df['local_max'] - self.df['local_min']) * level
            )

            # Extension-Levels nach unten
            self.df[f'fib_ext_down_{int(level * 1000)}'] = (
                    self.df['local_max'] - (self.df['local_max'] - self.df['local_min']) * level
            )

        return self.df

    def detect_chart_patterns(self, window=20):
        """
        Erkennt gängige Chartmuster wie Head-and-Shoulders, Double Top/Bottom, etc.
        """
        # Initialisiere Musterspalten mit 0 (kein Muster erkannt)
        patterns = [
            'double_top', 'double_bottom', 'head_shoulders', 'inv_head_shoulders',
            'triangle_ascending', 'triangle_descending', 'flag_bullish', 'flag_bearish'
        ]

        for pattern in patterns:
            self.df[f'pattern_{pattern}'] = 0

        # Extrahiere Hochs und Tiefs
        order = int(window / 4)  # Ordnung für lokale Extrema

        # Finde lokale Maxima
        ilocs_max = argrelextrema(self.df['high_price'].values, np.greater, order=order)[0]

        # Finde lokale Minima
        ilocs_min = argrelextrema(self.df['low_price'].values, np.less, order=order)[0]

        # Double Top Erkennung
        for i in range(len(ilocs_max) - 1):
            # Zwei ähnliche Hochs mit einem Tief dazwischen
            if i + 1 < len(ilocs_max):
                peak1 = self.df['high_price'].iloc[ilocs_max[i]]
                peak2 = self.df['high_price'].iloc[ilocs_max[i + 1]]

                # Finde das Tief zwischen den beiden Hochs
                between_idx = self.df.index[(self.df.index > ilocs_max[i]) &
                                            (self.df.index < ilocs_max[i + 1])]

                if len(between_idx) > 0:
                    valley = self.df['low_price'].loc[between_idx].min()

                    # Kriterien für Double Top:
                    # 1. Ähnliche Hochpunkte (nicht mehr als 3% Unterschied)
                    # 2. Valley mind. 3% unter den Hochs
                    if (abs(peak1 - peak2) / peak1 < 0.03 and
                            valley < min(peak1, peak2) * 0.97):
                        # Markiere das zweite Hoch als Double Top
                        self.df.loc[self.df.index[ilocs_max[i + 1]], 'pattern_double_top'] = 1

        # Double Bottom Erkennung (analog zu Double Top)
        for i in range(len(ilocs_min) - 1):
            if i + 1 < len(ilocs_min):
                valley1 = self.df['low_price'].iloc[ilocs_min[i]]
                valley2 = self.df['low_price'].iloc[ilocs_min[i + 1]]

                between_idx = self.df.index[(self.df.index > ilocs_min[i]) &
                                            (self.df.index < ilocs_min[i + 1])]

                if len(between_idx) > 0:
                    peak = self.df['high_price'].loc[between_idx].max()

                    if (abs(valley1 - valley2) / valley1 < 0.03 and
                            peak > max(valley1, valley2) * 1.03):
                        self.df.loc[self.df.index[ilocs_min[i + 1]], 'pattern_double_bottom'] = 1

        # Head and Shoulders Erkennung
        for i in range(len(ilocs_max) - 2):
            if i + 2 < len(ilocs_max):
                left_shoulder = self.df['high_price'].iloc[ilocs_max[i]]
                head = self.df['high_price'].iloc[ilocs_max[i + 1]]
                right_shoulder = self.df['high_price'].iloc[ilocs_max[i + 2]]

                # Kriterien für Head and Shoulders:
                # 1. Head ist höher als beide Schultern
                # 2. Schultern sind auf ähnlicher Höhe (±5%)
                if (head > left_shoulder and head > right_shoulder and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):

                    # Finde Nackenlinie (Tiefstwerte zwischen Schultern und Kopf)
                    left_valley_idx = self.df.index[(self.df.index > ilocs_max[i]) &
                                                    (self.df.index < ilocs_max[i + 1])]
                    right_valley_idx = self.df.index[(self.df.index > ilocs_max[i + 1]) &
                                                     (self.df.index < ilocs_max[i + 2])]

                    if len(left_valley_idx) > 0 and len(right_valley_idx) > 0:
                        left_valley = self.df['low_price'].loc[left_valley_idx].min()
                        right_valley = self.df['low_price'].loc[right_valley_idx].min()

                        # Nackenlinie sollte relativ gerade sein
                        if abs(left_valley - right_valley) / left_valley < 0.03:
                            # Markiere das Muster am rechten Rand
                            self.df.loc[self.df.index[ilocs_max[i + 2]], 'pattern_head_shoulders'] = 1

        # Inverse Head and Shoulders (analog zu Head and Shoulders)
        for i in range(len(ilocs_min) - 2):
            if i + 2 < len(ilocs_min):
                left_shoulder = self.df['low_price'].iloc[ilocs_min[i]]
                head = self.df['low_price'].iloc[ilocs_min[i + 1]]
                right_shoulder = self.df['low_price'].iloc[ilocs_min[i + 2]]

                if (head < left_shoulder and head < right_shoulder and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):

                    left_peak_idx = self.df.index[(self.df.index > ilocs_min[i]) &
                                                  (self.df.index < ilocs_min[i + 1])]
                    right_peak_idx = self.df.index[(self.df.index > ilocs_min[i + 1]) &
                                                   (self.df.index < ilocs_min[i + 2])]

                    if len(left_peak_idx) > 0 and len(right_peak_idx) > 0:
                        left_peak = self.df['high_price'].loc[left_peak_idx].max()
                        right_peak = self.df['high_price'].loc[right_peak_idx].max()

                        if abs(left_peak - right_peak) / left_peak < 0.03:
                            self.df.loc[self.df.index[ilocs_min[i + 2]], 'pattern_inv_head_shoulders'] = 1

        # Erkennung von Dreiecksformationen
        window_size = min(window, len(self.df) // 3)
        for i in range(window_size, len(self.df)):
            segment = self.df.iloc[i - window_size:i]

            # Aufsteigendes Dreieck: Flache Widerstände, steigende Unterstützung
            highs = segment['high_price'].rolling(window=5).max()
            lows = segment['low_price']

            high_line_slope = self._calculate_slope(highs.dropna())
            low_line_slope = self._calculate_slope(lows.dropna())

            if abs(high_line_slope) < 0.001 and low_line_slope > 0.002:
                self.df.loc[self.df.index[i - 1], 'pattern_triangle_ascending'] = 1

            # Absteigendes Dreieck: Flache Unterstützung, fallende Widerstände
            if abs(low_line_slope) < 0.001 and high_line_slope < -0.002:
                self.df.loc[self.df.index[i - 1], 'pattern_triangle_descending'] = 1

        return self.df

    def _calculate_slope(self, series):
        """Hilfsfunktion zur Berechnung der Steigung einer linearen Regression"""
        if len(series) < 2:
            return 0

        x = np.arange(len(series))
        y = series.values

        # Steigung der linearen Regression berechnen
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def calculate_supertrend(self, period=10, multiplier=3.0):
        """
        Berechnet den SuperTrend-Indikator, ein Trendfolge-Indikator basierend auf ATR
        """
        # ATR berechnen
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['close_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['close_price'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        # ATR berechnen
        atr = true_range.rolling(window=period).mean()

        # Grundlegende Berechnungen für SuperTrend
        hl2 = (self.df['high_price'] + self.df['low_price']) / 2

        # Oberes Band und unteres Band
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # SuperTrend-Berechnung
        supertrend = pd.Series(0.0, index=self.df.index)
        direction = pd.Series(1, index=self.df.index)  # 1: aufwärts, -1: abwärts

        # Erste Werte setzen
        supertrend.iloc[0] = lower_band.iloc[0]

        # SuperTrend für jeden Zeitpunkt berechnen
        for i in range(1, len(self.df)):
            curr_close = self.df['close_price'].iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i - 1]

            # Aktuellen SuperTrend berechnen basierend auf vorherigem Wert
            if prev_supertrend <= curr_upper:
                curr_supertrend = curr_lower
            else:
                curr_supertrend = curr_upper

            # Richtung bestimmen
            if curr_close <= curr_supertrend:
                direction.iloc[i] = -1  # abwärts
            else:
                direction.iloc[i] = 1  # aufwärts

            # SuperTrend-Wert aktualisieren
            supertrend.iloc[i] = curr_supertrend

        # SuperTrend und Richtung zum DataFrame hinzufügen
        self.df['supertrend'] = supertrend
        self.df['supertrend_direction'] = direction

        return self.df

    def calculate_elliott_wave_points(self, window=100):
        """
        Identifiziert potenzielle Elliott Wave Punkte im Kursverlauf.
        Dies ist eine vereinfachte Version, die mögliche Wendepunkte markiert.
        """
        # Lokale Maxima und Minima finden
        order = int(window / 10)  # Ordnung für lokale Extrema

        # Finde lokale Maxima
        max_idx = argrelextrema(self.df['high_price'].values, np.greater, order=order)[0]

        # Finde lokale Minima
        min_idx = argrelextrema(self.df['low_price'].values, np.less, order=order)[0]

        # Initialisiere Spalte für Elliott Wave Punkte
        self.df['elliott_wave_point'] = 0

        # Markiere potenzielle Elliott Wave Punkte
        for idx in max_idx:
            self.df.loc[self.df.index[idx], 'elliott_wave_point'] = 1  # 1 für Hochpunkte

        for idx in min_idx:
            self.df.loc[self.df.index[idx], 'elliott_wave_point'] = -1  # -1 für Tiefpunkte

        # Identifiziere 5-Wellen-Muster (vereinfacht)
        for i in range(len(self.df) - window, len(self.df)):
            if i - window >= 0:
                segment = self.df.iloc[i - window:i]
                wave_points = segment[segment['elliott_wave_point'] != 0]

                # Mindestens 5 Wendepunkte für ein Elliott-Wellen-Muster
                if len(wave_points) >= 5:
                    # Prüfe, ob die letzten 5 Wendepunkte ein abwechselndes Muster bilden
                    last_5_points = wave_points['elliott_wave_point'].tail(5).values

                    # Prüfe auf abwechselndes Muster (+1, -1, +1, -1, +1) oder (-1, +1, -1, +1, -1)
                    alternating = True
                    for j in range(1, 5):
                        if last_5_points[j] == last_5_points[j - 1]:
                            alternating = False
                            break

                    if alternating:
                        # Der letzte Punkt im 5-Wellen-Muster
                        self.df.loc[wave_points.index[-1], 'elliott_wave_pattern'] = last_5_points[-1]

        return self.df

    def calculate_vwap(self):
        """
        Berechnet den Volume Weighted Average Price (VWAP)
        """
        # Berechne typischen Preis für jeden Tag: (Hoch + Tief + Schluss) / 3
        typical_price = (self.df['high_price'] + self.df['low_price'] + self.df['close_price']) / 3

        # Berechne typischen Preis × Volumen
        price_volume = typical_price * self.df['volume']

        # Berechne kumulative Summen
        cumulative_price_volume = price_volume.cumsum()
        cumulative_volume = self.df['volume'].cumsum()

        # VWAP = Kumulative Summe (Preis × Volumen) / Kumulative Summe (Volumen)
        self.df['vwap'] = cumulative_price_volume / cumulative_volume

        return self.df

    def calculate_all_indicators(self):
        """
        Berechnet alle implementierten fortgeschrittenen Indikatoren
        """
        self.calculate_heikin_ashi()
        self.calculate_fibonacci_levels()
        self.calculate_supertrend()
        self.calculate_vwap()
        self.detect_chart_patterns()
        self.calculate_elliott_wave_points()

        return self.df


# Erweitere die TechnicalAnalyzer-Klasse um die fortgeschrittenen Indikatoren
def extend_technical_analyzer(TechnicalAnalyzer):
    """
    Erweitert die TechnicalAnalyzer-Klasse um die neuen fortgeschrittenen Indikatoren.
    Diese Funktion sollte aufgerufen werden, nachdem die TechnicalAnalyzer-Klasse definiert wurde.
    """

    # Füge neue Methoden zur TechnicalAnalyzer-Klasse hinzu
    def calculate_advanced_indicators(self):
        """Berechnet alle fortgeschrittenen Indikatoren"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_all_indicators()
        return self.df

    def calculate_heikin_ashi(self):
        """Berechnet Heikin-Ashi-Kerzen"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_heikin_ashi()
        return self.df

    def calculate_fibonacci_levels(self, window=100):
        """Berechnet Fibonacci-Retracement-Levels"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_fibonacci_levels(window)
        return self.df

    def detect_chart_patterns(self, window=20):
        """Erkennt gängige Chartmuster"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.detect_chart_patterns(window)
        return self.df

    def calculate_supertrend(self, period=10, multiplier=3.0):
        """Berechnet den SuperTrend-Indikator"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_supertrend(period, multiplier)
        return self.df

    def calculate_elliott_wave_points(self, window=100):
        """Identifiziert potenzielle Elliott Wave Punkte"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_elliott_wave_points(window)
        return self.df

    def calculate_vwap(self):
        """Berechnet den Volume Weighted Average Price"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_vwap()
        return self.df

    # Füge die neuen Methoden zur TechnicalAnalyzer-Klasse hinzu
    TechnicalAnalyzer.calculate_advanced_indicators = calculate_advanced_indicators
    TechnicalAnalyzer.calculate_heikin_ashi = calculate_heikin_ashi
    TechnicalAnalyzer.calculate_fibonacci_levels = calculate_fibonacci_levels
    TechnicalAnalyzer.detect_chart_patterns = detect_chart_patterns
    TechnicalAnalyzer.calculate_supertrend = calculate_supertrend
    TechnicalAnalyzer.calculate_elliott_wave_points = calculate_elliott_wave_points
    TechnicalAnalyzer.calculate_vwap = calculate_vwap

    # Überschreibe die calculate_indicators-Methode, um optional auch fortgeschrittene Indikatoren zu berechnen
    original_calculate_indicators = TechnicalAnalyzer.calculate_indicators

    def new_calculate_indicators(self, include_advanced=False):
        """
        Berechnet alle technischen Indikatoren, optional auch die fortgeschrittenen

        Args:
            include_advanced: Wenn True, werden auch fortgeschrittene Indikatoren berechnet
        """
        # Rufe zuerst die Original-Methode auf
        original_calculate_indicators(self)

        # Wenn fortgeschrittene Indikatoren gewünscht sind, berechne diese
        if include_advanced:
            self.calculate_advanced_indicators()

        return self.df

    # Überschreibe die originale Methode
    TechnicalAnalyzer.calculate_indicators = new_calculate_indicators

    return TechnicalAnalyzer

TechnicalAnalyzer = extend_technical_analyzer(TechnicalAnalyzer)
