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

        last_date = StockData.objects.filter(stock=self.stock).order_by('-date').first().date
        days = self._adjust_window_based_on_volatility(days)
        start_date = last_date - timedelta(days=days)

        # Historische Daten für den Zeitraum ab dem Startdatum laden
        data = StockData.objects.filter(stock=self.stock, date__range=[start_date, last_date]).order_by('date')

        self.df = pd.DataFrame(list(data.values()))

        if not self.df.empty:
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close', 'volume']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(float)

    def _adjust_window_based_on_volatility(self, base_days):
        try:
            recent_data = StockData.objects.filter(stock=self.stock).order_by('-date')[:60]
            if recent_data.count() < 30:
                return base_days

            df = pd.DataFrame(list(recent_data.values()))
            df['close_price'] = df['close_price'].astype(float)
            df['high_price'] = df['high_price'].astype(float)
            df['low_price'] = df['low_price'].astype(float)

            high_low = df['high_price'] - df['low_price']
            high_close = abs(df['high_price'] - df['close_price'].shift())
            low_close = abs(df['low_price'] - df['close_price'].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            last_close = df['close_price'].iloc[-1]
            atr_pct = (atr / last_close) * 100 if last_close else 5

            print(f"[VOLATILITY-DEBUG] ATR%: {atr_pct:.2f} → ", end="")

            if atr_pct < 2.0:
                print(f"niedrig → Fenster verdoppelt auf {base_days * 2}")
                return base_days * 2
            elif atr_pct < 4.0:
                print(f"moderat → Fenster erweitert auf {int(base_days * 1.5)}")
                return int(base_days * 1.5)
            else:
                print(f"hoch → Standardfenster {base_days} beibehalten")
                return base_days

        except Exception as e:
            print(f"[WARN] Volatilitätsanpassung fehlgeschlagen: {e}")
            return base_days

    def calculate_indicators(self, include_advanced=False):
        # Stelle sicher, dass Berechnungen in der richtigen Reihenfolge erfolgen
        try:
            # Grundlegende Berechnungen
            self._calculate_rsi()
            self._calculate_sma()
            self._calculate_macd()
            self._calculate_bollinger_bands()
            self._calculate_stochastic()
            self._calculate_adx()
            self._calculate_ichimoku()
            self._calculate_obv()
            self._calculate_atr()
            self._calculate_roc()
            self._calculate_psar()

            # Debug-Ausgabe aller berechneten Spalten
            print("INDIKATOR-DEBUG: Berechnete Spalten:", list(self.df.columns))

            # Optional: Erweiterte Indikatoren
            if include_advanced:
                advanced_indicators = AdvancedIndicators(self.df)
                self.df = advanced_indicators.calculate_all_indicators()

            return self.df

        except Exception as e:
            print(f"FEHLER bei Indikatorberechnung: {str(e)}")
            import traceback
            traceback.print_exc()
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

        self.df['obv'] = (np.sign(price_diff) * volume).fillna(0).cumsum()


    def _calculate_atr(self, period=14):
        """Berechnet den Average True Range (ATR)"""
        high_low = self.df['high_price'] - self.df['low_price']
        high_close = abs(self.df['high_price'] - self.df['close_price'].shift())
        low_close = abs(self.df['low_price'] - self.df['close_price'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        self.df['atr'] = true_range.rolling(window=period).mean()

    def calculate_technical_score(self):
        self.calculate_indicators(include_advanced=True)

        print("DEBUG: Methode calculate_technical_score gestartet")
        print(f"DEBUG: Stock Symbol {self.stock.symbol}")
        print("DEBUG: Verfügbare Spalten:", list(self.df.columns))
        print(f"DEBUG: DataFrame Länge: {len(self.df)}")
        print(f"DEBUG: Letzte Zeile:\n{self.df.iloc[-1]}")

        indicator_cols = [
            'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200', 'close_price'
        ]
        print("DEBUG: Indikatorwerte:")
        for col in indicator_cols:
            val = self.df[col].iloc[-1] if col in self.df.columns else "NICHT GEFUNDEN"
            print(f"{col}: {val}")

        required_cols = [
            'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200',
            'bollinger_upper', 'bollinger_lower', 'stoch_k', 'stoch_d',
            'senkou_span_a', 'senkou_span_b', 'adx', '+di', '-di'
        ]

        if self.df.empty:
            return None

        self.calculate_indicators()

        latest = self.df.dropna(subset=required_cols).iloc[-1:]
        if latest.empty:
            return None

        latest = latest.squeeze()
        score = 50
        signals = []

        weights = {
            'rsi': 1.5,
            'macd': 1.3,
            'sma': {20: 0.8, 50: 1.0, 200: 1.5},
            'bollinger': 1.2,
            'stochastic': 0.9,
            'ichimoku': 1.1,
            'adx': 1.0
        }

        # === RSI ===
        rsi = latest['rsi']
        if rsi < 30:
            bonus = min((30 - rsi) * weights['rsi'], 15)
            score += bonus
            signals.append(("RSI", "BUY", f"Überverkauft ({rsi:.2f}) → +{bonus:.1f}"))
        elif rsi > 70:
            malus = min((rsi - 70) * weights['rsi'], 15)
            score -= malus
            signals.append(("RSI", "SELL", f"Überkauft ({rsi:.2f}) → -{malus:.1f}"))
        print(f"[DEBUG] Score nach RSI: {score:.2f}")

        # === MACD ===
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        macd_diff = macd - macd_signal

        if macd > macd_signal:
            macd_boost = min(abs(macd_diff) * weights['macd'] * 10, 10)
            score += macd_boost
            signals.append(("MACD", "BUY", f"Bullish Momentum ({macd_diff:.2f}) → +{macd_boost:.1f}"))
        else:
            macd_penalty = min(abs(macd_diff) * weights['macd'] * 10, 10)
            score -= macd_penalty
            signals.append(("MACD", "SELL", f"Bearish Momentum ({macd_diff:.2f}) → -{macd_penalty:.1f}"))
        print(f"[DEBUG] Score nach MACD: {score:.2f}")

        # === SMA ===
        close_price = latest['close_price']
        for sma_p in [20, 50, 200]:
            sma_val = latest[f'sma_{sma_p}']
            weight = weights['sma'][sma_p]
            if close_price > sma_val:
                boost = 5 * weight
                score += boost
                signals.append((f"SMA {sma_p}", "BUY", f"Preis über SMA {sma_p} → +{boost:.1f}"))
            else:
                penalty = 5 * weight
                score -= penalty
                signals.append((f"SMA {sma_p}", "SELL", f"Preis unter SMA {sma_p} → -{penalty:.1f}"))
        print(f"[DEBUG] Score nach SMA: {score:.2f}")

        # === Bollinger Bands ===
        upper = latest['bollinger_upper']
        lower = latest['bollinger_lower']
        width = upper - lower
        pos = ((close_price - lower) / width) if width > 0 else 0.5

        if pos <= 0.2:
            boost = 7.5 * weights['bollinger']
            score += boost
            signals.append(("Bollinger", "BUY", f"Preis nahe unterem Band → +{boost:.1f}"))
        elif pos >= 0.8:
            penalty = 7.5 * weights['bollinger']
            score -= penalty
            signals.append(("Bollinger", "SELL", f"Preis nahe oberem Band → -{penalty:.1f}"))
        print(f"[DEBUG] Score nach Bollinger: {score:.2f}")

        # === Stochastik ===
        k, d = latest['stoch_k'], latest['stoch_d']
        if k < 20 and d < 20:
            boost = 5 * weights['stochastic']
            score += boost
            signals.append(("Stochastik", "BUY", f"Überverkauft → +{boost:.1f}"))
        elif k > 80 and d > 80:
            penalty = 5 * weights['stochastic']
            score -= penalty
            signals.append(("Stochastik", "SELL", f"Überkauft → -{penalty:.1f}"))
        print(f"[DEBUG] Score nach Stochastik: {score:.2f}")

        # === Ichimoku ===
        if close_price > latest['senkou_span_a'] and close_price > latest['senkou_span_b']:
            boost = 10 * weights['ichimoku']
            score += boost
            signals.append(("Ichimoku", "BUY", f"Preis über Cloud → +{boost:.1f}"))
        elif close_price < latest['senkou_span_a'] and close_price < latest['senkou_span_b']:
            penalty = 10 * weights['ichimoku']
            score -= penalty
            signals.append(("Ichimoku", "SELL", f"Preis unter Cloud → -{penalty:.1f}"))
        print(f"[DEBUG] Score nach Ichimoku: {score:.2f}")

        # === ADX ===
        adx = latest['adx']
        plus_di = latest['+di']
        minus_di = latest['-di']
        print(f"[DEBUG] ADX: {adx:.2f}, +DI: {plus_di:.2f}, -DI: {minus_di:.2f}")

        if adx > 25:
            if plus_di > minus_di:
                boost = score * 0.1
                score *= 1.1
                signals.append(("ADX", "BUY", f"Starker Aufwärtstrend → +{boost:.1f}"))
            else:
                penalty = score * 0.1
                score *= 0.9
                signals.append(("ADX", "SELL", f"Starker Abwärtstrend → -{penalty:.1f}"))
        print(f"[DEBUG] Score nach ADX: {score:.2f}")

        # Begrenzung
        score = max(0, min(100, score))

        # Kombi-Signal
        if rsi < 30 and macd > macd_signal and close_price > latest['sma_20']:
            score += 10
            signals.append(("Multi-Indikator", "BUY", "Stark bullisches Kombi-Signal → +10"))
        print(f"[DEBUG] Score nach Kombi-Check: {score:.2f}")

        # Confluence Score
        buys = sum(1 for s in signals if s[1] == 'BUY')
        sells = sum(1 for s in signals if s[1] == 'SELL')
        total = buys + sells
        confluence_score = round((buys / total) * 10) if total > 0 else 5

        # Empfehlung
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

        # Ausgabe der Signale
        print("=== TECHNISCHE SIGNALAUSWERTUNG ===")
        for sig in signals:
            print(f"{sig[0]} | {sig[1]} | {sig[2]}")
        print(f"FINALER SCORE: {round(score, 2)} | Empfehlung: {recommendation}")

        return {
            'score': round(score, 2),
            'recommendation': recommendation,
            'signals': signals,
            'confluence_score': confluence_score,
            'details': {
                k: float(v) for k, v in latest.items()
                if k in required_cols
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
        if result is None:
            print(f"[ERROR] Kein technisches Ergebnis für {self.stock.symbol}, nichts gespeichert.")
            return None

        # Sicherstellen, dass die Daten nach Datum sortiert sind (neueste zuletzt)
        if 'date' in self.df.columns:
            self.df = self.df.sort_values(by='date')
            print(f"DataFrame für Indikator-Speicherung sortiert: {self.df['date'].iloc[0]} bis {self.df['date'].iloc[-1]}")

        latest_date = self.df['date'].iloc[-1]

        # Direkt die letzten Werte aus dem DataFrame extrahieren
        latest_row = self.df.iloc[-1]

        # Debug-Ausgabe der letzten Werte
        print(f"Letzte Werte für {self.stock.symbol}:")
        for col in ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200', 'bollinger_upper', 'bollinger_lower']:
            if col in latest_row:
                print(f"  {col}: {latest_row[col]}")

        details = result.get('details', {})

        # Verwende die letzten Werte aus dem DataFrame, falls verfügbar
        rsi_value = latest_row.get('rsi') if 'rsi' in latest_row else details.get('rsi')
        macd_value = latest_row.get('macd') if 'macd' in latest_row else details.get('macd')
        macd_signal = latest_row.get('macd_signal') if 'macd_signal' in latest_row else details.get('macd_signal')
        sma_20 = latest_row.get('sma_20') if 'sma_20' in latest_row else details.get('sma_20')
        sma_50 = latest_row.get('sma_50') if 'sma_50' in latest_row else details.get('sma_50')
        sma_200 = latest_row.get('sma_200') if 'sma_200' in latest_row else details.get('sma_200')
        bollinger_upper = latest_row.get('bollinger_upper') if 'bollinger_upper' in latest_row else details.get('bollinger_upper')
        bollinger_lower = latest_row.get('bollinger_lower') if 'bollinger_lower' in latest_row else details.get('bollinger_lower')

        analysis_result, created = AnalysisResult.objects.update_or_create(
            stock=self.stock,
            date=latest_date.date() if hasattr(latest_date, 'date') else latest_date,  # Konvertiere zu date, falls datetime
            defaults={
                'technical_score': result['score'],
                'recommendation': result['recommendation'],
                'rsi_value': rsi_value,
                'macd_value': macd_value,
                'macd_signal': macd_signal,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'confluence_score': int((result['confluence_score'] + 10) / 20 * 100)  # Skaliert auf 0–100%
            }
        )

        print(f"Analyse-Ergebnis gespeichert mit RSI: {analysis_result.rsi_value}, MACD: {analysis_result.macd_value}")
        return analysis_result

    def calculate_advanced_indicators(self):
        """Berechnet alle fortgeschrittenen Indikatoren"""
        advanced_indicators = AdvancedIndicators(self.df)
        self.df = advanced_indicators.calculate_all_indicators()
        return self.df


def _calculate_indicators_for_dataframe(df):
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
        self.df['ha_trend_strength'] = np.where(
            self.df['ha_close'] > self.df['ha_open'],
            1,  # Grüne Kerze = Aufwärtstrend
            -1  # Rote Kerze = Abwärtstrend
        )

        return self.df

    def calculate_fibonacci_levels(self, window=100):
        """
        Berechnet Fibonacci-Retracement-Levels basierend auf den letzten X Tagen.
        """
        # Stelle sicher, dass genug Daten vorhanden sind
        if len(self.df) < window:
            window = max(len(self.df) // 2, 2)  # Mindestens 2 Datenpunkte

        # Bestimme lokale Hochs und Tiefs im letzten Fenster
        segment = self.df.tail(window)

        local_max = segment['high_price'].max()
        local_min = segment['low_price'].min()

        # Prüfe, ob die Werte gültig sind
        if pd.isna(local_max) or pd.isna(local_min) or local_max <= local_min:
            # Fallback: Verwende einfach den höchsten und niedrigsten Wert im gesamten DataFrame
            local_max = self.df['high_price'].max()
            local_min = self.df['low_price'].min()

            # Wenn immer noch ungültig, setze Standard-Werte
            if pd.isna(local_max) or pd.isna(local_min) or local_max <= local_min:
                if 'close_price' in self.df.columns and len(self.df) > 0:
                    most_recent = float(self.df['close_price'].iloc[-1])
                    if not pd.isna(most_recent) and most_recent > 0:
                        local_max = most_recent * 1.1  # 10% über aktuellem Preis
                        local_min = most_recent * 0.9  # 10% unter aktuellem Preis
                    else:
                        local_max = 110
                        local_min = 90
                else:
                    local_max = 110
                    local_min = 90

        # Fibonacci-Levels: 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

        # Bestimme den aktuellen Trend
        if 'close_price' in self.df.columns and len(self.df) >= 20:
            recent_trend = self.df['close_price'].tail(20).diff().mean()
            trend_is_up = recent_trend >= 0
        else:
            # Fallback: Nehme an, dass der Trend aufwärts ist
            trend_is_up = True

        # Retracement-Levels für beide Richtungen berechnen
        for level in fib_levels:
            # Retracement-Levels nach unten (von Hochs)
            level_val = int(level * 1000)
            self.df[f'fib_down_{level_val}'] = (
                    local_max - (local_max - local_min) * level
            )

            # Retracement-Levels nach oben (von Tiefs)
            self.df[f'fib_up_{level_val}'] = (
                    local_min + (local_max - local_min) * level
            )

        # Fibonacci-Extensions: 1.618, 2.618, 4.236
        ext_levels = [1.618, 2.618, 4.236]

        for level in ext_levels:
            level_val = int(level * 1000)
            # Extension-Levels nach oben
            self.df[f'fib_ext_up_{level_val}'] = (
                    local_min + (local_max - local_min) * level
            )

            # Extension-Levels nach unten
            self.df[f'fib_ext_down_{level_val}'] = (
                    local_max - (local_max - local_min) * level
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

        ilocs_max = argrelextrema(self.df['high_price'].values, np.greater, order=order)[0]
        ilocs_min = argrelextrema(self.df['low_price'].values, np.less, order=order)[0]

        self._detect_double_top(ilocs_max)
        self._detect_double_bottom(ilocs_min)
        self._detect_head_and_shoulders(ilocs_max)
        self._detect_inverse_head_and_shoulders(ilocs_min)
        self._detect_triangles(window)

    def _detect_double_top(self, ilocs_max):
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

    def _detect_double_bottom(self, ilocs_min):
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

    def _detect_head_and_shoulders(self, ilocs_max):
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

    def _detect_inverse_head_and_shoulders(self, ilocs_min):
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

    def _detect_triangles(self, window):
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
        Verbesserte Version: Wendepunkte über gesamte Zeitreihe.
        """
        if len(self.df) < window:
            window = len(self.df) // 2

        if window < 10:
            window = 10  # Minimal sinnvoller Wert

        # Ordnung kleiner wählen für feinere Wendepunkte
        order = max(1, window // 10)

        # Lokale Hochs und Tiefs finden
        local_max_idx = argrelextrema(self.df['high_price'].values, np.greater, order=order)[0]
        local_min_idx = argrelextrema(self.df['low_price'].values, np.less, order=order)[0]

        # Initialisieren
        self.df['elliott_wave_point'] = 0

        if len(local_max_idx) > 0:
            self.df.loc[self.df.index[local_max_idx], 'elliott_wave_point'] = 1  # Hochpunkte

        if len(local_min_idx) > 0:
            self.df.loc[self.df.index[local_min_idx], 'elliott_wave_point'] = -1  # Tiefpunkte

        # Versuche, abwechselnde Sequenzen von Wendepunkten zu erkennen
        recent_points = self.df[self.df['elliott_wave_point'] != 0].tail(10)

        if len(recent_points) >= 5:
            sequence = recent_points['elliott_wave_point'].values[-5:]

            alternating = True
            for i in range(1, 5):
                if sequence[i] == sequence[i - 1]:  # Kein Wechsel zwischen Hoch und Tief
                    alternating = False
                    break

            if alternating:
                # Markiere letztes Signal zusätzlich
                self.df.loc[recent_points.index[-1], 'elliott_wave_pattern'] = sequence[-1]

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
