# stock_analyzer/market_analysis.py
import numpy as np
import pandas as pd
from django.db.models import Avg

from .analysis import TechnicalAnalyzer
from .models import Stock, StockData


class MarketAnalyzer:
    @staticmethod
    def calculate_correlations(symbols, days=90):
        """Berechnet die Korrelationsmatrix zwischen mehreren Aktien"""
        if len(symbols) < 2:
            return None

        # Dictionary für die Kursdaten der einzelnen Aktien
        stock_data = {}

        for symbol in symbols:
            try:
                stock = Stock.objects.get(symbol=symbol)
                data = StockData.objects.filter(stock=stock).order_by('-date')[:days]

                if not data.exists():
                    continue  # Überspringe Aktien ohne Daten

                # Umwandeln in eine Zeitreihe
                prices = pd.Series(
                    [float(d.close_price) for d in data],
                    index=[d.date for d in data]
                ).sort_index()

                stock_data[symbol] = prices
            except Stock.DoesNotExist:
                continue
            except Exception as e:
                print(f"Fehler beim Laden der Daten für {symbol}: {str(e)}")
                continue

        if len(stock_data) < 2:
            # Nicht genug Daten für eine Korrelation
            return None

        try:
            # Dataframe aus den Zeitreihen erstellen
            df = pd.DataFrame(stock_data)

            # Fehlende Werte entfernen
            df = df.dropna()

            if df.empty or len(df) < 2:
                return None

            # Korrelationsmatrix berechnen
            return df.corr()
        except Exception as e:
            print(f"Fehler bei der Korrelationsberechnung: {str(e)}")
            return None

    @staticmethod
    def sector_performance(days=30):
        """Analysiert die Performance nach Sektoren"""
        sectors = Stock.objects.exclude(sector=None).values_list('sector', flat=True).distinct()

        sector_performance = {}

        for sector in sectors:
            stocks = Stock.objects.filter(sector=sector)
            performance_sum = 0
            count = 0

            for stock in stocks:
                # Neuesten und ältesten Preis im Zeitraum abrufen
                # WICHTIG: Slicing und anschließendes last() verursachen Fehler
                # Daher holen wir die Daten vollständig und arbeiten mit der Liste
                recent_prices = list(StockData.objects.filter(stock=stock).order_by('-date')[:days])

                if len(recent_prices) > 0:
                    newest_price = float(recent_prices[0].close_price)
                    # Wenn nicht genügend Daten vorhanden sind, nehmen wir einfach den ältesten verfügbaren
                    oldest_price = float(recent_prices[-1].close_price) if len(recent_prices) > 1 else newest_price

                    # Performance berechnen
                    if oldest_price > 0:
                        performance = (newest_price - oldest_price) / oldest_price * 100
                        performance_sum += performance
                        count += 1

            if count > 0:
                avg_performance = performance_sum / count
                sector_performance[sector] = {
                    'performance': avg_performance,
                    'stock_count': count
                }

        # Nach Performance sortieren
        sorted_performance = dict(sorted(
            sector_performance.items(),
            key=lambda x: x[1]['performance'],
            reverse=True
        ))

        return sorted_performance

    @staticmethod
    def market_breadth():
        """Analysiert die Marktbreite (Anzahl der Aktien über/unter MA50)"""
        stocks = Stock.objects.all()

        above_ma50 = 0
        below_ma50 = 0
        total_analyzed = 0

        for stock in stocks:
            try:
                # Aktuelle Daten für die Berechnung holen
                recent_data = StockData.objects.filter(stock=stock).order_by('-date')[:50]

                if recent_data.count() >= 50:
                    # Letzter Schlusskurs
                    last_price = float(recent_data.first().close_price)

                    # MA50 berechnen
                    prices = [float(d.close_price) for d in recent_data]
                    ma50 = sum(prices) / len(prices)

                    if last_price > ma50:
                        above_ma50 += 1
                    else:
                        below_ma50 += 1

                    total_analyzed += 1
            except Exception:
                continue

        if total_analyzed > 0:
            above_percent = (above_ma50 / total_analyzed) * 100
            below_percent = (below_ma50 / total_analyzed) * 100

            result = {
                'above_ma50': above_ma50,
                'below_ma50': below_ma50,
                'total_analyzed': total_analyzed,
                'above_percent': above_percent,
                'below_percent': below_percent
            }

            # Marktlage interpretieren
            if above_percent >= 70:
                result['market_state'] = 'Bullisch'
            elif above_percent <= 30:
                result['market_state'] = 'Bearisch'
            else:
                result['market_state'] = 'Neutral'

            return result

        return None

# stock_analyzer/market_analysis.py

class TraditionalAnalyzer:

    @staticmethod
    def evaluate_traditional_performance(symbol):
        """Neue intelligente Bewertung einer Aktie basierend auf technischer Analyse und Marktbreite"""
        try:
            analyzer = TechnicalAnalyzer(symbol)
            analyzer.calculate_indicators()

            df = analyzer.df
            latest = df.iloc[-1]

            close_price = latest.get('close_price', 1)

            # 1. RSI Bewertung
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                rsi_score = 90  # Stark überverkauft -> große Chance
            elif rsi > 70:
                rsi_score = 30  # Stark überkauft -> Risiko
            else:
                rsi_score = 70  # Neutral

            # 2. MACD Bewertung
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_score = 90 if macd > macd_signal else 40

            # 3. Volatilität Bewertung (ATR anstatt Standardabweichung)
            if 'atr' in df.columns:
                atr = latest.get('atr', 0)
                atr_percentage = atr / close_price if close_price else 0
                if atr_percentage < 0.02:
                    volatility_score = 85  # Ruhig = Stabil
                elif atr_percentage > 0.05:
                    volatility_score = 40  # Sehr volatil
                else:
                    volatility_score = 65  # Normal
            else:
                volatility_score = 65  # Fallback

            # 4. Trendbewertung (SMA 20 vs SMA 50)
            sma_20 = latest.get('sma_20', close_price)
            sma_50 = latest.get('sma_50', close_price)
            trend_score = 85 if sma_20 > sma_50 else 45

            # 5. Marktbreite (Market Breadth)
            market_breadth = MarketAnalyzer.market_breadth()
            if market_breadth:
                above_percent = market_breadth.get('above_percent', 50)
                if above_percent >= 70:
                    market_breadth_score = 85  # Bullischer Gesamtmarkt
                elif above_percent <= 30:
                    market_breadth_score = 40  # Bärischer Gesamtmarkt
                else:
                    market_breadth_score = 65  # Neutraler Markt
            else:
                market_breadth_score = 65  # Fallback-Wert

            # --- Finales Scoring nach deiner Gewichtung ---
            final_score = (
                rsi_score * 0.20 +
                macd_score * 0.25 +
                volatility_score * 0.15 +
                trend_score * 0.25 +
                market_breadth_score * 0.15
            )

            return {
                'accuracy': round(rsi_score, 1),
                'return': round(macd_score, 1),
                'speed': round(volatility_score, 1),
                'adaptability': round(trend_score, 1),
                'robustness': round(market_breadth_score, 1),
                'final_score': round(final_score, 1)
            }

        except Exception as e:
            print(f"Fehler in TraditionalAnalyzer: {str(e)}")
            return {
                'accuracy': 65,
                'return': 70,
                'speed': 75,
                'adaptability': 60,
                'robustness': 65,
                'final_score': 67
            }


