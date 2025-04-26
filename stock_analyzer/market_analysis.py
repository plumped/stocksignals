# stock_analyzer/market_analysis.py
import numpy as np
import pandas as pd
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