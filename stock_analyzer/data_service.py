# stock_analyzer/data_service.py
import yfinance as yf
from datetime import datetime, timedelta
from .models import Stock, StockData


class StockDataService:
    @staticmethod
    def update_stock_data(symbol, days=365):
        """Lädt historische Daten für eine Aktie herunter"""
        try:
            # Aktie in der Datenbank abrufen oder erstellen
            stock, created = Stock.objects.get_or_create(
                symbol=symbol.upper(),
                defaults={'name': symbol.upper()}  # Vorläufiger Name, wird später aktualisiert
            )

            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Name aktualisieren, falls verfügbar
            if hasattr(ticker, 'info') and 'longName' in ticker.info:
                stock.name = ticker.info['longName']
                if 'sector' in ticker.info:
                    stock.sector = ticker.info['sector']
                stock.save()

            # Historische Daten abrufen
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            hist_data = ticker.history(start=start_date, end=end_date)

            if hist_data.empty:
                return False, f"Keine Daten für {symbol} gefunden"

            # Daten in die Datenbank speichern
            for index, row in hist_data.iterrows():
                date = index.date()
                StockData.objects.update_or_create(
                    stock=stock,
                    date=date,
                    defaults={
                        'open_price': row['Open'],
                        'high_price': row['High'],
                        'low_price': row['Low'],
                        'close_price': row['Close'],
                        'adjusted_close': row['Close'],  # Yahoo Finance gibt manchmal keine adj close zurück
                        'volume': row['Volume']
                    }
                )

            return True, f"Daten für {symbol} erfolgreich aktualisiert"
        except Exception as e:
            return False, f"Fehler beim Abrufen der Daten für {symbol}: {str(e)}"

    @staticmethod
    def update_multiple_stocks(symbols):
        """Aktualisiert Daten für mehrere Aktien"""
        results = {}
        for symbol in symbols:
            success, message = StockDataService.update_stock_data(symbol)
            results[symbol] = {'success': success, 'message': message}
        return results