# stock_analyzer/data_service.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .models import Stock, StockData
from decimal import Decimal

class StockDataService:
    @staticmethod
    def update_stock_data(symbol, days=730):
        """Lädt historische Daten für eine Aktie herunter"""
        try:
            # Aktie in der Datenbank abrufen oder erstellen
            stock, created = Stock.objects.get_or_create(
                symbol=symbol.upper(),
                defaults={'name': symbol.upper()}  # Vorläufiger Name, wird später aktualisiert
            )

            # Ticker-Objekt erstellen
            ticker = yf.Ticker(symbol)

            # Name und Sektor aktualisieren, falls verfügbar
            if hasattr(ticker, 'info') and ticker.info.get('longName'):
                stock.name = ticker.info.get('longName', stock.name)
                stock.sector = ticker.info.get('sector', stock.sector)
                stock.save()

            # Historische Daten abrufen (2 Jahre!)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            hist_data = ticker.history(start=start_date, end=end_date, interval="1d")

            if hist_data.empty:
                return False, f"Keine Daten für {symbol} gefunden."

            # Kursdaten in die Datenbank speichern
            for index, row in hist_data.iterrows():
                date = index.date()
                StockData.objects.update_or_create(
                    stock=stock,
                    date=date,
                    defaults={
                        'open_price': Decimal(str(row['Open'])),
                        'high_price': Decimal(str(row['High'])),
                        'low_price': Decimal(str(row['Low'])),
                        'close_price': Decimal(str(row['Close'])),
                        'adjusted_close': Decimal(str(row.get('Adj Close', row['Close']))),
                        'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                    }
                )

            return True, f"Daten für {symbol} erfolgreich aktualisiert."

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
