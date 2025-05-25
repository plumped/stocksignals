# stock_analyzer/data_service.py
import pandas as pd
import numpy as np
from twelvedata import TDClient
from datetime import datetime, timedelta
import time
import functools
import random
from django.core.cache import cache
from django.db import transaction, OperationalError
from .models import Stock, StockData
from decimal import Decimal
from .config import TWELVEDATA_API_KEY

# Cache-Dauer in Sekunden (24 Stunden)
CACHE_DURATION = 86400

# Rate-Limit-Einstellungen (Anfragen pro Minute)
RATE_LIMIT = 8
RATE_LIMIT_PERIOD = 60

# Database retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 0.1  # seconds
MAX_RETRY_DELAY = 2.0  # seconds

class StockDataService:
    _last_request_time = 0
    _request_count = 0
    _td_client = None

    @staticmethod
    def retry_on_db_lock(operation_name="database operation"):
        """
        Execute a database operation with retry logic for handling database locks.

        Args:
            operation_name (str): Name of the operation for logging purposes

        Usage:
            with StockDataService.retry_on_db_lock("update stock data"):
                StockData.objects.update_or_create(...)
        """
        class RetryContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is OperationalError and "database is locked" in str(exc_val).lower():
                    retries = 0
                    while retries < MAX_RETRIES:
                        retries += 1
                        # Calculate delay with exponential backoff and jitter
                        delay = min(INITIAL_RETRY_DELAY * (2 ** (retries - 1)), MAX_RETRY_DELAY)
                        # Add jitter to avoid thundering herd problem
                        delay = delay * (0.5 + random.random())
                        print(f"Database locked during {operation_name}, retrying in {delay:.2f} seconds (attempt {retries}/{MAX_RETRIES})")
                        time.sleep(delay)

                        try:
                            # Try the operation again in a new transaction
                            with transaction.atomic():
                                # The operation will be retried when the context is re-entered
                                return True  # Suppress the exception and retry
                        except OperationalError as e:
                            if "database is locked" not in str(e).lower() or retries >= MAX_RETRIES:
                                # If it's not a lock error or we've exceeded max retries, let the exception propagate
                                print(f"Database error during {operation_name} after {retries} retries: {str(e)}")
                                return False  # Don't suppress the exception

                    # If we've exhausted all retries
                    print(f"Failed to execute {operation_name} after {MAX_RETRIES} retries")
                    return False  # Don't suppress the exception

                # For other types of exceptions, don't suppress
                return False

        return RetryContextManager()

    @staticmethod
    def _detect_outliers(df, columns=None, z_threshold=3.0, iqr_multiplier=1.5):
        """
        Detect outliers in stock data using multiple methods.

        Args:
            df (pandas.DataFrame): DataFrame containing stock data
            columns (list): List of columns to check for outliers. If None, checks all numeric columns.
            z_threshold (float): Threshold for Z-score method. Values with absolute Z-score above this are outliers.
            iqr_multiplier (float): Multiplier for IQR method. Values outside Q1-multiplier*IQR and Q3+multiplier*IQR are outliers.

        Returns:
            pandas.DataFrame: DataFrame with additional columns indicating outliers
        """
        if df.empty:
            return df

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Add a column to track if any outliers were detected
        result_df['has_outliers'] = False

        for col in columns:
            if col not in df.columns:
                continue

            # Skip columns with all NaN values
            if df[col].isna().all():
                continue

            # Z-score method
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            z_outliers = z_scores > z_threshold

            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (df[col] < (Q1 - iqr_multiplier * IQR)) | (df[col] > (Q3 + iqr_multiplier * IQR))

            # Combine methods - a point is an outlier if detected by either method
            outliers = z_outliers | iqr_outliers

            # Add outlier indicator column
            result_df[f'{col}_outlier'] = outliers

            # Update the overall outlier flag
            result_df['has_outliers'] = result_df['has_outliers'] | outliers

        return result_df

    @staticmethod
    def _handle_outliers(df, method='winsorize', columns=None, winsorize_limits=(0.05, 0.05)):
        """
        Handle outliers in stock data using the specified method.

        Args:
            df (pandas.DataFrame): DataFrame containing stock data with outlier indicators
            method (str): Method to handle outliers. Options: 'winsorize', 'remove', 'impute'
            columns (list): List of columns to handle outliers for. If None, handles all columns with outlier indicators.
            winsorize_limits (tuple): Limits for winsorization (lower, upper) as fractions of data to replace

        Returns:
            pandas.DataFrame: DataFrame with outliers handled
        """
        if df.empty or 'has_outliers' not in df.columns:
            return df

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # If no columns specified, find all columns with outlier indicators
        if columns is None:
            columns = [col.replace('_outlier', '') for col in df.columns if col.endswith('_outlier')]

        for col in columns:
            outlier_col = f'{col}_outlier'
            if outlier_col not in df.columns or col not in df.columns:
                continue

            # Skip columns with all NaN values
            if df[col].isna().all():
                continue

            if method == 'winsorize':
                # Winsorize: cap extreme values at specified percentiles
                lower_limit = df[col].quantile(winsorize_limits[0])
                upper_limit = df[col].quantile(1 - winsorize_limits[1])
                result_df.loc[df[outlier_col], col] = df.loc[df[outlier_col], col].clip(lower=lower_limit, upper=upper_limit)

            elif method == 'remove':
                # Remove: set outliers to NaN (they'll be handled later)
                result_df.loc[df[outlier_col], col] = np.nan

            elif method == 'impute':
                # Impute: replace outliers with the median of non-outlier values
                median_value = df.loc[~df[outlier_col], col].median()
                result_df.loc[df[outlier_col], col] = median_value

        return result_df

    @classmethod
    def get_client(cls):
        """Gibt eine Singleton-Instanz des TDClient zurück"""
        if cls._td_client is None:
            api_key = TWELVEDATA_API_KEY
            if not api_key:
                raise ValueError("Twelvedata API-Schlüssel nicht gefunden")
            cls._td_client = TDClient(apikey=api_key)
        return cls._td_client

    @classmethod
    def rate_limit(cls):
        """Implementiert Rate-Limiting für API-Anfragen"""
        current_time = time.time()
        time_passed = current_time - cls._last_request_time

        # Zurücksetzen des Zählers, wenn die Zeitperiode abgelaufen ist
        if time_passed > RATE_LIMIT_PERIOD:
            cls._request_count = 0
            cls._last_request_time = current_time

        # Wenn das Limit erreicht ist, warten
        if cls._request_count >= RATE_LIMIT:
            sleep_time = RATE_LIMIT_PERIOD - time_passed
            if sleep_time > 0:
                print(f"Rate-Limit erreicht. Warte {sleep_time:.2f} Sekunden...")
                time.sleep(sleep_time)
                cls._request_count = 0
                cls._last_request_time = time.time()

        # Zähler erhöhen
        cls._request_count += 1
        if cls._request_count == 1:
            cls._last_request_time = time.time()

    @staticmethod
    def cache_key(method_name, *args, **kwargs):
        """Generiert einen Cache-Schlüssel basierend auf Methodenname und Parametern"""
        key_parts = [method_name]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "_".join(key_parts)

    @staticmethod
    def cached(duration=CACHE_DURATION):
        """Dekorator für Caching von Methoden"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Cache-Schlüssel generieren
                key = StockDataService.cache_key(func.__name__, *args, **kwargs)

                # Versuchen, aus dem Cache zu laden
                cached_result = cache.get(key)
                if cached_result is not None:
                    print(f"Cache-Treffer für {key}")
                    return cached_result

                # Funktion ausführen und Ergebnis cachen
                result = func(*args, **kwargs)
                cache.set(key, result, duration)
                return result
            return wrapper
        return decorator

    @staticmethod
    def update_stock_data(symbol, days=730, force_update=False, use_live_data=False):
        """Lädt historische oder Live-Daten für eine Aktie über Twelvedata herunter"""
        try:
            # Interval basierend auf Live-Daten-Option festlegen
            interval = "1min" if use_live_data else "1day"

            # Cache-Schlüssel für diese Aktie (mit Interval-Information)
            cache_key = f"stock_data_{symbol.upper()}_{days}_{interval}"

            # Wenn nicht erzwungen, prüfen, ob Daten im Cache sind
            if not force_update:
                cached_result = cache.get(cache_key)
                if cached_result:
                    print(f"Verwende gecachte Daten für {symbol} (Interval: {interval})")
                    return cached_result

            # Aktie in der Datenbank abrufen oder erstellen
            stock, created = Stock.objects.get_or_create(
                symbol=symbol.upper(),
                defaults={'name': symbol.upper()}  # Vorläufiger Name, wird später aktualisiert
            )
            print(f"Stock {symbol}: {'neu erstellt' if created else 'existiert bereits'}")

            # Zeitraum berechnen
            end_date = datetime.now()
            # Bei Live-Daten kürzeren Zeitraum verwenden
            if use_live_data:
                # Für Live-Daten nur die letzten 1-2 Tage abrufen
                start_date = end_date - timedelta(days=2)
                outputsize = 1000  # Maximale Anzahl von Datenpunkten für Live-Daten
            else:
                start_date = end_date - timedelta(days=days)
                outputsize = days

            # Rate-Limiting anwenden
            StockDataService.rate_limit()

            # Twelvedata Client holen
            td = StockDataService.get_client()

            # Historische oder Live-Daten abrufen
            ts = td.time_series(
                symbol=symbol,
                interval=interval,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d %H:%M:%S') if use_live_data else end_date.strftime('%Y-%m-%d'),
                outputsize=outputsize
            )
            hist_data = ts.as_pandas()

            # Unternehmensinformationen abrufen (falls benötigt)
            try:
                # Rate-Limiting anwenden
                StockDataService.rate_limit()

                profile = td.profile(symbol=symbol).as_json()
                if profile:
                    stock.name = profile.get('name', stock.name)
                    stock.sector = profile.get('sector', stock.sector)
                    stock.save()
            except Exception as e:
                print(f"Warnung: Konnte Unternehmensprofil nicht abrufen: {str(e)}")

            print(f"Historische Daten für {symbol}: {len(hist_data)} Datenpunkte")

            if hist_data.empty:
                print(f"WARNUNG: Keine Daten für {symbol} gefunden.")
                result = (False, f"Keine Daten für {symbol} gefunden.")
                cache.set(cache_key, result, CACHE_DURATION)
                return result

            # Outlier-Erkennung und -Behandlung
            price_columns = ['open', 'high', 'low', 'close']

            # Detect outliers in price data
            hist_data_with_outliers = StockDataService._detect_outliers(
                hist_data, 
                columns=price_columns,
                z_threshold=3.5,  # Etwas höherer Schwellenwert für Finanzdaten
                iqr_multiplier=2.0  # Etwas höherer Multiplikator für Finanzdaten
            )

            # Count and log outliers
            outlier_count = hist_data_with_outliers['has_outliers'].sum()
            if outlier_count > 0:
                print(f"WARNUNG: {outlier_count} Ausreißer in den Daten für {symbol} gefunden.")

                # Log details about outliers
                for col in price_columns:
                    col_outliers = hist_data_with_outliers[f'{col}_outlier'].sum()
                    if col_outliers > 0:
                        print(f"  - {col_outliers} Ausreißer in {col}-Preisen")

                        # Show some examples of outliers
                        outlier_examples = hist_data_with_outliers[hist_data_with_outliers[f'{col}_outlier']]
                        if not outlier_examples.empty:
                            for idx, row in outlier_examples.head(3).iterrows():
                                date_str = idx if isinstance(idx, str) else idx.strftime('%Y-%m-%d')
                                print(f"    * {date_str}: {row[col]} (Z-Score: {abs((row[col] - hist_data[col].mean()) / hist_data[col].std()):.2f})")

            # Handle outliers using winsorization (cap extreme values)
            hist_data_cleaned = StockDataService._handle_outliers(
                hist_data_with_outliers,
                method='winsorize',
                columns=price_columns,
                winsorize_limits=(0.01, 0.01)  # Cap at 1st and 99th percentiles
            )

            # Kursdaten in die Datenbank speichern
            for index, row in hist_data_cleaned.iterrows():
                date = datetime.strptime(index, '%Y-%m-%d').date() if isinstance(index, str) else index.date()

                # Überprüfen, ob das Datum in der Zukunft liegt
                if date > datetime.now().date():
                    print(f"Überspringe Datenpunkt für {date}, da es in der Zukunft liegt")
                    continue

                # Überprüfen, ob das Volumen zu niedrig ist (könnte auf fehlerhafte Daten hindeuten)
                volume = int(row['volume']) if not pd.isna(row['volume']) else 0
                if volume < 1000:  # Annahme: Volumen unter 1000 ist verdächtig niedrig
                    print(f"Warnung: Niedriges Volumen ({volume}) für {stock.symbol} am {date}")

                # Check if this data point was an outlier
                was_outlier = False
                if 'has_outliers' in row and row['has_outliers']:
                    was_outlier = True
                    outlier_fields = []
                    for col in price_columns:
                        if f'{col}_outlier' in row and row[f'{col}_outlier']:
                            outlier_fields.append(col)
                    print(f"Korrigierter Ausreißer für {stock.symbol} am {date} in Feldern: {', '.join(outlier_fields)}")

                with StockDataService.retry_on_db_lock(f"update stock data for {stock.symbol} on {date}"):
                    with transaction.atomic():
                        StockData.objects.update_or_create(
                            stock=stock,
                            date=date,
                            defaults={
                                'open_price': Decimal(str(row['open'])),
                                'high_price': Decimal(str(row['high'])),
                                'low_price': Decimal(str(row['low'])),
                                'close_price': Decimal(str(row['close'])),
                                'adjusted_close': Decimal(str(row.get('adjusted_close', row['close']))),
                                'volume': volume
                            }
                        )

                # Small delay between database operations to reduce contention
                time.sleep(0.01)

            result = (True, f"Daten für {symbol} erfolgreich aktualisiert.")
            cache.set(cache_key, result, CACHE_DURATION)
            return result

        except Exception as e:
            print(f"Fehler beim Aktualisieren von {symbol}: {str(e)}")
            result = (False, f"Fehler beim Abrufen der Daten für {symbol}: {str(e)}")
            return result

    @staticmethod
    def update_multiple_stocks(symbols, days=730, force_update=False, use_live_data=False):
        """Aktualisiert Daten für mehrere Aktien mit optimierter Batch-Verarbeitung"""
        results = {}

        # Interval basierend auf Live-Daten-Option festlegen
        interval = "1min" if use_live_data else "1day"

        # Aktien in Gruppen von 5 aufteilen, um Batch-Verarbeitung zu optimieren
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            print(f"Verarbeite Batch {i//batch_size + 1}: {batch_symbols} (Interval: {interval})")

            # Zeitraum berechnen
            end_date = datetime.now()
            # Bei Live-Daten kürzeren Zeitraum verwenden
            if use_live_data:
                # Für Live-Daten nur die letzten 1-2 Tage abrufen
                start_date = end_date - timedelta(days=2)
                outputsize = 1000  # Maximale Anzahl von Datenpunkten für Live-Daten
            else:
                start_date = end_date - timedelta(days=days)
                outputsize = days

            # Aktien in der Datenbank abrufen oder erstellen
            stocks = {}
            for symbol in batch_symbols:
                stock, created = Stock.objects.get_or_create(
                    symbol=symbol.upper(),
                    defaults={'name': symbol.upper()}
                )
                stocks[symbol] = stock
                print(f"Stock {symbol}: {'neu erstellt' if created else 'existiert bereits'}")

            try:
                # Rate-Limiting anwenden
                StockDataService.rate_limit()

                # Twelvedata Client holen
                td = StockDataService.get_client()

                # Batch-Anfrage für Unternehmensinformationen
                batch_request = td.batch()
                for symbol in batch_symbols:
                    batch_request.profile(symbol=symbol)

                # Rate-Limiting anwenden
                StockDataService.rate_limit()

                # Batch-Anfrage ausführen
                batch_profile_data = batch_request.as_json()

                # Unternehmensinformationen verarbeiten
                for symbol in batch_symbols:
                    if symbol in batch_profile_data and batch_profile_data[symbol]:
                        profile = batch_profile_data[symbol]
                        stock = stocks[symbol]
                        stock.name = profile.get('name', stock.name)
                        stock.sector = profile.get('sector', stock.sector)
                        stock.save()

                # Batch-Anfrage für historische oder Live-Daten
                batch_request = td.batch()
                for symbol in batch_symbols:
                    batch_request.time_series(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d %H:%M:%S') if use_live_data else end_date.strftime('%Y-%m-%d'),
                        outputsize=outputsize
                    )

                # Rate-Limiting anwenden
                StockDataService.rate_limit()

                # Batch-Anfrage ausführen
                batch_hist_data = batch_request.as_json()

                # Historische Daten verarbeiten
                for symbol in batch_symbols:
                    try:
                        if symbol in batch_hist_data and 'values' in batch_hist_data[symbol]:
                            hist_data = pd.DataFrame(batch_hist_data[symbol]['values'])

                            # Spalten umbenennen, um dem Format von as_pandas() zu entsprechen
                            if not hist_data.empty:
                                hist_data.set_index('datetime', inplace=True)

                                print(f"Historische Daten für {symbol}: {len(hist_data)} Datenpunkte")

                                # Kursdaten in die Datenbank speichern
                                for index, row in hist_data.iterrows():
                                    date = datetime.strptime(index, '%Y-%m-%d').date() if isinstance(index, str) else index.date()

                                    # Überprüfen, ob das Datum in der Zukunft liegt
                                    if date > datetime.now().date():
                                        print(f"Überspringe Datenpunkt für {symbol} am {date}, da es in der Zukunft liegt")
                                        continue

                                    # Überprüfen, ob das Volumen zu niedrig ist (könnte auf fehlerhafte Daten hindeuten)
                                    volume = int(row['volume']) if not pd.isna(row['volume']) else 0
                                    if volume < 1000:  # Annahme: Volumen unter 1000 ist verdächtig niedrig
                                        print(f"Warnung: Niedriges Volumen ({volume}) für {symbol} am {date}")

                                    with StockDataService.retry_on_db_lock(f"update stock data for {symbol} on {date} (batch)"):
                                        with transaction.atomic():
                                            StockData.objects.update_or_create(
                                                stock=stocks[symbol],
                                                date=date,
                                                defaults={
                                                    'open_price': Decimal(str(row['open'])),
                                                    'high_price': Decimal(str(row['high'])),
                                                    'low_price': Decimal(str(row['low'])),
                                                    'close_price': Decimal(str(row['close'])),
                                                    'adjusted_close': Decimal(str(row.get('adjusted_close', row['close']))),
                                                    'volume': volume
                                                }
                                            )

                                    # Small delay between database operations to reduce contention
                                    time.sleep(0.01)

                                results[symbol] = {'success': True, 'message': f"Daten für {symbol} erfolgreich aktualisiert."}
                            else:
                                print(f"WARNUNG: Keine Daten für {symbol} gefunden.")
                                results[symbol] = {'success': False, 'message': f"Keine Daten für {symbol} gefunden."}
                        else:
                            print(f"WARNUNG: Keine Daten für {symbol} in der API-Antwort.")
                            results[symbol] = {'success': False, 'message': f"Keine Daten für {symbol} in der API-Antwort."}
                    except Exception as e:
                        print(f"Fehler bei der Verarbeitung von {symbol}: {str(e)}")
                        results[symbol] = {'success': False, 'message': f"Fehler bei der Verarbeitung: {str(e)}"}

            except Exception as e:
                print(f"Fehler bei der Batch-Verarbeitung: {str(e)}")
                for symbol in batch_symbols:
                    if symbol not in results:
                        results[symbol] = {'success': False, 'message': f"Fehler bei der Batch-Verarbeitung: {str(e)}"}

            # Kurze Pause zwischen Batches, um Rate-Limits zu respektieren
            time.sleep(1)

        return results

    @staticmethod
    @cached()
    def search_stocks(query):
        """Sucht nach Aktien basierend auf Symbol oder Name mit Caching"""
        try:
            # Rate-Limiting anwenden
            StockDataService.rate_limit()

            # Twelvedata Client holen
            td = StockDataService.get_client()

            # Nach dem Symbol bei Twelvedata suchen
            search_results = td.symbol_search(symbol=query).as_json()

            if not search_results or 'data' not in search_results:
                return []

            # Ergebnisse formatieren
            results = []
            for item in search_results['data']:
                results.append({
                    'symbol': item.get('symbol', ''),
                    'name': item.get('instrument_name', ''),
                    'exchange': item.get('exchange', ''),
                    'country': item.get('country', ''),
                    'type': item.get('type', '')
                })

            return results

        except Exception as e:
            print(f"Fehler bei der Aktiensuche: {str(e)}")
            return []
