# stock_analyzer/management/commands/run_ml_predictions.py
from django.core.management.base import BaseCommand
from stock_analyzer.models import Stock, WatchList
from stock_analyzer.ml_models import MLPredictor, batch_ml_predictions
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Führt Machine Learning Vorhersagen für Aktien durch'

    def add_arguments(self, parser):
        parser.add_argument(
            '--watchlist',
            action='store_true',
            help='Nur Aktien in Watchlists vorhersagen',
        )
        parser.add_argument(
            '--retrain',
            action='store_true',
            help='Modelle neu trainieren',
        )
        parser.add_argument(
            '--symbol',
            type=str,
            help='Vorhersage nur für ein bestimmtes Symbol durchführen',
        )

    def handle(self, *args, **options):
        only_watchlist = options['watchlist']
        force_retrain = options['retrain']
        symbol = options.get('symbol')

        if symbol:
            # Vorhersage für ein einzelnes Symbol
            self.stdout.write(f'Führe ML-Vorhersage für {symbol} durch...')
            try:
                predictor = MLPredictor(symbol)

                if force_retrain:
                    self.stdout.write(f'Trainiere Modelle für {symbol} neu...')
                    predictor._train_model('price')
                    predictor._train_model('signal')

                prediction = predictor.predict()

                if prediction:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✓ {symbol}: Vorhersage {prediction["predicted_return"]:.2%}, '
                            f'Empfehlung: {prediction["recommendation"]}, '
                            f'Konfidenz: {prediction["confidence"]:.2f}'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f'✗ {symbol}: Vorhersage fehlgeschlagen')
                    )
            except Exception as e:
                import traceback
                self.stdout.write(
                    self.style.ERROR(f'✗ {symbol}: Fehler bei der Vorhersage: {str(e)}')
                )
                self.stdout.write(self.style.ERROR(f'Traceback:\n{traceback.format_exc()}'))
                logger.error(f'Fehler bei der ML-Vorhersage für {symbol}: {str(e)}')
        else:
            # Bestimme die zu verarbeitenden Symbole
            if only_watchlist:
                # Aktien in Watchlists ermitteln
                watchlist_stocks = WatchList.objects.values_list('stocks__symbol', flat=True).distinct()
                symbols = list(filter(None, watchlist_stocks))  # Filtere None-Werte
                self.stdout.write(f'Führe ML-Vorhersagen für {len(symbols)} Aktien aus Watchlists durch...')
            else:
                # Alle Aktien mit ausreichend Daten
                from django.db.models import Count
                from stock_analyzer.models import StockData

                # Finde Aktien mit mindestens 200 Tagen an Daten
                stocks_with_data = StockData.objects.values('stock') \
                    .annotate(data_count=Count('id')) \
                    .filter(data_count__gte=200)

                stock_ids = [item['stock'] for item in stocks_with_data]
                symbols = list(Stock.objects.filter(id__in=stock_ids).values_list('symbol', flat=True))
                self.stdout.write(f'Führe ML-Vorhersagen für {len(symbols)} Aktien mit ausreichend Daten durch...')

            # Batch-Verarbeitung durchführen
            results = batch_ml_predictions(symbols, force_retrain)

            success_count = 0
            error_count = 0

            for symbol, result in results.items():
                if result['status'] == 'success':
                    prediction = result['prediction']
                    self.stdout.write(
                        f'✓ {symbol}: Vorhersage {prediction["predicted_return"]:.2%}, '
                        f'Empfehlung: {prediction["recommendation"]}, '
                        f'Konfidenz: {prediction["confidence"]:.2f}'
                    )
                    success_count += 1
                else:
                    self.stdout.write(
                        self.style.ERROR(f'✗ {symbol}: {result.get("message", "Unbekannter Fehler")}')
                    )
                    error_count += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f'ML-Vorhersagen abgeschlossen: {success_count} erfolgreich, {error_count} fehlgeschlagen'
                )
            )
