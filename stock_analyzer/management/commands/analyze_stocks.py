# stock_analyzer/management/commands/analyze_stocks.py
from django.core.management.base import BaseCommand
from stock_analyzer.models import Stock, WatchList
from stock_analyzer.data_service import StockDataService
from stock_analyzer.analysis import TechnicalAnalyzer
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Aktualisiert und analysiert alle Aktien'

    def add_arguments(self, parser):
        parser.add_argument(
            '--watchlist',
            action='store_true',
            help='Nur Aktien in Watchlists aktualisieren',
        )

    def handle(self, *args, **options):
        only_watchlist = options['watchlist']

        if only_watchlist:
            # Aktien in Watchlists ermitteln
            watchlist_stocks = WatchList.objects.values_list('stocks__symbol', flat=True).distinct()
            stocks = Stock.objects.filter(symbol__in=watchlist_stocks)
            self.stdout.write(f'Analysiere {stocks.count()} Aktien aus Watchlists...')
        else:
            # Alle Aktien
            stocks = Stock.objects.all()
            self.stdout.write(f'Analysiere alle {stocks.count()} Aktien...')

        success_count = 0
        error_count = 0

        for stock in stocks:
            try:
                # Daten aktualisieren
                success, message = StockDataService.update_stock_data(stock.symbol)

                if success:
                    # Analysieren
                    analyzer = TechnicalAnalyzer(stock.symbol)
                    result = analyzer.save_analysis_result()

                    self.stdout.write(
                        f'✓ {stock.symbol}: Score {result.technical_score}, Empfehlung: {result.recommendation}')
                    success_count += 1
                else:
                    self.stdout.write(self.style.ERROR(f'✗ {stock.symbol}: {message}'))
                    error_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'✗ {stock.symbol}: {str(e)}'))
                logger.error(f'Fehler bei der Analyse von {stock.symbol}: {str(e)}')
                error_count += 1

        self.stdout.write(
            self.style.SUCCESS(f'Analyse abgeschlossen: {success_count} erfolgreich, {error_count} fehlgeschlagen'))