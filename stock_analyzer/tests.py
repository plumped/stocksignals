from django.test import TestCase
from .models import Stock, StockData, AnalysisResult, WatchList
from django.contrib.auth.models import User
from decimal import Decimal
from datetime import date


class StockModelTest(TestCase):
    """Tests für das Stock-Model und zugehörige Funktionalitäten"""

    def setUp(self):
        """Test-Setup mit Beispiel-Daten"""
        # Benutzer erstellen
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )

        # Aktie erstellen
        self.stock = Stock.objects.create(
            symbol='AAPL',
            name='Apple Inc.',
            sector='Technology'
        )

        # Historische Daten erstellen
        for i in range(10):
            day = date(2025, 1, i + 1)
            price = 150 + i * 2  # Steigender Preis

            StockData.objects.create(
                stock=self.stock,
                date=day,
                open_price=Decimal(price - 1),
                high_price=Decimal(price + 2),
                low_price=Decimal(price - 2),
                close_price=Decimal(price),
                adjusted_close=Decimal(price),
                volume=1000000 + i * 100000
            )

        # Watchlist erstellen
        self.watchlist = WatchList.objects.create(
            user=self.user,
            name='Test Watchlist'
        )
        self.watchlist.stocks.add(self.stock)

        # Analyse-Ergebnis erstellen
        self.analysis = AnalysisResult.objects.create(
            stock=self.stock,
            date=date(2025, 1, 10),
            technical_score=Decimal('75.50'),
            recommendation='BUY',
            rsi_value=Decimal('30.00'),
            macd_value=Decimal('2.50'),
            macd_signal=Decimal('1.20'),
            sma_20=Decimal('155.00'),
            sma_50=Decimal('150.00'),
            sma_200=Decimal('145.00'),
            bollinger_upper=Decimal('165.00'),
            bollinger_lower=Decimal('145.00')
        )

    def test_stock_creation(self):
        """Test: Aktie wurde korrekt angelegt"""
        self.assertEqual(self.stock.symbol, 'AAPL')
        self.assertEqual(self.stock.name, 'Apple Inc.')
        self.assertEqual(self.stock.sector, 'Technology')

    def test_stock_data_creation(self):
        """Test: Historische Daten wurden korrekt angelegt"""
        data_count = StockData.objects.filter(stock=self.stock).count()
        self.assertEqual(data_count, 10)

        # Letzten Schlusskurs prüfen
        last_data = StockData.objects.filter(stock=self.stock).order_by('-date').first()
        self.assertEqual(last_data.close_price, Decimal('168'))

    def test_watchlist_functionality(self):
        """Test: Watchlist funktioniert korrekt"""
        # Test: Aktie ist in Watchlist
        self.assertTrue(self.watchlist.stocks.filter(symbol='AAPL').exists())

        # Test: Watchlist gehört zum Benutzer
        user_watchlists = WatchList.objects.filter(user=self.user)
        self.assertEqual(user_watchlists.count(), 1)
        self.assertEqual(user_watchlists.first().name, 'Test Watchlist')

        # Aktie aus Watchlist entfernen
        self.watchlist.stocks.remove(self.stock)
        self.assertFalse(self.watchlist.stocks.filter(symbol='AAPL').exists())

    def test_analysis_result(self):
        """Test: Analyseergebnis wurde korrekt gespeichert"""
        # Test: Werte stimmen
        self.assertEqual(self.analysis.technical_score, Decimal('75.50'))
        self.assertEqual(self.analysis.recommendation, 'BUY')

        # Test: Zuordnung zur Aktie funktioniert
        stock_analysis = AnalysisResult.objects.filter(stock=self.stock).first()
        self.assertEqual(stock_analysis.id, self.analysis.id)