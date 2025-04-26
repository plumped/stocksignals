# stock_analyzer/models.py
from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')

    # Benachrichtigungseinstellungen
    notify_on_buy_signals = models.BooleanField(default=True)
    notify_on_sell_signals = models.BooleanField(default=True)
    min_score_for_buy_notification = models.IntegerField(default=75)
    max_score_for_sell_notification = models.IntegerField(default=25)

    # Analyse-Präferenzen
    preferred_indicators = models.JSONField(default=dict)
    custom_weights = models.JSONField(default=dict)

    # Risikoprofil (1-5, wobei 1 konservativ und 5 aggressiv ist)
    risk_profile = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)], default=3)

    def __str__(self):
        return f"Profil von {self.user.username}"

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    sector = models.CharField(max_length=100, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol} - {self.name}"


class StockData(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='historical_data')
    date = models.DateField()
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    adjusted_close = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('stock', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.symbol} - {self.date}"


class AnalysisResult(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='analysis_results')
    date = models.DateField()
    technical_score = models.DecimalField(max_digits=5, decimal_places=2)
    recommendation = models.CharField(max_length=10)  # BUY, SELL, HOLD

    # Indikatorwerte speichern
    rsi_value = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    macd_value = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    macd_signal = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    sma_20 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    sma_50 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    sma_200 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    bollinger_upper = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    bollinger_lower = models.DecimalField(max_digits=10, decimal_places=2, null=True)

    class Meta:
        unique_together = ('stock', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.symbol} - {self.date} - {self.recommendation}"


class WatchList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlists')
    name = models.CharField(max_length=100)
    stocks = models.ManyToManyField(Stock, related_name='watchlists')

    def __str__(self):
        return f"{self.name} ({self.user.username})"