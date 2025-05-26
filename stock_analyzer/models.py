# stock_analyzer/models.py
from django.db import models
from django.contrib.auth.models import User
from decimal import Decimal


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')

    # Benachrichtigungseinstellungen
    notify_on_buy_signals = models.BooleanField(default=True)
    notify_on_sell_signals = models.BooleanField(default=True)
    min_score_for_buy_notification = models.IntegerField(default=75)
    max_score_for_sell_notification = models.IntegerField(default=25)

    # Analyse-Präferenzen
    preferred_indicators = models.JSONField(default=dict)

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
    confluence_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('stock', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.symbol} - {self.date} - {self.recommendation}"


class MLPrediction(models.Model):
    """Machine Learning predictions for stocks"""
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='ml_predictions')
    date = models.DateField()
    predicted_return = models.FloatField(help_text="Vorhergesagte prozentuale Veränderung")
    predicted_price = models.FloatField(help_text="Vorhergesagter Preis")
    recommendation = models.CharField(max_length=10, help_text="BUY, SELL, HOLD")
    confidence = models.FloatField(default=0.0, help_text="Konfidenz der ML-Vorhersage (0-1)")
    prediction_days = models.IntegerField(default=5, help_text="Anzahl der Tage für die Vorhersage")

    class Meta:
        unique_together = ('stock', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.symbol} - {self.date} - {self.recommendation} ({self.confidence:.2f})"


class MLModelMetrics(models.Model):
    """Speichert Metriken für ML-Modelle"""
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='ml_metrics')
    date = models.DateField(auto_now_add=True)
    accuracy = models.FloatField(help_text="Genauigkeit des Klassifikationsmodells (0-1)")
    rmse = models.FloatField(help_text="RMSE des Regressionsmodells")
    feature_importance = models.JSONField(null=True, blank=True, help_text="Feature Importance als JSON")
    confusion_matrix = models.JSONField(null=True, blank=True, help_text="Confusion Matrix als JSON")
    directional_accuracy = models.FloatField(null=True, blank=True,
                                             help_text="Genauigkeit der Richtungsvorhersage (0-1)")
    market_regimes = models.JSONField(null=True, blank=True, help_text="Marktregime-Informationen als JSON")
    model_version = models.CharField(max_length=50, default="v1", help_text="Version des Modells")
    sentiment_data = models.JSONField(null=True, blank=True, help_text="Sentiment-Analyse-Daten als JSON")

    class Meta:
        unique_together = ('stock', 'date', 'model_version')
        ordering = ['-date']

    def __str__(self):
        return f"{self.stock.symbol} ML-Metrik ({self.date})"


class WatchList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlists')
    name = models.CharField(max_length=100)
    stocks = models.ManyToManyField(Stock, related_name='watchlists')

    def __str__(self):
        return f"{self.name} ({self.user.username})"


# stock_analyzer/models.py - Add these new models to your existing models.py file

class Portfolio(models.Model):
    """Model for user portfolios"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Portfolio statistics (cached values for performance)
    total_value = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_cost = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_gain_loss = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    percent_gain_loss = models.DecimalField(max_digits=6, decimal_places=2, default=0)

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    def update_statistics(self):
        """Update portfolio statistics based on current positions and market prices"""
        from decimal import Decimal

        # Get all active positions
        positions = self.positions.all()

        total_value = Decimal('0')
        total_cost = Decimal('0')

        for position in positions:
            position.update_values()
            total_value += position.current_value
            total_cost += position.cost_basis

        self.total_value = total_value
        self.total_cost = total_cost

        if total_cost > 0:
            self.total_gain_loss = total_value - total_cost
            self.percent_gain_loss = (self.total_gain_loss / total_cost) * 100
        else:
            self.total_gain_loss = 0
            self.percent_gain_loss = 0

        self.save()
        return self.total_value


class Position(models.Model):
    """Model for positions within a portfolio (current holdings)"""
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='positions')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='positions')
    shares = models.DecimalField(max_digits=10, decimal_places=4, default=0)

    # Cached values for performance
    cost_basis = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    average_price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    current_value = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    gain_loss = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    percent_gain_loss = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('portfolio', 'stock')

    def __str__(self):
        return f"{self.stock.symbol} - {self.shares} shares"

    def update_values(self):
        """Update the position's values based on the latest market price"""
        from decimal import Decimal

        # Get latest price
        latest_price_obj = StockData.objects.filter(stock=self.stock).order_by('-date').first()

        if latest_price_obj:
            latest_price = Decimal(str(latest_price_obj.close_price))

            # Calculate current value
            self.current_value = self.shares * latest_price

            # Calculate gain/loss
            if self.cost_basis > 0:
                self.gain_loss = self.current_value - self.cost_basis
                self.percent_gain_loss = (self.gain_loss / self.cost_basis) * 100
            else:
                self.gain_loss = 0
                self.percent_gain_loss = 0

            self.save()

        return self.current_value

    @property
    def raw_cost_basis(self):
        from .models import Trade
        trades = Trade.objects.filter(portfolio=self.portfolio, stock=self.stock, trade_type='BUY')
        return sum(t.price * t.shares for t in trades)

    @property
    def gross_gain(self):
        return self.current_value - self.raw_cost_basis

    @property
    def gross_return(self):
        if self.raw_cost_basis > 0:
            return (self.gross_gain / self.raw_cost_basis) * 100
        return 0

    @property
    def total_fees(self):
        from .models import Trade
        trades = Trade.objects.filter(portfolio=self.portfolio, stock=self.stock, trade_type='BUY')
        return sum(t.fees for t in trades)



class Trade(models.Model):
    """Model for individual trades within a portfolio"""
    TRADE_TYPES = (
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
        ('DIVIDEND', 'Dividend'),
        ('SPLIT', 'Stock Split'),
        ('TRANSFER_IN', 'Transfer In'),
        ('TRANSFER_OUT', 'Transfer Out'),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='trades')
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='trades')
    trade_type = models.CharField(max_length=15, choices=TRADE_TYPES)

    date = models.DateField()
    shares = models.DecimalField(max_digits=10, decimal_places=4)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    fees = models.DecimalField(max_digits=8, decimal_places=2, default=0)

    # Calculated values
    total_value = models.DecimalField(max_digits=12, decimal_places=2)
    notes = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.trade_type} {self.shares} {self.stock.symbol} @ {self.price}"

    def save(self, *args, **kwargs):
        is_update = self.pk is not None

        if is_update:
            original = Trade.objects.get(pk=self.pk)

            changed = (
                    original.shares != self.shares or
                    original.price != self.price or
                    original.fees != self.fees or
                    original.trade_type != self.trade_type or
                    original.stock != self.stock or
                    original.portfolio != self.portfolio
            )

            if changed:
                self._reverse_position_effect(original)

        # Korrekt: Wert **immer** neu berechnen
        self.total_value = (self.shares * self.price) + self.fees

        # Speichern
        super().save(*args, **kwargs)

        # Position **nicht einfach draufaddieren** – besser: alles neu berechnen
        self._recalculate_position()

        # Portfolio aktualisieren
        self.portfolio.update_statistics()

    def _recalculate_position(self):
        """Berechnet Position neu aus allen zugehörigen Trades"""
        from decimal import Decimal

        trades = Trade.objects.filter(
            portfolio=self.portfolio,
            stock=self.stock
        ).order_by('date')

        position, _ = Position.objects.get_or_create(
            portfolio=self.portfolio,
            stock=self.stock,
            defaults={'shares': 0, 'cost_basis': 0, 'average_price': 0}
        )

        shares = Decimal('0')
        cost = Decimal('0')

        for t in trades:
            if t.trade_type == 'BUY' or t.trade_type == 'TRANSFER_IN':
                shares += t.shares
                cost += t.total_value
            elif t.trade_type == 'SELL' or t.trade_type == 'TRANSFER_OUT':
                if shares > 0:
                    cost_reduction = (t.shares / shares) * cost
                    cost -= cost_reduction
                shares -= t.shares
            elif t.trade_type == 'SPLIT':
                shares *= t.price  # split ratio

        if shares > 0:
            avg_price = cost / shares
        else:
            avg_price = Decimal('0')
            cost = Decimal('0')

        position.shares = shares
        position.cost_basis = cost
        position.average_price = avg_price

        position.update_values()
        position.save()

    def _reverse_position_effect(self, old_trade):
        position = Position.objects.filter(
            portfolio=old_trade.portfolio,
            stock=old_trade.stock
        ).first()

        if not position:
            return

        if old_trade.trade_type == 'BUY':
            position.shares -= old_trade.shares
            position.cost_basis -= old_trade.total_value

        elif old_trade.trade_type == 'SELL':
            position.shares += old_trade.shares
            position.cost_basis += old_trade.price * old_trade.shares

        # Korrektur: Average Price aktualisieren
        if position.shares > 0:
            position.average_price = position.cost_basis / position.shares
        else:
            position.average_price = 0
            position.cost_basis = 0

        position.update_values()
        position.save()

    def _update_position(self):
        """Update the associated position after a trade"""
        from decimal import Decimal

        position, created = Position.objects.get_or_create(
            portfolio=self.portfolio,
            stock=self.stock,
            defaults={'shares': 0, 'cost_basis': 0, 'average_price': 0}
        )

        # Handle different trade types
        if self.trade_type == 'BUY':
            # Calculate new cost basis and average price
            new_shares = position.shares + self.shares
            new_cost = position.cost_basis + self.total_value

            if new_shares > 0:
                new_avg_price = new_cost / new_shares
            else:
                new_avg_price = 0

            position.shares = new_shares
            position.cost_basis = new_cost
            position.average_price = new_avg_price

        elif self.trade_type == 'SELL':
            # Handle selling shares
            if position.shares >= self.shares:
                # Calculate the portion of cost basis being sold
                if position.shares > 0:
                    cost_reduction = (self.shares / position.shares) * position.cost_basis
                else:
                    cost_reduction = 0

                new_shares = position.shares - self.shares
                new_cost = position.cost_basis - cost_reduction

                if new_shares > 0:
                    new_avg_price = new_cost / new_shares
                else:
                    new_avg_price = 0

                position.shares = new_shares
                position.cost_basis = new_cost
                position.average_price = new_avg_price

        elif self.trade_type == 'DIVIDEND':
            # Dividends don't change the position directly
            pass




        elif self.trade_type == 'SPLIT':
            if self.price > 0:
                split_ratio = self.price
                # Berechne Netto-Anzahl vor dem Split
                trades_before_split = Trade.objects.filter(
                    portfolio=self.portfolio,
                    stock=self.stock,
                    date__lt=self.date

                ).order_by('date')
                net_shares = Decimal('0')
                for trade in trades_before_split:
                    if trade.trade_type == 'BUY':
                        net_shares += trade.shares
                    elif trade.trade_type == 'SELL':
                        net_shares -= trade.shares

                # Berechne unberührte Anteile (nach dem Split)
                trades_after_split = Trade.objects.filter(
                    portfolio=self.portfolio,
                    stock=self.stock,
                    date__gt=self.date
                )

                post_split_shares = Decimal('0')
                for trade in trades_after_split:
                    if trade.trade_type == 'BUY':
                        post_split_shares += trade.shares
                    elif trade.trade_type == 'SELL':
                        post_split_shares -= trade.shares

                if net_shares > 0:
                    new_split_shares = net_shares * split_ratio
                    position.shares = new_split_shares + post_split_shares
                    if position.shares > 0:
                        position.average_price = position.cost_basis / position.shares
                    else:
                        position.average_price = 0




        elif self.trade_type == 'TRANSFER_IN':
            # Similar to buy but may have different accounting
            new_shares = position.shares + self.shares
            new_cost = position.cost_basis + self.total_value

            if new_shares > 0:
                new_avg_price = new_cost / new_shares
            else:
                new_avg_price = 0

            position.shares = new_shares
            position.cost_basis = new_cost
            position.average_price = new_avg_price

        elif self.trade_type == 'TRANSFER_OUT':
            # Similar to sell
            if position.shares >= self.shares:
                if position.shares > 0:
                    cost_reduction = (self.shares / position.shares) * position.cost_basis
                else:
                    cost_reduction = 0

                new_shares = position.shares - self.shares
                new_cost = position.cost_basis - cost_reduction

                if new_shares > 0:
                    new_avg_price = new_cost / new_shares
                else:
                    new_avg_price = 0

                position.shares = new_shares
                position.cost_basis = new_cost
                position.average_price = new_avg_price

        # Save the updated position
        position.update_values()
        position.save()
