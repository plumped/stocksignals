# stock_analyzer/backtesting.py
import pandas as pd
from datetime import datetime, timedelta
from .models import Stock, StockData
from .analysis import TechnicalAnalyzer


class BacktestStrategy:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.stock = Stock.objects.get(symbol=symbol)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.shares = 0
        self.trades = []

    def run_backtest(self):
        """Führt den Backtest der Handelsstrategie durch"""
        # Historische Daten abrufen und in ein DataFrame umwandeln
        data = StockData.objects.filter(
            stock=self.stock,
            date__gte=self.start_date,
            date__lte=self.end_date
        ).order_by('date')

        if not data.exists():
            return {
                'success': False,
                'message': 'Keine Daten für den angegebenen Zeitraum gefunden.'
            }

        # DataFrame erstellen
        df = pd.DataFrame(list(data.values()))

        # Technische Indikatoren berechnen
        analyzer = TechnicalAnalyzer(self.stock.symbol)
        df_with_indicators = analyzer._calculate_indicators_for_dataframe(df)

        # Loop durch die Daten und Handelssignale simulieren
        position_open = False
        entry_price = 0
        entry_date = None

        for index, row in df_with_indicators.iterrows():
            if pd.isna(row['rsi']):  # Überspringen, wenn keine vollständigen Indikatoren
                continue

            # Signale nach der Strategie berechnen
            buy_signal = (
                    row['rsi'] < 30 and
                    row['macd'] > row['macd_signal'] and
                    row['close_price'] > row['sma_20']
            )

            sell_signal = (
                    row['rsi'] > 70 or
                    row['macd'] < row['macd_signal'] or
                    row['close_price'] < row['sma_50']
            )

            # Handel simulieren
            if not position_open and buy_signal:
                # Kaufen
                entry_price = row['close_price']
                entry_date = row['date']
                self.shares = self.current_capital / entry_price
                position_open = True

                self.trades.append({
                    'type': 'BUY',
                    'date': entry_date,
                    'price': entry_price,
                    'shares': self.shares,
                    'value': self.shares * entry_price
                })

            elif position_open and sell_signal:
                # Verkaufen
                exit_price = row['close_price']
                exit_date = row['date']
                value = self.shares * exit_price
                profit_loss = value - (self.shares * entry_price)
                profit_loss_pct = (exit_price - entry_price) / entry_price * 100

                self.current_capital = value
                self.trades.append({
                    'type': 'SELL',
                    'date': exit_date,
                    'price': exit_price,
                    'shares': self.shares,
                    'value': value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct
                })

                self.shares = 0
                position_open = False

        # Wenn am Ende der Periode noch eine Position offen ist, zum letzten Preis schließen
        if position_open:
            last_price = df_with_indicators['close_price'].iloc[-1]
            last_date = df_with_indicators['date'].iloc[-1]
            value = self.shares * last_price
            profit_loss = value - (self.shares * entry_price)
            profit_loss_pct = (last_price - entry_price) / entry_price * 100

            self.current_capital = value
            self.trades.append({
                'type': 'SELL (END)',
                'date': last_date,
                'price': last_price,
                'shares': self.shares,
                'value': value,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })

            self.shares = 0

        # Performance-Metriken berechnen
        total_return = self.current_capital - self.initial_capital
        percent_return = (total_return / self.initial_capital) * 100

        # Buy-and-Hold-Vergleich
        first_price = df['close_price'].iloc[0]
        last_price = df['close_price'].iloc[-1]
        buy_hold_return = ((last_price - first_price) / first_price) * 100

        # Gewinne und Verluste analysieren
        winning_trades = [t for t in self.trades if t['type'].startswith('SELL') and t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t['type'].startswith('SELL') and t.get('profit_loss', 0) <= 0]

        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100 if (len(winning_trades) + len(
            losing_trades)) > 0 else 0

        # Ergebnisse zurückgeben
        return {
            'success': True,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'percent_return': percent_return,
            'buy_hold_return': buy_hold_return,
            'trades': self.trades,
            'num_trades': len(self.trades) // 2,  # Jeder vollständige Handel hat einen Kauf und Verkauf
            'win_rate': win_rate,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }