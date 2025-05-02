# stock_analyzer/backtesting.py
import pandas as pd
from datetime import datetime, timedelta
from .models import Stock, StockData
from .analysis import _calculate_indicators_for_dataframe
from .ml_models import MLPredictor


class BacktestStrategy:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000,
                 use_ml=False, slippage_pct=0.001, transaction_cost_pct=0.001,
                 retrain_frequency=30):
        self.stock = Stock.objects.get(symbol=symbol)
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.shares = 0
        self.trades = []
        self.use_ml = use_ml
        self.slippage_pct = slippage_pct
        self.transaction_cost_pct = transaction_cost_pct
        self.retrain_frequency = retrain_frequency
        self.predictor = MLPredictor(symbol) if use_ml else None

    def run_backtest(self):
        data = StockData.objects.filter(
            stock=self.stock,
            date__gte=self.start_date,
            date__lte=self.end_date
        ).order_by('date')

        if not data.exists():
            return {'success': False, 'message': 'Keine Daten gefunden.'}

        df = pd.DataFrame(list(data.values()))
        df_with_indicators = _calculate_indicators_for_dataframe(df)
        position_open = False
        entry_price = 0
        entry_date = None
        last_train_date = self.start_date

        for index, row in df_with_indicators.iterrows():
            if pd.isna(row['rsi']):
                continue

            current_date = row['date']
            close_price = row['close_price']

            if self.use_ml:
                if (current_date - last_train_date).days >= self.retrain_frequency:
                    self.predictor._train_model('price')
                    self.predictor._train_model('signal')
                    last_train_date = current_date

                prediction = self.predictor.predict()
                if prediction is None:
                    continue

                buy_signal = prediction['recommendation'] == 'BUY' and prediction['confidence'] >= 0.6
                sell_signal = prediction['recommendation'] == 'SELL' and prediction['confidence'] >= 0.6

            else:
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

            if not position_open and buy_signal:
                effective_price = close_price * (1 + self.slippage_pct)
                cost = effective_price * (1 + self.transaction_cost_pct)
                self.shares = self.current_capital / cost
                position_open = True
                entry_price = effective_price
                entry_date = current_date

                self.trades.append({
                    'type': 'BUY',
                    'date': entry_date,
                    'price': effective_price,
                    'shares': self.shares,
                    'value': self.shares * effective_price
                })

            elif position_open and sell_signal:
                effective_price = close_price * (1 - self.slippage_pct)
                proceeds = self.shares * effective_price * (1 - self.transaction_cost_pct)
                profit_loss = proceeds - (self.shares * entry_price)
                profit_loss_pct = (effective_price - entry_price) / entry_price * 100

                self.current_capital = proceeds
                self.trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': effective_price,
                    'shares': self.shares,
                    'value': proceeds,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct
                })
                self.shares = 0
                position_open = False

        if position_open:
            last_price = df_with_indicators['close_price'].iloc[-1]
            last_date = df_with_indicators['date'].iloc[-1]
            effective_price = last_price * (1 - self.slippage_pct)
            proceeds = self.shares * effective_price * (1 - self.transaction_cost_pct)
            profit_loss = proceeds - (self.shares * entry_price)
            profit_loss_pct = (effective_price - entry_price) / entry_price * 100

            self.current_capital = proceeds
            self.trades.append({
                'type': 'SELL (END)',
                'date': last_date,
                'price': effective_price,
                'shares': self.shares,
                'value': proceeds,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })
            self.shares = 0

        total_return = self.current_capital - self.initial_capital
        percent_return = (total_return / self.initial_capital) * 100
        first_price = df['close_price'].iloc[0]
        last_price = df['close_price'].iloc[-1]
        buy_hold_return = ((last_price - first_price) / first_price) * 100

        winning_trades = [t for t in self.trades if t['type'].startswith('SELL') and t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t['type'].startswith('SELL') and t.get('profit_loss', 0) <= 0]

        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100 if (len(winning_trades) + len(
            losing_trades)) > 0 else 0

        return {
            'success': True,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'percent_return': percent_return,
            'buy_hold_return': buy_hold_return,
            'trades': self.trades,
            'num_trades': len(self.trades) // 2,
            'win_rate': win_rate,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
