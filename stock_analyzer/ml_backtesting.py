# stock_analyzer/ml_backtesting.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, \
    mean_absolute_error
import logging
from decimal import Decimal

from .models import Stock, StockData, MLPrediction, MLModelMetrics
from .ml_models import MLPredictor

logger = logging.getLogger(__name__)


class MLBacktester:
    """
    Backtesting system for ML models that simulates trading decisions
    based on ML predictions over historical data.
    """

    def __init__(self, symbol, start_date=None, end_date=None, initial_capital=10000,
                 prediction_days=5, confidence_threshold=0.65,
                 stop_loss_pct=0.05, take_profit_pct=0.10):
        """
        Initialize the ML backtester.

        Args:
            symbol (str): Stock symbol to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital for trading
            prediction_days (int): Number of days to look ahead for predictions
            confidence_threshold (float): Minimum confidence to execute trades
            stop_loss_pct (float): Stop loss percentage (e.g., 0.05 for 5%)
            take_profit_pct (float): Take profit percentage (e.g., 0.10 for 10%)
        """
        self.symbol = symbol
        self.initial_capital = float(initial_capital)
        self.current_capital = self.initial_capital
        self.prediction_days = prediction_days
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Get stock object
        self.stock = Stock.objects.get(symbol=symbol)

        # Set date range
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            # Default to 1 year before end_date
            start_date = end_date - timedelta(days=365)

        self.start_date = start_date
        self.end_date = end_date

        # Initialize tracking variables
        self.positions = 0  # Number of shares held
        self.trades = []
        self.daily_portfolio_values = []
        self.entry_price = 0
        self.last_position_type = None  # 'long', 'short', or None

        # Trading signals and performance metrics
        self.signals = []
        self.metrics = {}

        # Validation flags
        self._data_validated = False
        self._has_sufficient_data = False

    def validate_data(self):
        """
        Validate that there's sufficient data for backtesting.
        Returns True if validation passes, False otherwise.
        """
        # Check if we have enough price data
        price_data = StockData.objects.filter(
            stock=self.stock,
            date__range=[self.start_date, self.end_date]
        ).order_by('date')

        if price_data.count() < 30:  # Minimum data requirement
            logger.warning(f"Insufficient price data for {self.symbol}. Need at least 30 data points.")
            self._has_sufficient_data = False
            return False

        # Check if the stock has ML model data
        model_path = f'ml_models/{self.symbol}_price_model.pkl'
        import os
        if not os.path.exists(model_path):
            logger.warning(f"No ML model found for {self.symbol}.")
            self._has_sufficient_data = False
            return False

        # All validations passed
        self._data_validated = True
        self._has_sufficient_data = True
        return True

    def run_backtest(self):
        """
        Execute the backtest over the specified time period.

        Returns:
            dict: Results of the backtest including trades, performance metrics
        """
        if not self._data_validated:
            if not self.validate_data():
                return {
                    "success": False,
                    "message": f"Insufficient data for backtesting {self.symbol}."
                }

        # Get historical price data
        price_data = StockData.objects.filter(
            stock=self.stock,
            date__range=[self.start_date, self.end_date]
        ).order_by('date')

        # Convert to DataFrame
        df = pd.DataFrame(list(price_data.values()))

        # Convert price columns to float
        for col in ['open_price', 'high_price', 'low_price', 'close_price']:
            df[col] = df[col].astype(float)

        # Create a sliding window to simulate real-time predictions
        test_windows = self._create_test_windows(df)

        # For each trading day in our backtest period
        for i, window in enumerate(test_windows):
            if i % 10 == 0:  # Log progress every 10 windows
                logger.info(f"Processing window {i + 1} of {len(test_windows)}")

            # Extract current window data
            train_df = window['train']
            test_day = window['test']

            # Skip if test_day is empty
            if test_day.empty:
                continue

            # Current price and date
            current_date = test_day['date'].iloc[0]
            current_price = float(test_day['close_price'].iloc[0])

            # Generate prediction using only data up to this point
            prediction = self._generate_prediction(train_df, test_day)

            if prediction:
                # Store the prediction signal
                signal = {
                    'date': current_date,
                    'price': current_price,
                    'prediction': prediction['recommendation'],
                    'predicted_return': prediction['predicted_return'],
                    'confidence': prediction['confidence']
                }
                self.signals.append(signal)

                # Execute trading logic based on prediction
                self._execute_trading_strategy(signal)

            # Record daily portfolio value
            self._update_portfolio_value(current_date, current_price)

        # Calculate performance metrics
        self._calculate_performance_metrics(df)

        # Close any open positions at the end of the backtest
        if self.positions != 0:
            last_price = float(df['close_price'].iloc[-1])
            last_date = df['date'].iloc[-1]

            # Close position
            if self.positions > 0:
                self._close_position(last_date, last_price, "END_OF_BACKTEST")
            elif self.positions < 0 and self.last_position_type == 'short':
                self._close_short_position(last_date, last_price, "END_OF_BACKTEST")

        # Format and return results
        return self._format_results()

    def _create_test_windows(self, df):
        """
        Create sliding windows of training/test data to simulate predictions over time.

        Args:
            df (DataFrame): Historical price data

        Returns:
            list: List of dictionaries containing train/test splits
        """
        windows = []
        min_train_size = 200  # Minimum days needed for training

        if len(df) < min_train_size + 1:
            logger.warning(f"Not enough data points for {self.symbol} to create test windows.")
            return windows

        # Start with minimum training size and slide the window forward
        for i in range(min_train_size, len(df)):
            train = df.iloc[:i]
            test = df.iloc[i:i + 1]  # One day at a time

            windows.append({
                'train': train,
                'test': test
            })

        return windows

    def _generate_prediction(self, train_df, test_day):
        """
        Generate an ML prediction using only the training data available up to this point.

        This method properly simulates how predictions would have been made in real-time
        during the historical period, avoiding look-ahead bias by training models only
        on data available at each point in time.

        Args:
            train_df (DataFrame): Training data up to current day
            test_day (DataFrame): Current day's data

        Returns:
            dict: Prediction results or None if prediction fails
        """
        try:
            test_date = test_day['date'].iloc[0]

            # Convert data to the format expected by MLPredictor
            # We need to use only train_df to prepare features
            stock_data = []

            # Format each row as a StockData object-like dictionary
            for _, row in train_df.iterrows():
                stock_data.append({
                    'date': row['date'],
                    'open_price': row['open_price'],
                    'high_price': row['high_price'],
                    'low_price': row['low_price'],
                    'close_price': row['close_price'],
                    'volume': row['volume'],
                    'adjusted_close': row.get('adjusted_close', row['close_price'])
                })

            from .ml_models import MLPredictor
            import tempfile
            import os
            import joblib

            # Create a temporary directory to store model files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set up a custom MLPredictor that will save its models to our temp directory
                class BacktestMLPredictor(MLPredictor):
                    def __init__(self, symbol, historical_data, temp_dir, prediction_days=5):
                        self.stock_symbol = symbol
                        self.prediction_days = prediction_days
                        self.historical_data = historical_data

                        # Override models directory
                        self.models_dir = temp_dir

                        # Set model paths
                        self.price_model_path = os.path.join(self.models_dir, f'{symbol}_price_model.pkl')
                        self.signal_model_path = os.path.join(self.models_dir, f'{symbol}_signal_model.pkl')

                        # Always train fresh models
                        self.price_model = self._train_model('price')
                        self.signal_model = self._train_model('signal')

                    def prepare_data(self):
                        """Override to use our historical data instead of querying the database"""
                        import pandas as pd
                        import numpy as np

                        # Convert list of dicts to DataFrame
                        df = pd.DataFrame(self.historical_data)

                        # Convert Decimal fields to float
                        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                            df[col] = df[col].astype(float)

                        # Calculate features
                        df = self._calculate_features(df)

                        # Make a copy to avoid SettingWithCopyWarning
                        features = df.copy()

                        # Drop any rows with NaN values first
                        features = features.dropna()

                        if len(features) < 30:  # Need at least 30 data points
                            return None, None, None

                        # Target is the next n-day percentage change
                        features.loc[:, 'future_return'] = features['close_price'].pct_change(
                            self.prediction_days).shift(-self.prediction_days)

                        # Create signal target
                        features.loc[:, 'signal_target'] = 0
                        threshold = 0.02  # 2% movement threshold for signal
                        features.loc[features['future_return'] > threshold, 'signal_target'] = 1
                        features.loc[features['future_return'] < -threshold, 'signal_target'] = -1

                        # Remove NaN values
                        features = features.dropna()

                        # Features for models
                        feature_columns = [col for col in features.columns if col not in
                                           ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

                        X = features[feature_columns]
                        y_price = features['future_return']
                        y_signal = features['signal_target']

                        # Scale features
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        X_scaled = scaler.fit_transform(X)

                        return X_scaled, y_price, y_signal

                # Initialize backtest predictor with historical data
                predictor = BacktestMLPredictor(
                    symbol=self.symbol,
                    historical_data=stock_data,
                    temp_dir=temp_dir,
                    prediction_days=self.prediction_days
                )

                # Make prediction
                prediction = predictor.predict()

                return prediction

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating prediction for {self.symbol} on {test_day['date'].iloc[0]}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _execute_trading_strategy(self, signal):
        """
        Execute trading strategy based on the signal.

        Args:
            signal (dict): Trading signal including prediction
        """
        date = signal['date']
        price = signal['price']
        prediction = signal['prediction']
        confidence = signal['confidence']

        # Only execute trades with sufficient confidence
        if confidence < self.confidence_threshold:
            return

        # Check if we should enter a position
        if prediction == "BUY" and self.positions <= 0:  # No position or short position
            # Close any existing short position first
            if self.positions < 0:
                self._close_short_position(date, price, "SIGNAL_REVERSAL")

            # Calculate how many shares we can buy
            shares_to_buy = (self.current_capital * 0.95) // price  # Use 95% of capital

            if shares_to_buy > 0:
                # Enter new long position
                self._enter_long_position(date, price, shares_to_buy)

        # Check if we should exit a long position
        elif prediction == "SELL" and self.positions > 0:
            # Close long position
            self._close_position(date, price, "ML_SELL_SIGNAL")

            # Optional: Enter short position (for advanced strategies)
            # Uncomment to enable shorting:
            # shares_to_short = (self.current_capital * 0.95) // price
            # if shares_to_short > 0:
            #     self._enter_short_position(date, price, shares_to_short)

        # Risk management: Check stop loss/take profit for long positions
        if self.positions > 0:
            position_return = (price - self.entry_price) / self.entry_price

            # Stop loss hit
            if position_return <= -self.stop_loss_pct:
                self._close_position(date, price, "STOP_LOSS")

            # Take profit hit
            elif position_return >= self.take_profit_pct:
                self._close_position(date, price, "TAKE_PROFIT")

        # Risk management for short positions (if enabled)
        elif self.positions < 0 and self.last_position_type == 'short':
            position_return = (self.entry_price - price) / self.entry_price

            # Stop loss hit for short position
            if position_return <= -self.stop_loss_pct:
                self._close_short_position(date, price, "STOP_LOSS")

            # Take profit hit for short position
            elif position_return >= self.take_profit_pct:
                self._close_short_position(date, price, "TAKE_PROFIT")

    def _enter_long_position(self, date, price, shares):
        """Enter a long position"""
        cost = shares * price
        self.current_capital -= cost
        self.positions = shares
        self.entry_price = price
        self.last_position_type = 'long'

        # Record the trade
        self.trades.append({
            'date': date,
            'type': 'BUY',
            'shares': shares,
            'price': price,
            'value': cost,
            'capital_after': self.current_capital,
            'profit_loss': None
        })

    def _close_position(self, date, price, reason):
        """Close a long position"""
        if self.positions <= 0:
            return

        proceeds = self.positions * price
        profit_loss = proceeds - (self.positions * self.entry_price)
        self.current_capital += proceeds

        # Record the trade
        self.trades.append({
            'date': date,
            'type': 'SELL',
            'reason': reason,
            'shares': self.positions,
            'price': price,
            'value': proceeds,
            'capital_after': self.current_capital,
            'profit_loss': profit_loss,
            'profit_loss_pct': (profit_loss / (self.positions * self.entry_price)) * 100
        })

        # Reset position
        self.positions = 0
        self.entry_price = 0
        self.last_position_type = None

    def _enter_short_position(self, date, price, shares):
        """Enter a short position"""
        # In a real backtest, this would involve borrowing shares
        # Here we're simplifying by just tracking the position
        self.positions = -shares  # Negative for short
        self.entry_price = price
        self.last_position_type = 'short'

        # Record the trade
        self.trades.append({
            'date': date,
            'type': 'SHORT',
            'shares': shares,
            'price': price,
            'value': shares * price,
            'capital_after': self.current_capital,
            'profit_loss': None
        })

    def _close_short_position(self, date, price, reason):
        """Close a short position"""
        if self.positions >= 0 or self.last_position_type != 'short':
            return

        shares = -self.positions  # Convert to positive
        cost = shares * price
        profit_loss = (shares * self.entry_price) - cost
        self.current_capital += profit_loss

        # Record the trade
        self.trades.append({
            'date': date,
            'type': 'COVER',
            'reason': reason,
            'shares': shares,
            'price': price,
            'value': cost,
            'capital_after': self.current_capital,
            'profit_loss': profit_loss,
            'profit_loss_pct': (profit_loss / (shares * self.entry_price)) * 100
        })

        # Reset position
        self.positions = 0
        self.entry_price = 0
        self.last_position_type = None

    def _update_portfolio_value(self, date, current_price):
        """Update the portfolio value for a given date"""
        # Calculate total portfolio value (cash + positions)
        position_value = self.positions * current_price if self.positions > 0 else 0
        portfolio_value = self.current_capital + position_value

        # For short positions, subtract the potential cost to cover
        if self.positions < 0:
            position_liability = -self.positions * current_price
            portfolio_value = self.current_capital - position_liability

        # Record the daily value
        self.daily_portfolio_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'position_value': position_value if self.positions > 0 else -position_value if self.positions < 0 else 0,
            'positions': self.positions
        })

    def _calculate_performance_metrics(self, price_df):
        """
        Calculate performance metrics for the backtest

        Args:
            price_df (DataFrame): Historical price data
        """
        if not self.daily_portfolio_values:
            self.metrics = {
                'total_return': 0,
                'total_return_pct': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'avg_profit_per_trade': 0,
                'avg_loss_per_trade': 0
            }
            return

        # Calculate return metrics
        initial_value = self.initial_capital
        final_value = self.daily_portfolio_values[-1]['portfolio_value']
        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100

        # Calculate daily returns
        portfolio_values = [day['portfolio_value'] for day in self.daily_portfolio_values]
        daily_returns = []

        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            daily_returns.append(daily_return)

        # Annualized metrics
        days = (self.end_date - self.start_date).days
        if days > 0:
            annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100
        else:
            annualized_return = 0

        # Risk metrics
        if daily_returns:
            std_dev = np.std(daily_returns) * np.sqrt(252)  # Annualized
            if std_dev > 0:
                sharpe_ratio = (annualized_return / 100) / std_dev
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        running_max = initial_value
        max_drawdown = 0

        for value in portfolio_values:
            if value > running_max:
                running_max = value
            drawdown = (running_max - value) / running_max * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Trading metrics
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) is not None and t.get('profit_loss', 0) <= 0]

        num_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

        # Gross profit and loss
        gross_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        gross_loss = sum(abs(t.get('profit_loss', 0)) for t in losing_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Store the metrics
        self.metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_profit_per_trade': gross_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss_per_trade': -gross_loss / len(losing_trades) if losing_trades else 0,
        }

    def _format_results(self):
        """Format the backtest results"""
        if not self._has_sufficient_data:
            return {
                'success': False,
                'message': f"Insufficient data for backtesting {self.symbol}."
            }

        # Filter trades to show only actual buys and sells
        actual_trades = [t for t in self.trades if t['type'] in ['BUY', 'SELL', 'SHORT', 'COVER']]

        # Format metrics to match the expected output
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.daily_portfolio_values[-1][
                'portfolio_value'] if self.daily_portfolio_values else self.initial_capital,
            'total_return': self.metrics['total_return'],
            'percent_return': self.metrics['total_return_pct'],
            'buy_hold_return': 0,  # This will be calculated after
            'trades': actual_trades,
            'num_trades': self.metrics['num_trades'],
            'win_rate': self.metrics['win_rate'],
            'winning_trades': self.metrics['winning_trades'],
            'losing_trades': self.metrics['losing_trades'],
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'max_drawdown': self.metrics['max_drawdown'],
            'profit_factor': self.metrics['profit_factor']
        }

        # Calculate buy and hold return for comparison
        if len(self.daily_portfolio_values) >= 2:
            first_price = float(StockData.objects.filter(
                stock=self.stock,
                date__gte=self.start_date
            ).order_by('date').first().close_price)

            last_price = float(StockData.objects.filter(
                stock=self.stock,
                date__lte=self.end_date
            ).order_by('-date').first().close_price)

            buy_hold_return = (last_price - first_price) / first_price * 100
            metrics['buy_hold_return'] = buy_hold_return

        # Return complete results
        return {
            'success': True,
            'metrics': metrics,
            'trades': actual_trades,
            'daily_portfolio_values': self.daily_portfolio_values,
            'signals': self.signals
        }

    def generate_performance_charts(self):
        """
        Generate visual charts showing backtest performance.

        Returns:
            dict: Dictionary of base64-encoded charts
        """
        if not self.daily_portfolio_values:
            return {}

        charts = {}

        # Portfolio value chart
        plt.figure(figsize=(10, 6))
        dates = [day['date'] for day in self.daily_portfolio_values]
        values = [day['portfolio_value'] for day in self.daily_portfolio_values]

        plt.plot(dates, values, label='Portfolio Value')

        # Calculate buy and hold line for comparison
        if len(self.daily_portfolio_values) >= 2:
            start_price = float(StockData.objects.filter(
                stock=self.stock,
                date__gte=self.start_date
            ).order_by('date').first().close_price)

            # Calculate how many shares could be bought with initial capital
            shares_bought = self.initial_capital / start_price

            # Calculate buy and hold value for each day
            buy_hold_values = []
            for day in self.daily_portfolio_values:
                date = day['date']
                price = float(StockData.objects.filter(
                    stock=self.stock,
                    date=date
                ).first().close_price)
                buy_hold_value = shares_bought * price
                buy_hold_values.append(buy_hold_value)

            plt.plot(dates, buy_hold_values, label='Buy & Hold')

        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['portfolio_value'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Drawdown chart
        plt.figure(figsize=(10, 6))

        # Calculate running maximum and drawdown
        running_max = self.initial_capital
        drawdowns = []

        for value in values:
            if value > running_max:
                running_max = value
            drawdown = (running_max - value) / running_max * 100
            drawdowns.append(drawdown)

        plt.plot(dates, drawdowns)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['drawdown'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Trade distribution chart
        if self.trades:
            plt.figure(figsize=(10, 6))

            # Extract profit/loss percentages from trades
            trade_returns = [t.get('profit_loss_pct', 0) for t in self.trades if t.get('profit_loss_pct') is not None]

            if trade_returns:
                plt.hist(trade_returns, bins=20)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Trade Return Distribution')
                plt.xlabel('Return (%)')
                plt.ylabel('Number of Trades')
                plt.grid(True)

                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                charts['trade_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()

        return charts


def compare_ml_models(symbol, start_date=None, end_date=None, initial_capital=10000):
    """
    Compare different ML model configurations for a given stock.

    Args:
        symbol (str): Stock symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital for trading

    Returns:
        dict: Comparison results
    """
    # Define different configurations to test
    configurations = [
        {
            'name': 'High Confidence Strategy',
            'confidence_threshold': 0.75,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        {
            'name': 'Medium Confidence Strategy',
            'confidence_threshold': 0.60,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        {
            'name': 'Aggressive Strategy',
            'confidence_threshold': 0.50,
            'stop_loss_pct': 0.10,
            'take_profit_pct': 0.20
        },
        {
            'name': 'Conservative Strategy',
            'confidence_threshold': 0.70,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.07
        }
    ]

    results = {}

    for config in configurations:
        # Run backtest with this configuration
        backtester = MLBacktester(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            confidence_threshold=config['confidence_threshold'],
            stop_loss_pct=config['stop_loss_pct'],
            take_profit_pct=config['take_profit_pct']
        )

        backtest_result = backtester.run_backtest()

        if backtest_result['success']:
            # Store the result
            results[config['name']] = {
                'metrics': backtest_result['metrics'],
                'num_trades': len(backtest_result['trades']),
                'config': config
            }

    # Find best strategy based on risk-adjusted return
    best_strategy = None
    best_score = -float('inf')

    for name, result in results.items():
        metrics = result['metrics']

        # Calculate score based on return and risk metrics
        # Use Sharpe ratio if available, otherwise use return / max_drawdown
        if metrics['sharpe_ratio'] > 0:
            score = metrics['sharpe_ratio']
        else:
            score = metrics['percent_return'] / max(0.1, metrics['max_drawdown'])

        if score > best_score:
            best_score = score
            best_strategy = name

    return