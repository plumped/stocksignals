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
import os
import joblib
import tempfile

from .models import Stock, StockData, MLPrediction, MLModelMetrics

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

        # Pre-load the ML model if it exists
        self.model_exists = False
        model_dir = 'ml_models'
        price_model_path = os.path.join(model_dir, f'{symbol}_price_model.pkl')
        signal_model_path = os.path.join(model_dir, f'{symbol}_signal_model.pkl')

        if os.path.exists(price_model_path) and os.path.exists(signal_model_path):
            self.model_exists = True
            self.price_model = joblib.load(price_model_path)
            self.signal_model = joblib.load(signal_model_path)

        print(f"ML-Modelle geladen: {self.model_exists}")

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

        # Check if the stock has enough historical data for ML modeling
        all_price_data = StockData.objects.filter(stock=self.stock).count()
        if all_price_data < 200:
            logger.warning(
                f"Insufficient historical data for {self.symbol}. Need at least 200 data points for ML modeling.")
            self._has_sufficient_data = False
            return False

        # Check if the stock has ML model data or can we create it
        if not self.model_exists:
            logger.warning(f"No ML model found for {self.symbol}. Will use mock predictions.")

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
        test_days = df['date'].tolist()

        # For each trading day in our backtest period
        for i, test_date in enumerate(test_days):
            if i % 10 == 0:  # Log progress every 10 days
                logger.info(f"Processing day {i + 1} of {len(test_days)}")

            # Get current price
            current_row = df[df['date'] == test_date]

            # Skip if no data for this date
            if current_row.empty:
                continue

            current_price = float(current_row['close_price'].iloc[0])

            # Generate prediction for this day
            # Use either the pre-loaded model or create a mock prediction
            if self.model_exists:
                prediction = self._generate_prediction_from_model(test_date)
            else:
                prediction = self._generate_mock_prediction(test_date, current_price)

            if prediction:
                # Store the prediction signal
                signal = {
                    'date': test_date,
                    'price': current_price,
                    'prediction': prediction['recommendation'],
                    'predicted_return': prediction['predicted_return'],
                    'confidence': prediction['confidence']
                }
                self.signals.append(signal)

                # Execute trading logic based on prediction
                self._execute_trading_strategy(signal)

            # Record daily portfolio value
            self._update_portfolio_value(test_date, current_price)

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

    def _generate_prediction_from_model(self, test_date):
        """
        Generate predictions using the actual ML model.

        Args:
            test_date: The date for which to make predictions

        Returns:
            dict: Prediction results or None if prediction fails
        """
        try:
            from .ml_models import MLPredictor

            # Instantiate the predictor
            predictor = MLPredictor(stock_symbol=self.symbol, prediction_days=self.prediction_days)

            # Prepare data
            X, _, _ = predictor.prepare_data()

            if X is None or X.shape[0] == 0:
                logger.warning(f"No valid feature data available for prediction on {test_date}")
                return None

            latest_features = X[-1].reshape(1, -1)

            predicted_return = 0.0
            confidence = 0.0

            if predictor.price_model is not None:
                predicted_return = predictor.price_model.predict(latest_features)[0]

            if predictor.signal_model is not None and hasattr(predictor.signal_model, 'predict_proba'):
                probas = predictor.signal_model.predict_proba(latest_features)
                confidence = max(probas[0])
                signal_class = predictor.signal_model.predict(latest_features)[0]
                recommendation = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}.get(signal_class, 'HOLD')
            else:
                recommendation = 'HOLD'

            # Aktueller Preis
            current_price_obj = StockData.objects.filter(
                stock=self.stock,
                date__lte=test_date
            ).order_by('-date').first()

            if not current_price_obj:
                return None

            current_price = float(current_price_obj.close_price)
            predicted_price = current_price * (1 + predicted_return)

            return {
                'stock_symbol': self.symbol,
                'current_price': current_price,
                'predicted_return': round(predicted_return * 100, 2),
                'predicted_price': round(predicted_price, 2),
                'recommendation': recommendation,
                'confidence': round(confidence, 2),
                'prediction_days': self.prediction_days
            }

        except Exception as e:
            logger.error(f"Error generating real ML prediction for {self.symbol} on {test_date}: {str(e)}")
            return None

    def _generate_mock_prediction(self, test_date, current_price):
        """
        Generate a mock prediction when ML model isn't available.
        This uses technical signals to simulate what an ML model might predict.

        Args:
            test_date: The date for which to make predictions
            current_price: Current price of the stock

        Returns:
            dict: Mock prediction results
        """
        try:
            # Get historical data leading up to this date
            historical_data = StockData.objects.filter(
                stock=self.stock,
                date__lt=test_date
            ).order_by('-date')[:30]  # Last 30 days

            if historical_data.count() < 20:
                return None

            # Calculate simple technical indicators
            prices = [float(data.close_price) for data in historical_data]
            prices.reverse()  # Put in chronological order

            if len(prices) < 20:
                return None

            # Simple moving averages
            sma_5 = sum(prices[-5:]) / 5
            sma_20 = sum(prices[-20:]) / 20

            # Generate a mock prediction based on SMAs
            if sma_5 > sma_20:
                prediction = "BUY"
                predicted_return = np.random.uniform(0.01, 0.05)  # 1-5% positive return
                confidence = np.random.uniform(0.60, 0.80)  # 60-80% confidence
            elif sma_5 < sma_20:
                prediction = "SELL"
                predicted_return = np.random.uniform(-0.05, -0.01)  # 1-5% negative return
                confidence = np.random.uniform(0.60, 0.80)  # 60-80% confidence
            else:
                prediction = "HOLD"
                predicted_return = np.random.uniform(-0.01, 0.01)  # -1% to 1% return
                confidence = np.random.uniform(0.40, 0.60)  # 40-60% confidence

            # Add some randomness to simulate ML model behavior
            if np.random.random() < 0.2:  # 20% chance to flip the prediction
                if prediction == "BUY":
                    prediction = "SELL"
                    predicted_return *= -1
                elif prediction == "SELL":
                    prediction = "BUY"
                    predicted_return *= -1

            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)
            print(f"MOCK {test_date}: {prediction=} {predicted_return=} {confidence=}")

            return {
                'stock_symbol': self.symbol,
                'current_price': current_price,
                'predicted_return': round(predicted_return * 100, 2),  # in percent
                'predicted_price': round(predicted_price, 2),
                'recommendation': prediction,
                'confidence': round(confidence, 2),
                'prediction_days': self.prediction_days
            }

        except Exception as e:
            logger.error(f"Error generating mock prediction for {self.symbol} on {test_date}: {str(e)}")
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

        print(f"{date}: {prediction} @ {confidence:.2f}")

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
            'reason': 'ML_BUY_SIGNAL',
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
            'reason': 'ML_SELL_SIGNAL',
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
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'percent_return': 0,
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
        total_return_pct = (total_return / initial_value) * 100 if initial_value > 0 else 0

        # Calculate daily returns
        portfolio_values = [day['portfolio_value'] for day in self.daily_portfolio_values]
        daily_returns = []

        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1] if \
            portfolio_values[i - 1] > 0 else 0
            daily_returns.append(daily_return)

        # Annualized metrics
        days = (self.end_date - self.start_date).days
        if days > 0:
            annualized_return = ((final_value / initial_value) ** (365 / days) - 1) * 100 if initial_value > 0 else 0
        else:
            annualized_return = 0

        # Risk metrics
        if daily_returns and len(daily_returns) > 1:
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
            drawdown = (running_max - value) / running_max * 100 if running_max > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Trading metrics
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) is not None and t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) is not None and t.get('profit_loss', 0) <= 0]

        num_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

        # Gross profit and loss
        gross_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        gross_loss = sum(abs(t.get('profit_loss', 0)) for t in losing_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Store the metrics
        self.metrics = {
            'initial_capital': initial_value,
            'final_capital': final_value,
            'total_return': total_return,
            'percent_return': total_return_pct,
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

        # Calculate buy and hold return for comparison
        buy_hold_return = 0
        if len(self.daily_portfolio_values) >= 2:
            try:
                first_price = float(StockData.objects.filter(
                    stock=self.stock,
                    date__gte=self.start_date
                ).order_by('date').first().close_price)

                last_price = float(StockData.objects.filter(
                    stock=self.stock,
                    date__lte=self.end_date
                ).order_by('-date').first().close_price)

                buy_hold_return = (last_price - first_price) / first_price * 100 if first_price > 0 else 0
                self.metrics['buy_hold_return'] = buy_hold_return
            except:
                self.metrics['buy_hold_return'] = 0
        else:
            self.metrics['buy_hold_return'] = 0

        # Return complete results
        return {
            'success': True,
            'metrics': self.metrics,
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
            try:
                start_price = float(StockData.objects.filter(
                    stock=self.stock,
                    date__gte=self.start_date
                ).order_by('date').first().close_price)

                # Calculate how many shares could be bought with initial capital
                shares_bought = self.initial_capital / start_price if start_price > 0 else 0

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
            except:
                pass

        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
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
            drawdown = (running_max - value) / running_max * 100 if running_max > 0 else 0
            drawdowns.append(drawdown)

        plt.plot(dates, drawdowns)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization

        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
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
                plt.savefig(buffer, format='png', bbox_inches='tight')
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
            'name': 'Conservative Strategy',
            'confidence_threshold': 0.70,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.07
        },
        {
            'name': 'Balanced Strategy',
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
            'name': 'High Confidence Strategy',
            'confidence_threshold': 0.75,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
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
            # Avoid division by zero
            max_drawdown = max(0.1, metrics['max_drawdown'])
            score = metrics['percent_return'] / max_drawdown

        if score > best_score:
            best_score = score
            best_strategy = name

    # Return the comparison results
    return {
        'results': results,
        'best_strategy': best_strategy if best_strategy else "No strategy performed well"
    }