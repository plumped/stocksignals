# stock_analyzer/ml_models.py
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import os
from datetime import datetime, timedelta
import logging
from .models import Stock, StockData, AnalysisResult, MLPrediction, MLModelMetrics

logger = logging.getLogger(__name__)


class MLPredictor:
    """Machine Learning model for stock price prediction and signal generation"""

    def __init__(self, stock_symbol, prediction_days=5, training_window=365):
        self.stock_symbol = stock_symbol
        self.prediction_days = prediction_days

        # 🚀 Training Window automatisch anpassen je nach Volatilität
        self.training_window = self._adjust_training_window(training_window)

        self.models_dir = 'ml_models'

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.price_model_path = os.path.join(self.models_dir, f'{stock_symbol}_price_model.pkl')
        self.signal_model_path = os.path.join(self.models_dir, f'{stock_symbol}_signal_model.pkl')

        self.price_model = self._load_or_train_model('price')
        self.signal_model = self._load_or_train_model('signal')

    def prepare_data(self):
        """Prepare data for model training and prediction"""
        try:
            # Get stock data
            stock = Stock.objects.get(symbol=self.stock_symbol)

            # Get historical data with enough history for training
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.training_window + 100)  # Extra buffer

            data = StockData.objects.filter(
                stock=stock,
                date__range=[start_date, end_date]
            ).order_by('date')

            if not data.exists():
                logger.error(f"No data found for {self.stock_symbol} in the specified date range")
                return None, None, None

            # Convert to DataFrame
            df = pd.DataFrame(list(data.values()))

            # Convert Decimal fields to float before any calculations
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                df[col] = df[col].astype(float)

            # Calculate additional features
            df = self._calculate_features(df)

            # Make a copy of the dataframe to avoid SettingWithCopyWarning
            features = df.copy()

            # Drop any rows with NaN values first
            features = features.dropna()

            if len(features) < 30:  # Need at least 30 data points for meaningful modeling
                logger.error(f"Not enough data points for {self.stock_symbol} after feature calculation")
                return None, None, None

            # For price prediction: target is the next n-day percentage change
            features.loc[:, 'future_return'] = features['close_price'].pct_change(self.prediction_days).shift(
                -self.prediction_days)

            # For signal prediction: create a categorical target (1=Buy, 0=Hold, -1=Sell) based on future returns
            features.loc[:, 'signal_target'] = 0
            threshold = 0.02  # 2% movement threshold for signal
            features.loc[features['future_return'] > threshold, 'signal_target'] = 1
            features.loc[features['future_return'] < -threshold, 'signal_target'] = -1

            # Remove NaN values
            features = features.dropna()

            # Features for models (exclude target columns and non-feature columns)
            feature_columns = [col for col in features.columns if col not in
                               ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

            X = features[feature_columns]
            y_price = features['future_return']
            y_signal = features['signal_target']

            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            return X_scaled, y_price, y_signal

        except Exception as e:
            logger.error(f"Error preparing data for {self.stock_symbol}: {str(e)}")
            return None, None, None

    def _adjust_training_window(self, base_window):
        """Dynamically adjust training window based on stock volatility"""
        try:
            from .models import Stock, StockData

            stock = Stock.objects.get(symbol=self.stock_symbol)
            recent_data = StockData.objects.filter(stock=stock).order_by('-date')[:60]

            if recent_data.count() < 30:
                return base_window

            df = pd.DataFrame(list(recent_data.values()))
            df['close_price'] = df['close_price'].astype(float)
            df['high_price'] = df['high_price'].astype(float)
            df['low_price'] = df['low_price'].astype(float)

            high_low = df['high_price'] - df['low_price']
            high_close = abs(df['high_price'] - df['close_price'].shift())
            low_close = abs(df['low_price'] - df['close_price'].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)

            atr = true_range.rolling(window=14).mean().iloc[-1]
            last_close = df['close_price'].iloc[-1]
            atr_pct = (atr / last_close) * 100 if last_close != 0 else 5

            # 🚀 Anpassungslogik
            if atr_pct < 2.0:
                adjusted = base_window * 2
            elif atr_pct < 4.0:
                adjusted = int(base_window * 1.5)
            else:
                adjusted = base_window

            logger.info(
                f"Adjusted training window for {self.stock_symbol}: {base_window} → {adjusted} days (ATR {atr_pct:.2f}%)")
            return adjusted

        except Exception as e:
            logger.error(f"Error adjusting training window for {self.stock_symbol}: {str(e)}")
            return base_window

    def _calculate_features(self, df):
        """Calculate technical indicators and other features for ML models"""
        # Copy dataframe to avoid modifying the original
        df_features = df.copy()

        # Ensure all numeric columns are float type
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        # Calculate returns
        df_features['daily_return'] = df_features['close_price'].pct_change()
        df_features['weekly_return'] = df_features['close_price'].pct_change(5)
        df_features['monthly_return'] = df_features['close_price'].pct_change(20)

        # Price ratios
        df_features['hl_ratio'] = df_features['high_price'] / df_features['low_price']
        df_features['co_ratio'] = df_features['close_price'] / df_features['open_price']

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df_features[f'ma_{window}'] = df_features['close_price'].rolling(window=window).mean()
            # Distance from moving average (%)
            df_features[f'ma_{window}_dist'] = (df_features['close_price'] - df_features[f'ma_{window}']) / df_features[
                f'ma_{window}']

        # Moving average crossovers
        df_features['ma_5_10_cross'] = df_features['ma_5'] - df_features['ma_10']
        df_features['ma_10_50_cross'] = df_features['ma_10'] - df_features['ma_50']
        df_features['ma_50_200_cross'] = df_features['ma_50'] - df_features['ma_200']

        # Volatility measures
        df_features['volatility_5'] = df_features['daily_return'].rolling(window=5).std()
        df_features['volatility_20'] = df_features['daily_return'].rolling(window=20).std()

        # Volume features
        df_features['volume_ma_5'] = df_features['volume'].rolling(window=5).mean()
        df_features['volume_ma_20'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_20']

        # RSI
        delta = df_features['close_price'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df_features['rsi'] = 100 - (100 / (1 + rs))

        # Candlestick Features
        df_features['candle_body'] = df_features['close_price'] - df_features['open_price']
        df_features['upper_shadow'] = df_features['high_price'] - df_features[['close_price', 'open_price']].max(axis=1)
        df_features['lower_shadow'] = df_features[['close_price', 'open_price']].min(axis=1) - df_features['low_price']

        # Trend Direction
        df_features['is_bullish'] = (df_features['close_price'] > df_features['open_price']).astype(int)

        # MACD
        df_features['ema_12'] = df_features['close_price'].ewm(span=12, adjust=False).mean()
        df_features['ema_26'] = df_features['close_price'].ewm(span=26, adjust=False).mean()
        df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
        df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
        df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

        # Lag-Features
        for lag in [1, 2, 3]:
            df_features[f'close_lag_{lag}'] = df_features['close_price'].shift(lag)
            df_features[f'rsi_lag_{lag}'] = df_features['rsi'].shift(lag)
            df_features[f'macd_lag_{lag}'] = df_features['macd'].shift(lag)

        # Volatility Classification
        df_features['volatility_category'] = pd.qcut(df_features['volatility_20'], q=3, labels=[0, 1, 2])

        # Rolling Momentum
        df_features['trend_strength_10'] = df_features['close_price'].diff(10)

        # Signal Counter
        df_features['bullish_signals'] = (
                (df_features['macd'] > df_features['macd_signal']).astype(int) +
                (df_features['rsi'] < 30).astype(int) +
                (df_features['close_price'] > df_features['ma_20']).astype(int)
        )



        # Bollinger Bands
        df_features['bb_middle'] = df_features['close_price'].rolling(window=20).mean()
        std = df_features['close_price'].rolling(window=20).std()
        df_features['bb_upper'] = df_features['bb_middle'] + 2 * std
        df_features['bb_lower'] = df_features['bb_middle'] - 2 * std
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']

        # Ensure bb_upper and bb_lower are not the same to avoid division by zero
        df_features['bb_position'] = np.where(
            (df_features['bb_upper'] - df_features['bb_lower']) > 0,
            (df_features['close_price'] - df_features['bb_lower']) / (
                        df_features['bb_upper'] - df_features['bb_lower']),
            0.5  # Default value when bands are identical
        )

        # Price momentum
        for window in [5, 10, 20]:
            df_features[f'momentum_{window}'] = df_features['close_price'] / df_features['close_price'].shift(
                window) - 1

        # Check for any infinities or NaNs and replace with 0
        df_features = df_features.replace([np.inf, -np.inf], np.nan)

        df_features = df_features.dropna()

        return df_features

    def _load_or_train_model(self, model_type):
        """Load a saved model or train a new one if no saved model exists"""
        model_path = self.price_model_path if model_type == 'price' else self.signal_model_path

        try:
            # Try to load the model
            if os.path.exists(model_path):
                model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))

                # Retrain model if it's older than 30 days
                if model_age.days > 30:
                    return self._train_model(model_type)

                logger.info(f"Loading existing {model_type} model for {self.stock_symbol}")
                return joblib.load(model_path)
            else:
                # Train a new model if none exists
                return self._train_model(model_type)

        except Exception as e:
            logger.error(f"Error loading {model_type} model for {self.stock_symbol}: {str(e)}")
            # Train a new model if there was an error loading
            return self._train_model(model_type)

    def _train_model(self, model_type):
        """Train a new model with hyperparameter search and optional calibration, then save it."""
        X, y_price, y_signal = self.prepare_data()

        if X is None or len(X) < 30:
            logger.error(f"Insufficient data to train {model_type} model for {self.stock_symbol}")
            return None

        # Zielvariable & CV-Strategie bestimmen
        if model_type == 'signal':
            y = y_signal
            class_counts = y.value_counts()
            min_class_samples = class_counts.min()

            if min_class_samples < 3:
                logger.warning(
                    f"Abbruch: Zu wenige Klassendaten für '{self.stock_symbol}' – minimalste Klasse hat nur {min_class_samples} Beispiel(e)"
                )
                return None

            if min_class_samples >= 20:
                cv_strategy = StratifiedKFold(n_splits=3)
                logger.info(f"[{self.stock_symbol}] StratifiedKFold aktiviert für Klassifikation")
            else:
                n_splits = min(3, min_class_samples)
                cv_strategy = TimeSeriesSplit(n_splits=n_splits)
                logger.info(f"[{self.stock_symbol}] TimeSeriesSplit verwendet (min_class_samples={min_class_samples})")

        else:
            y = y_price
            n_splits = max(5, min(len(X) // 100, 10))
            cv_strategy = TimeSeriesSplit(n_splits=n_splits)

        try:
            # Modell & Parameter definieren
            if model_type == 'price':
                model_path = self.price_model_path
                base_model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 1.0]
                }
                scoring = 'neg_mean_squared_error'
            else:
                model_path = self.signal_model_path
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [4, 6, 8, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                scoring = 'accuracy'

            best_score = float('-inf')
            best_model = None

            for train_idx, test_idx in cv_strategy.split(X, y if model_type == 'signal' else None):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                cv_folds = min(3, y_train.value_counts().min()) if model_type == 'signal' else 3

                # Hyperparameter-Suche
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=cv_folds,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_

                # Kalibrierung
                if model_type == 'signal' and cv_folds >= 2:
                    model = CalibratedClassifierCV(model, method='isotonic', cv=cv_folds)
                    model.fit(X_train, y_train)
                elif model_type == 'signal':
                    logger.warning(f"Kalibrierung für {self.stock_symbol} übersprungen – zu wenige Daten")

                # Bewertung
                if model_type == 'price':
                    score = -mean_squared_error(y_test, model.predict(X_test))
                else:
                    score = accuracy_score(y_test, model.predict(X_test))

                if score > best_score:
                    best_score = score
                    best_model = model

            if best_model is not None:
                joblib.dump(best_model, model_path)
                logger.info(
                    f"Trained and saved new {model_type} model for {self.stock_symbol} with best score: {best_score:.4f}"
                )
                return best_model
            else:
                logger.error(f"Failed to train {model_type} model for {self.stock_symbol}")
                return None

        except Exception as e:
            logger.error(f"Error training {model_type} model for {self.stock_symbol}: {str(e)}")
            return None

    def predict(self):
        """Make predictions with the trained models (verbessert)"""
        try:
            X, _, _ = self.prepare_data()

            if X is None or X.shape[0] == 0:
                logger.error(f"No data available for prediction for {self.stock_symbol}")
                return None

            latest_features = X[-1].reshape(1, -1)

            predicted_return = 0.0
            confidence = 0.0

            # Price Prediction
            if self.price_model is not None:
                predicted_return = self.price_model.predict(latest_features)[0]

            # Signal Prediction
            if self.signal_model is not None and hasattr(self.signal_model, 'predict_proba'):
                probas = self.signal_model.predict_proba(latest_features)
                confidence = max(probas[0])

            # Aktueller Kurs
            stock = Stock.objects.get(symbol=self.stock_symbol)
            current_price_obj = StockData.objects.filter(stock=stock).order_by('-date').first().close_price
            current_price = float(current_price_obj)

            # Vorhergesagter Kurs
            predicted_price = current_price * (1 + predicted_return)

            # NEUE Logik: Empfehlung basierend auf Mindest-Renditen
            min_return_for_buy = 0.02  # Mindestens +2% erwartet
            min_return_for_sell = -0.02  # Mindestens -2% erwartet

            recommendation = 'HOLD'  # Default

            if predicted_return >= min_return_for_buy:
                recommendation = 'BUY'
            elif predicted_return <= min_return_for_sell:
                recommendation = 'SELL'

            # Zusätzlich: Sicherheit bei Penny Stocks
            if current_price < 1.0 and abs(predicted_return) < 0.05:
                recommendation = 'HOLD'

            result = {
                'stock_symbol': self.stock_symbol,
                'current_price': current_price,
                'predicted_return': round(predicted_return * 100, 2),  # in Prozent
                'predicted_price': round(predicted_price, 2),
                'recommendation': recommendation,
                'confidence': round(confidence, 2),
                'prediction_days': self.prediction_days
            }

            self._save_prediction(result)

            return result

        except Exception as e:
            logger.error(f"Error making prediction for {self.stock_symbol}: {str(e)}")
            return None

    def _save_prediction(self, prediction):
        """Save the prediction to the database"""
        try:
            stock = Stock.objects.get(symbol=self.stock_symbol)

            # Create or update the prediction
            MLPrediction.objects.update_or_create(
                stock=stock,
                date=datetime.now().date(),
                defaults={
                    'predicted_return': float(prediction['predicted_return']),
                    'predicted_price': float(prediction['predicted_price']),
                    'recommendation': prediction['recommendation'],
                    'confidence': float(prediction['confidence']),
                    'prediction_days': int(self.prediction_days)
                }
            )

            logger.info(f"Saved prediction for {self.stock_symbol}")

        except Exception as e:
            logger.error(f"Error saving prediction for {self.stock_symbol}: {str(e)}")
            # Nur Logging, keine Exception werfen, um den Prozess nicht zu unterbrechen

    def evaluate_model_performance(self):
        """Evaluate the performance of the models on historical data"""
        try:
            X, y_price, y_signal = self.prepare_data()

            if X is None or y_price is None or y_signal is None:
                return None

            # Split data for evaluation (time-based)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_price_train, y_price_test = y_price[:train_size].to_numpy(), y_price[train_size:].to_numpy()
            y_signal_train, y_signal_test = y_signal[:train_size].to_numpy(), y_signal[train_size:].to_numpy()

            performance = {}

            # Evaluate price prediction model
            if self.price_model is not None:
                price_pred = self.price_model.predict(X_test)
                price_mse = mean_squared_error(y_price_test, price_pred)
                performance['price_mse'] = price_mse
                performance['price_rmse'] = np.sqrt(price_mse)

                # Calculate directional accuracy (correct prediction of up/down)
                direction_actual = np.sign(y_price_test)
                direction_pred = np.sign(price_pred)
                direction_accuracy = np.mean(direction_actual == direction_pred)
                performance['price_direction_accuracy'] = direction_accuracy

            # Evaluate signal prediction model
            if self.signal_model is not None:
                signal_pred = self.signal_model.predict(X_test)
                signal_accuracy = accuracy_score(y_signal_test, signal_pred)
                performance['signal_accuracy'] = signal_accuracy

                # Classification report
                report = classification_report(y_signal_test, signal_pred, output_dict=True)
                performance['classification_report'] = report

                # Feature importance
                if hasattr(self.signal_model, 'feature_importances_'):
                    # Get feature names from original dataframe
                    stock = Stock.objects.get(symbol=self.stock_symbol)
                    df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

                    # Ensure numeric conversion
                    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                        df[col] = df[col].astype(float)

                    df = self._calculate_features(df)
                    feature_columns = [col for col in df.columns if col not in
                                       ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

                    # Create importance dictionary
                    importances = self.signal_model.feature_importances_
                    feature_importance = dict(zip(feature_columns, importances))
                    performance['feature_importance'] = {k: v for k, v in sorted(
                        feature_importance.items(), key=lambda item: item[1], reverse=True
                    )}
            try:
                stock = Stock.objects.get(symbol=self.stock_symbol)

                MLModelMetrics.objects.update_or_create(
                    stock=stock,
                    date=datetime.now().date(),
                    model_version='v1',  # oder später dynamisch, wenn du Versionierung brauchst
                    defaults={
                        'accuracy': performance.get('signal_accuracy', 0),
                        'rmse': performance.get('price_rmse', 0),
                        'feature_importance': performance.get('feature_importance', {}),
                        'confusion_matrix': performance.get('classification_report', {}),
                        'directional_accuracy': performance.get('price_direction_accuracy', 0),
                    }
                )
                logger.info(f"ML-Metriken erfolgreich gespeichert für {self.stock_symbol}")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der ML-Metriken für {self.stock_symbol}: {str(e)}")

            return performance

        except Exception as e:
            logger.error(f"Error evaluating model performance for {self.stock_symbol}: {str(e)}")
            return None


class AdaptiveAnalyzer:
    """
    Analyzer that combines traditional technical analysis with ML predictions
    to create an adaptive scoring system
    """

    def __init__(self, stock_symbol):
        """Initialize the adaptive analyzer"""
        self.stock_symbol = stock_symbol
        self.ml_predictor = MLPredictor(stock_symbol)

    def get_adaptive_score(self):
        """
        Calculate an adaptive technical score that combines traditional
        technical analysis with ML predictions
        """
        try:
            from .analysis import TechnicalAnalyzer

            # === Technische Analyse ausführen ===
            ta = TechnicalAnalyzer(self.stock_symbol)
            ta_result = ta.calculate_technical_score()
            ta_score = ta_result['score']
            ta_recommendation = ta_result['recommendation']

            print(f"[DEBUG] TA Score: {ta_score:.2f} | Empfehlung: {ta_recommendation}")

            # === ML-Vorhersage holen ===
            ml_prediction = self.ml_predictor.predict()

            if ml_prediction is None:
                print("[DEBUG] ML Prediction fehlgeschlagen – verwende nur TA")
                return ta_result

            # === ML-Auswertung ===
            confidence = ml_prediction['confidence']
            predicted_return = ml_prediction['predicted_return']  # bereits in Prozent
            ml_score_modifier = 0

            if ml_prediction['recommendation'] == 'BUY':
                ml_score_modifier = +10 * confidence
            elif ml_prediction['recommendation'] == 'SELL':
                ml_score_modifier = -10 * confidence

            # === Return modifier berechnen und begrenzen ===
            return_modifier = predicted_return  # z. B. +3.5 oder -6.7
            ml_modifier_total = ml_score_modifier + return_modifier

            # Begrenze die ML-Modifikation auf ±20 Punkte
            ml_modifier_total = max(-20, min(20, ml_modifier_total))

            # === Gewichtung ML/TA ===
            ml_weight = min(0.3, confidence * 0.4)
            ta_weight = 1 - ml_weight

            print(f"[DEBUG] ML Empfehlung: {ml_prediction['recommendation']}, Confidence: {confidence:.2f}")
            print(f"[DEBUG] ML Modifier: {ml_score_modifier:.2f} + {return_modifier:.2f} = {ml_modifier_total:.2f}")
            print(f"[DEBUG] Weights → TA: {ta_weight:.2f}, ML: {ml_weight:.2f}")

            # === Adaptive Score berechnen ===
            raw_adaptive_score = (ta_weight * ta_score) + (ml_weight * (ta_score + ml_modifier_total))
            adaptive_score = max(0, min(100, raw_adaptive_score))

            print(f"[DEBUG] Adaptive Score (roh): {raw_adaptive_score:.2f}")
            print(f"[DEBUG] Adaptive Score (begrenzt): {adaptive_score:.2f}")

            # === Empfehlung basierend auf adaptivem Score ===
            if adaptive_score >= 90:
                recommendation = "STRONG BUY"
            elif adaptive_score >= 70:
                recommendation = "BUY"
            elif adaptive_score >= 40:
                recommendation = "HOLD"
            elif adaptive_score >= 20:
                recommendation = "SELL"
            else:
                recommendation = "STRONG SELL"

            # === Signale kombinieren ===
            ml_signals = [
                ("ML Prediction", ml_prediction['recommendation'],
                 f"ML prognostiziert {predicted_return:.2f}% in {ml_prediction['prediction_days']} Tagen")
            ]
            all_signals = ta_result['signals'] + ml_signals

            return {
                'score': round(adaptive_score, 2),
                'recommendation': recommendation,
                'signals': all_signals,
                'details': ta_result['details'],
                'confluence_score': ta_result.get('confluence_score'),
                'ml_prediction': ml_prediction
            }

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating adaptive score for {self.stock_symbol}: {str(e)}")
            from .analysis import TechnicalAnalyzer
            ta = TechnicalAnalyzer(self.stock_symbol)
            return ta.calculate_technical_score()

    def save_analysis_result(self):
        """Save the adaptive analysis result"""
        try:
            from .models import Stock

            result = self.get_adaptive_score()
            stock = Stock.objects.get(symbol=self.stock_symbol)

            # Current date
            from datetime import datetime
            date = datetime.now().date()

            # Get details from the result
            details = result.get('details', {})

            # Save to AnalysisResult
            analysis_result, created = AnalysisResult.objects.update_or_create(
                stock=stock,
                date=date,
                defaults={
                    'technical_score': result['score'],
                    'recommendation': result['recommendation'],
                    'confluence_score': result.get('confluence_score'),
                    'rsi_value': details.get('rsi'),
                    'macd_value': details.get('macd'),
                    'macd_signal': details.get('macd_signal'),
                    'sma_20': details.get('sma_20'),
                    'sma_50': details.get('sma_50'),
                    'sma_200': details.get('sma_200'),
                    'bollinger_upper': details.get('bollinger_upper'),
                    'bollinger_lower': details.get('bollinger_lower')
                }
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Error saving adaptive analysis for {self.stock_symbol}: {str(e)}")
            # Fall back to traditional analysis save
            from .analysis import TechnicalAnalyzer
            ta = TechnicalAnalyzer(self.stock_symbol)
            return ta.save_analysis_result()


# stock_analyzer/ml_models.py

def batch_ml_predictions(symbols=None, force_retrain=False):
    """Run ML predictions for multiple stocks"""
    from .models import Stock, StockData
    from django.db.models import Count
    from .ml_models import MLPredictor

    if symbols is None:
        stocks_with_data = StockData.objects.values('stock') \
            .annotate(data_count=Count('id')) \
            .filter(data_count__gte=200)
        stock_ids = [item['stock'] for item in stocks_with_data]
        symbols = Stock.objects.filter(id__in=stock_ids).values_list('symbol', flat=True)

    results = {}

    for symbol in symbols:
        try:
            predictor = MLPredictor(symbol)

            if force_retrain:
                predictor._train_model('price')
                predictor._train_model('signal')

            prediction = predictor.predict()

            if prediction:
                predicted_return = prediction['predicted_return']
                confidence = prediction['confidence']

                if predicted_return > 0.05 and confidence >= 0.65:
                    signal_color = "green"
                elif 0.01 < predicted_return <= 0.05 or 0.50 <= confidence < 0.65:
                    signal_color = "yellow"
                else:
                    signal_color = "red"

                results[symbol] = {
                    'status': 'success',
                    'prediction': prediction,
                    'signal_color': signal_color
                }
            else:
                results[symbol] = {
                    'status': 'error',
                    'message': 'Prediction failed'
                }
        except Exception as e:
            results[symbol] = {
                'status': 'error',
                'message': str(e)
            }

    return results

