# stock_analyzer/ml_models.py
import os
# Set environment variable to silence joblib warning about CPU cores
# User has 12 cores and wants to use as many as possible
os.environ["LOKY_MAX_CPU_COUNT"] = "12"
# Set OpenMP threads to 1 to control parallelism in numerical libraries
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from django.db.models import Count
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.inspection import permutation_importance
import shap
from scipy import stats
import joblib
from datetime import datetime, timedelta
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Attention, MultiHeadAttention, LayerNormalization, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from .models import Stock, StockData, AnalysisResult, MLPrediction, MLModelMetrics

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regimes using unsupervised learning algorithms.

    This class implements various clustering algorithms to identify different market regimes
    based on price, volatility, volume, and other technical indicators.
    """

    def __init__(self, stock_symbol, lookback_period=252, n_regimes=4, algorithm='kmeans'):
        """
        Initialize the market regime detector.

        Args:
            stock_symbol (str): The stock symbol to analyze
            lookback_period (int): Number of days to look back for regime detection
            n_regimes (int): Number of regimes to identify
            algorithm (str): Clustering algorithm to use ('kmeans', 'gmm', or 'dbscan')
        """
        self.stock_symbol = stock_symbol
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
        self.regime_characteristics = {}
        self.regime_transitions = []

    def prepare_data(self):
        """
        Prepare data for regime detection.

        Returns:
            tuple: (X, dates) where X is the feature matrix and dates are the corresponding dates
        """
        try:
            # Get stock data
            stock = Stock.objects.get(symbol=self.stock_symbol)
            data = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            if len(data) < 30:
                logger.warning(f"Insufficient data for regime detection for {self.stock_symbol}")
                return None, None

            # Ensure numeric conversion
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                data[col] = data[col].astype(float)

            # Calculate features for regime detection
            data = self._calculate_regime_features(data)

            # Limit to lookback period
            if len(data) > self.lookback_period:
                data = data.iloc[-self.lookback_period:]

            # Store dates for reference
            dates = data['date'].values

            # Select features for clustering
            feature_cols = [col for col in data.columns if col not in ['date', 'stock_id', 'id']]
            self.feature_names = feature_cols

            # Create feature matrix
            X = data[feature_cols].values

            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)

            return X, dates

        except Exception as e:
            logger.error(f"Error preparing data for regime detection: {str(e)}")
            return None, None

    def _calculate_regime_features(self, data):
        """
        Calculate features for regime detection.

        Args:
            data (pd.DataFrame): Stock data

        Returns:
            pd.DataFrame: Data with additional features for regime detection
        """
        # Returns
        data['daily_return'] = data['close_price'].pct_change().fillna(0)
        data['weekly_return'] = data['close_price'].pct_change(5).fillna(0)
        data['monthly_return'] = data['close_price'].pct_change(21).fillna(0)

        # Volatility
        data['volatility_5d'] = data['daily_return'].rolling(window=5).std().fillna(0)
        data['volatility_21d'] = data['daily_return'].rolling(window=21).std().fillna(0)

        # Moving averages
        data['sma_10'] = data['close_price'].rolling(window=10).mean().fillna(0)
        data['sma_50'] = data['close_price'].rolling(window=50).mean().fillna(0)

        # Moving average ratios
        data['sma_ratio'] = (data['sma_10'] / data['sma_50']).fillna(1)

        # Price relative to moving averages
        data['price_to_sma_10'] = (data['close_price'] / data['sma_10']).fillna(1)
        data['price_to_sma_50'] = (data['close_price'] / data['sma_50']).fillna(1)

        # Volume features
        data['volume_sma_10'] = data['volume'].rolling(window=10).mean().fillna(0)
        data['volume_ratio'] = (data['volume'] / data['volume_sma_10']).fillna(1)

        # Trend features
        data['trend_10d'] = (data['close_price'] - data['close_price'].shift(10)) / data['close_price'].shift(10)
        data['trend_10d'] = data['trend_10d'].fillna(0)

        # RSI (simplified)
        delta = data['close_price'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        rs = gain / loss.replace(0, np.nan)
        data['rsi_14'] = 100 - (100 / (1 + rs))
        data['rsi_14'] = data['rsi_14'].fillna(50)

        # Bollinger Bands
        data['bb_middle'] = data['close_price'].rolling(window=20).mean().fillna(0)
        data['bb_std'] = data['close_price'].rolling(window=20).std().fillna(0)
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
        data['bb_width'] = ((data['bb_upper'] - data['bb_lower']) / data['bb_middle']).fillna(0)

        # Price position within Bollinger Bands
        data['bb_position'] = ((data['close_price'] - data['bb_lower']) / 
                              (data['bb_upper'] - data['bb_lower'])).fillna(0.5)

        # Drop rows with NaN values
        data = data.dropna()

        return data

    def detect_regimes(self):
        """
        Detect market regimes using the specified clustering algorithm.

        Returns:
            dict: Dictionary containing regime information
        """
        X, dates = self.prepare_data()

        if X is None or len(X) < 10:
            logger.warning(f"Insufficient data for regime detection for {self.stock_symbol}")
            return None

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction and visualization
        self.pca = PCA(n_components=min(5, X.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)

        # Detect regimes using the specified algorithm
        labels, model = self._apply_clustering(X_scaled)
        self.model = model

        # Analyze regime characteristics
        regime_characteristics = self._analyze_regime_characteristics(X, labels)

        # Detect regime transitions
        regime_transitions = self._detect_regime_transitions(labels, dates)

        # Create regime visualization
        visualization_path = self._create_regime_visualization(X_pca, labels, dates)

        # Determine current regime
        current_regime = int(labels[-1]) if len(labels) > 0 else None
        current_regime_name = self._get_regime_name(current_regime, regime_characteristics)

        # Prepare result
        result = {
            'current_regime': current_regime,
            'current_regime_name': current_regime_name,
            'regime_labels': labels.tolist(),
            'regime_dates': [d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) for d in dates],
            'regime_characteristics': regime_characteristics,
            'regime_transitions': regime_transitions,
            'visualization_path': visualization_path,
            'n_regimes': self.n_regimes,
            'algorithm': self.algorithm
        }

        return result

    def _apply_clustering(self, X):
        """
        Apply the specified clustering algorithm.

        Args:
            X (numpy.ndarray): Scaled feature matrix

        Returns:
            tuple: (labels, model) where labels are the cluster assignments and model is the fitted model
        """
        if self.algorithm == 'kmeans':
            model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = model.fit_predict(X)

        elif self.algorithm == 'gmm':
            model = GaussianMixture(n_components=self.n_regimes, random_state=42, n_init=10)
            labels = model.fit_predict(X)

        elif self.algorithm == 'dbscan':
            # DBSCAN doesn't require a predefined number of clusters
            # We'll estimate a reasonable eps value based on the data
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(X)-1))
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            distances = np.sort(distances[:, -1])
            knee_point = np.argmax(distances[1:] - distances[:-1]) + 1
            eps = distances[knee_point]

            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(X)

            # Handle noise points (label -1) by assigning them to the nearest cluster
            if -1 in labels:
                noise_indices = np.where(labels == -1)[0]
                non_noise_indices = np.where(labels != -1)[0]

                if len(non_noise_indices) > 0:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(X[non_noise_indices])
                    _, indices = nn.kneighbors(X[noise_indices])

                    for i, idx in enumerate(noise_indices):
                        labels[idx] = labels[non_noise_indices[indices[i, 0]]]
                else:
                    # If all points are noise, assign them to a single cluster
                    labels = np.zeros_like(labels)
        else:
            # Default to KMeans
            model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = model.fit_predict(X)

        # Ensure labels are consecutive integers starting from 0
        unique_labels = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])

        return labels, model

    def _analyze_regime_characteristics(self, X, labels):
        """
        Analyze the characteristics of each regime.

        Args:
            X (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Cluster labels

        Returns:
            dict: Dictionary containing regime characteristics
        """
        regime_characteristics = {}

        # Get unique regimes
        unique_regimes = np.unique(labels)

        for regime in unique_regimes:
            # Get data points for this regime
            regime_data = X[labels == regime]

            # Calculate mean values for each feature
            feature_means = np.mean(regime_data, axis=0)

            # Calculate standard deviations for each feature
            # Ensure regime_data is float before using with np.std
            regime_data_float = regime_data.astype(float)
            feature_stds = np.std(regime_data_float, axis=0)

            # Calculate min and max values for each feature
            feature_mins = np.min(regime_data, axis=0)
            feature_maxs = np.max(regime_data, axis=0)

            # Store characteristics
            regime_characteristics[int(regime)] = {
                'size': int(np.sum(labels == regime)),
                'percentage': float(np.mean(labels == regime) * 100),
                'features': {
                    self.feature_names[i]: {
                        'mean': float(feature_means[i]),
                        'std': float(feature_stds[i]),
                        'min': float(feature_mins[i]),
                        'max': float(feature_maxs[i])
                    } for i in range(len(self.feature_names))
                }
            }

            # Determine regime type based on characteristics
            regime_characteristics[int(regime)]['type'] = self._determine_regime_type(
                regime_characteristics[int(regime)]
            )

        self.regime_characteristics = regime_characteristics
        return regime_characteristics

    def _determine_regime_type(self, regime_data):
        """
        Determine the type of market regime based on its characteristics.

        Args:
            regime_data (dict): Characteristics of the regime

        Returns:
            str: Regime type (e.g., 'bull', 'bear', 'sideways', 'volatile')
        """
        features = regime_data['features']

        # Check for key features that determine regime type
        daily_return = features.get('daily_return', {}).get('mean', 0)
        volatility = features.get('volatility_21d', {}).get('mean', 0)
        trend = features.get('trend_10d', {}).get('mean', 0)
        volume_ratio = features.get('volume_ratio', {}).get('mean', 1)

        # Determine regime type based on features
        if trend > 0.02:
            if volatility > 0.02:
                return 'volatile_bull'
            else:
                return 'steady_bull'
        elif trend < -0.02:
            if volatility > 0.02:
                return 'volatile_bear'
            else:
                return 'steady_bear'
        elif volatility > 0.02:
            return 'volatile_sideways'
        else:
            return 'low_volatility_sideways'

    def _get_regime_name(self, regime_id, regime_characteristics):
        """
        Get a descriptive name for a regime.

        Args:
            regime_id (int): Regime ID
            regime_characteristics (dict): Characteristics of all regimes

        Returns:
            str: Descriptive name for the regime
        """
        if regime_id is None or regime_id not in regime_characteristics:
            return "Unknown"

        regime_type = regime_characteristics[regime_id]['type']

        # Map regime types to descriptive names
        regime_names = {
            'steady_bull': 'Stabiler Aufwärtstrend',
            'volatile_bull': 'Volatiler Aufwärtstrend',
            'steady_bear': 'Stabiler Abwärtstrend',
            'volatile_bear': 'Volatiler Abwärtstrend',
            'volatile_sideways': 'Volatile Seitwärtsbewegung',
            'low_volatility_sideways': 'Ruhige Seitwärtsbewegung'
        }

        return regime_names.get(regime_type, regime_type)

    def _detect_regime_transitions(self, labels, dates):
        """
        Detect transitions between regimes.

        Args:
            labels (numpy.ndarray): Cluster labels
            dates (numpy.ndarray): Corresponding dates

        Returns:
            list: List of regime transitions
        """
        transitions = []

        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transition = {
                    'date': dates[i].strftime('%Y-%m-%d') if isinstance(dates[i], datetime) else str(dates[i]),
                    'from_regime': int(labels[i-1]),
                    'to_regime': int(labels[i])
                }
                transitions.append(transition)

        self.regime_transitions = transitions
        return transitions

    def _create_regime_visualization(self, X_pca, labels, dates):
        """
        Create visualization of market regimes.

        Args:
            X_pca (numpy.ndarray): PCA-transformed feature matrix
            labels (numpy.ndarray): Cluster labels
            dates (numpy.ndarray): Corresponding dates

        Returns:
            str: Path to the saved visualization
        """
        try:
            # Create directory for regime visualizations if it doesn't exist
            regime_plot_path = os.path.join('static', 'images', 'regimes')
            os.makedirs(regime_plot_path, exist_ok=True)

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

            # 1. Scatter plot of regimes in PCA space
            if X_pca.shape[1] >= 2:
                # Create a colormap with distinct colors for each regime
                n_regimes = len(np.unique(labels))
                colors = cm.rainbow(np.linspace(0, 1, n_regimes))
                cmap = ListedColormap(colors)

                scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, 
                                     alpha=0.7, s=50, edgecolors='k')

                # Add labels for the last 5 points to show the trajectory
                for i in range(max(0, len(X_pca)-5), len(X_pca)):
                    ax1.annotate(
                        dates[i].strftime('%Y-%m-%d') if isinstance(dates[i], datetime) else str(dates[i]),
                        (X_pca[i, 0], X_pca[i, 1]),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )

                # Connect points with a line to show the trajectory
                ax1.plot(X_pca[:, 0], X_pca[:, 1], 'k-', alpha=0.3)

                # Add a legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=colors[i], markersize=10,
                                             label=f'Regime {i}: {self._get_regime_name(i, self.regime_characteristics)}')
                                  for i in range(n_regimes)]
                ax1.legend(handles=legend_elements, loc='upper right')

                ax1.set_title(f'Market Regimes for {self.stock_symbol} (PCA Visualization)')
                ax1.set_xlabel('Principal Component 1')
                ax1.set_ylabel('Principal Component 2')
                ax1.grid(True, alpha=0.3)

            # 2. Time series plot of regime changes
            # Convert dates to datetime if they're not already
            if not isinstance(dates[0], datetime):
                date_objects = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in dates]
            else:
                date_objects = dates

            # Get stock price data for the same period
            stock = Stock.objects.get(symbol=self.stock_symbol)
            price_data = pd.DataFrame(list(StockData.objects.filter(
                stock=stock, 
                date__in=dates
            ).order_by('date').values()))

            if not price_data.empty:
                # Ensure numeric conversion
                if 'close_price' in price_data.columns:
                    price_data['close_price'] = pd.to_numeric(price_data['close_price'], errors='coerce')

                # Plot stock price
                ax2.plot(date_objects, price_data['close_price'], 'k-', alpha=0.7, label='Close Price')

                # Color the background based on regime
                for regime in np.unique(labels):
                    regime_indices = np.where(labels == regime)[0]
                    if len(regime_indices) > 0:
                        regime_dates = [date_objects[i] for i in regime_indices]
                        regime_prices = [price_data['close_price'].iloc[i] if i < len(price_data) else None for i in regime_indices]
                        regime_prices = [p for p in regime_prices if p is not None]

                        if regime_prices:
                            ax2.scatter(regime_dates[:len(regime_prices)], regime_prices, 
                                       color=colors[regime], alpha=0.7, s=30, 
                                       label=f'Regime {regime}')

                # Add vertical lines for regime transitions
                for transition in self.regime_transitions:
                    transition_date = datetime.strptime(transition['date'], '%Y-%m-%d')
                    ax2.axvline(x=transition_date, color='r', linestyle='--', alpha=0.5)

                ax2.set_title(f'Stock Price and Regime Changes for {self.stock_symbol}')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Price')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left')

            plt.tight_layout()

            # Save the figure
            filename = f'{self.stock_symbol}_market_regimes.png'
            filepath = os.path.join(regime_plot_path, filename)
            plt.savefig(filepath)
            plt.close(fig)

            return f'/static/images/regimes/{filename}'

        except Exception as e:
            logger.error(f"Error creating regime visualization: {str(e)}")
            return None

    def predict_current_regime(self, X=None):
        """
        Predict the current market regime.

        Args:
            X (numpy.ndarray, optional): Feature matrix for prediction. If None, uses the latest data.

        Returns:
            dict: Dictionary containing the predicted regime
        """
        if self.model is None:
            # If model hasn't been trained, detect regimes first
            self.detect_regimes()

        if self.model is None:
            logger.error(f"Failed to train regime detection model for {self.stock_symbol}")
            return None

        if X is None:
            # Get the latest data
            X, _ = self.prepare_data()

            if X is None or len(X) == 0:
                logger.error(f"No data available for regime prediction for {self.stock_symbol}")
                return None

            # Use the last data point
            X = X[-1:].reshape(1, -1)

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Predict regime
        if self.algorithm == 'kmeans':
            regime = self.model.predict(X_scaled)[0]
        elif self.algorithm == 'gmm':
            regime = self.model.predict(X_scaled)[0]
        elif self.algorithm == 'dbscan':
            # For DBSCAN, find the nearest cluster
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.model.components_)
            _, indices = nn.kneighbors(X_scaled)
            regime = self.model.labels_[indices[0, 0]]
        else:
            regime = self.model.predict(X_scaled)[0]

        # Get regime name
        regime_name = self._get_regime_name(regime, self.regime_characteristics)

        return {
            'regime': int(regime),
            'regime_name': regime_name,
            'regime_type': self.regime_characteristics.get(int(regime), {}).get('type', 'unknown')
        }

    def save_regime_data(self):
        """
        Save regime data to the database.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Detect regimes if not already done
            if not hasattr(self, 'regime_characteristics') or not self.regime_characteristics:
                self.detect_regimes()

            # Get the stock
            stock = Stock.objects.get(symbol=self.stock_symbol)

            # Get or create MLModelMetrics for today
            metrics, created = MLModelMetrics.objects.get_or_create(
                stock=stock,
                date=datetime.now().date(),
                defaults={
                    'accuracy': 0.0,
                    'rmse': 0.0
                }
            )

            # Prepare regime data
            regime_data = {
                'current_regime': self.predict_current_regime(),
                'regimes': self.regime_characteristics,
                'transitions': self.regime_transitions,
                'algorithm': self.algorithm,
                'n_regimes': self.n_regimes,
                'visualization_path': self._create_regime_visualization(
                    self.pca.transform(self.scaler.transform(self.prepare_data()[0])),
                    self._apply_clustering(self.scaler.transform(self.prepare_data()[0]))[0],
                    self.prepare_data()[1]
                )
            }

            # Save regime data to the database
            metrics.market_regimes = regime_data
            metrics.save()

            return True

        except Exception as e:
            logger.error(f"Error saving regime data: {str(e)}")
            return False


class LSTMModel(BaseEstimator, RegressorMixin):
    """LSTM model for sequence prediction, compatible with scikit-learn"""

    def __init__(self, sequence_length=10, units=50, dropout=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        """Initialize the LSTM model with parameters"""
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.n_features_in_ = None
        self.scaler = MinMaxScaler()

    def _create_model(self, input_shape):
        """Create the LSTM model architecture"""
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def _prepare_sequences(self, X):
        """Prepare input data as sequences for LSTM"""
        # Reshape data into 3D format: [samples, time steps, features]
        n_samples = X.shape[0] - self.sequence_length
        n_features = X.shape[1]

        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i+self.sequence_length]

        return sequences

    def fit(self, X, y):
        """Fit the LSTM model to the data"""
        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)
        y_seq = y[self.sequence_length:]

        # Create and compile the model
        self.model = self._create_model((self.sequence_length, self.n_features_in_))

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        return self

    def predict(self, X):
        """Make predictions with the LSTM model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # For prediction, we need to handle the case where X has fewer samples than sequence_length
        if X.shape[0] < self.sequence_length:
            # Pad with zeros to reach sequence_length
            padding = np.zeros((self.sequence_length - X.shape[0], X.shape[1]))
            X_padded = np.vstack((padding, X_scaled))
            X_seq = X_padded.reshape(1, self.sequence_length, self.n_features_in_)
        else:
            # Use the last sequence_length samples
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features_in_)

        # Make prediction
        return self.model.predict(X_seq, verbose=0).flatten()


class TransformerModel(BaseEstimator, RegressorMixin):
    """Transformer model with attention mechanisms for sequence prediction"""

    def __init__(self, sequence_length=10, embed_dim=32, num_heads=2, ff_dim=32, dropout=0.2, 
                 learning_rate=0.001, epochs=50, batch_size=32):
        """Initialize the Transformer model with parameters"""
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.n_features_in_ = None
        self.scaler = MinMaxScaler()

    def _create_model(self, input_shape):
        """Create the Transformer model architecture with attention mechanisms"""
        inputs = Input(shape=input_shape)

        # Embedding layer to project input features to embed_dim
        x = Dense(self.embed_dim)(inputs)

        # Add positional information
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = Embedding(input_dim=self.sequence_length, output_dim=self.embed_dim)(positions)
        x = x + position_embedding

        # Transformer block with multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)

        # Feed-forward network
        ff = Dense(self.ff_dim, activation="relu")(x)
        ff = Dense(self.embed_dim)(ff)
        ff = Dropout(self.dropout)(ff)
        x = LayerNormalization(epsilon=1e-6)(ff + x)

        # Global average pooling and output layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        return model

    def _prepare_sequences(self, X):
        """Prepare input data as sequences for Transformer"""
        # Reshape data into 3D format: [samples, time steps, features]
        n_samples = X.shape[0] - self.sequence_length
        n_features = X.shape[1]

        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i+self.sequence_length]

        return sequences

    def fit(self, X, y):
        """Fit the Transformer model to the data"""
        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Prepare sequences
        X_seq = self._prepare_sequences(X_scaled)
        y_seq = y[self.sequence_length:]

        # Create and compile the model
        self.model = self._create_model((self.sequence_length, self.n_features_in_))

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        return self

    def predict(self, X):
        """Make predictions with the Transformer model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # For prediction, we need to handle the case where X has fewer samples than sequence_length
        if X.shape[0] < self.sequence_length:
            # Pad with zeros to reach sequence_length
            padding = np.zeros((self.sequence_length - X.shape[0], X.shape[1]))
            X_padded = np.vstack((padding, X_scaled))
            X_seq = X_padded.reshape(1, self.sequence_length, self.n_features_in_)
        else:
            # Use the last sequence_length samples
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, self.n_features_in_)

        # Make prediction
        return self.model.predict(X_seq, verbose=0).flatten()


class HybridModel(BaseEstimator, RegressorMixin):
    """Hybrid model combining technical and fundamental analysis"""

    def __init__(self, technical_model='lstm', fundamental_weight=0.3, sequence_length=10, 
                 units=50, embed_dim=32, num_heads=2, dropout=0.2, learning_rate=0.001, 
                 epochs=50, batch_size=32):
        """Initialize the hybrid model with parameters"""
        self.technical_model = technical_model
        self.fundamental_weight = fundamental_weight
        self.sequence_length = sequence_length
        self.units = units
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.n_features_in_ = None
        self.scaler_technical = MinMaxScaler()
        self.scaler_fundamental = MinMaxScaler()

    def _create_model(self, technical_shape, fundamental_shape):
        """Create the hybrid model architecture"""
        # Technical analysis branch (sequence data)
        technical_input = Input(shape=technical_shape, name='technical_input')

        # Choose the technical model type
        if self.technical_model == 'transformer':
            # Embedding layer for transformer
            x_technical = Dense(self.embed_dim)(technical_input)

            # Add positional information
            positions = tf.range(start=0, limit=self.sequence_length, delta=1)
            position_embedding = Embedding(input_dim=self.sequence_length, output_dim=self.embed_dim)(positions)
            x_technical = x_technical + position_embedding

            # Transformer block with multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embed_dim
            )(x_technical, x_technical)
            x_technical = LayerNormalization(epsilon=1e-6)(attention_output + x_technical)

            # Feed-forward network
            ff = Dense(self.embed_dim * 2, activation="relu")(x_technical)
            ff = Dense(self.embed_dim)(ff)
            ff = Dropout(self.dropout)(ff)
            x_technical = LayerNormalization(epsilon=1e-6)(ff + x_technical)

            # Global average pooling
            x_technical = tf.keras.layers.GlobalAveragePooling1D()(x_technical)

        else:  # Default to LSTM
            # LSTM layers
            x_technical = LSTM(units=self.units, return_sequences=True)(technical_input)
            x_technical = Dropout(self.dropout)(x_technical)
            x_technical = LSTM(units=self.units)(x_technical)
            x_technical = Dropout(self.dropout)(x_technical)

        # Fundamental analysis branch (static data)
        fundamental_input = Input(shape=fundamental_shape, name='fundamental_input')
        x_fundamental = Dense(32, activation='relu')(fundamental_input)
        x_fundamental = Dropout(self.dropout)(x_fundamental)
        x_fundamental = Dense(16, activation='relu')(x_fundamental)

        # Combine technical and fundamental branches
        combined = Concatenate()([x_technical, x_fundamental])

        # Final layers
        x = Dense(32, activation='relu')(combined)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1)(x)

        # Create model with multiple inputs
        model = Model(inputs=[technical_input, fundamental_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        return model

    def _split_features(self, X):
        """Split features into technical and fundamental groups"""
        # For demonstration, we'll consider the first 70% of features as technical
        # and the remaining 30% as fundamental. In a real implementation, this would
        # be based on actual feature types.
        n_features = X.shape[1]
        n_technical = int(n_features * 0.7)

        technical_features = X[:, :n_technical]
        fundamental_features = X[:, n_technical:]

        return technical_features, fundamental_features

    def _prepare_sequences(self, X):
        """Prepare technical data as sequences"""
        # Reshape data into 3D format: [samples, time steps, features]
        n_samples = X.shape[0] - self.sequence_length
        n_features = X.shape[1]

        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i+self.sequence_length]

        return sequences

    def fit(self, X, y):
        """Fit the hybrid model to the data"""
        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Split features into technical and fundamental
        technical_features, fundamental_features = self._split_features(X)

        # Scale the data
        technical_scaled = self.scaler_technical.fit_transform(technical_features)
        fundamental_scaled = self.scaler_fundamental.fit_transform(fundamental_features)

        # Prepare sequences for technical data
        technical_seq = self._prepare_sequences(technical_scaled)

        # For fundamental data, use the values at the end of each sequence
        fundamental_seq = fundamental_scaled[self.sequence_length:]

        # Target values
        y_seq = y[self.sequence_length:]

        # Create and compile the model
        self.model = self._create_model(
            (self.sequence_length, technical_features.shape[1]),
            (fundamental_features.shape[1],)
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        self.model.fit(
            [technical_seq, fundamental_seq], y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        return self

    def predict(self, X):
        """Make predictions with the hybrid model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Split features into technical and fundamental
        technical_features, fundamental_features = self._split_features(X)

        # Scale the data
        technical_scaled = self.scaler_technical.transform(technical_features)
        fundamental_scaled = self.scaler_fundamental.transform(fundamental_features)

        # For prediction with a single sample or few samples
        if technical_scaled.shape[0] < self.sequence_length:
            # Pad with zeros to reach sequence_length
            padding = np.zeros((self.sequence_length - technical_scaled.shape[0], technical_scaled.shape[1]))
            technical_padded = np.vstack((padding, technical_scaled))
            technical_seq = technical_padded.reshape(1, self.sequence_length, technical_scaled.shape[1])
            fundamental_seq = fundamental_scaled.reshape(1, fundamental_scaled.shape[1])
        else:
            # Use the last sequence_length samples for technical
            technical_seq = technical_scaled[-self.sequence_length:].reshape(1, self.sequence_length, technical_scaled.shape[1])
            # Use the last sample for fundamental
            fundamental_seq = fundamental_scaled[-1:].reshape(1, fundamental_scaled.shape[1])

        # Make prediction
        return self.model.predict([technical_seq, fundamental_seq], verbose=0).flatten()


class ProphetModel(BaseEstimator, RegressorMixin):
    """Facebook Prophet model for time series forecasting, compatible with scikit-learn"""

    def __init__(self, changepoint_prior_scale=0.05, seasonality_mode='additive', 
                 yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto',
                 uncertainty_samples=1000, prediction_horizon=5):
        """Initialize the Prophet model with parameters"""
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.uncertainty_samples = uncertainty_samples
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.n_features_in_ = None
        self.date_column = None
        self.target_column = None
        self.feature_columns = None
        self.last_date = None
        self.freq = 'D'  # Default to daily data

    def fit(self, X, y):
        """Fit the Prophet model to the data"""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet is not installed. Install it with: pip install prophet")

        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Create a DataFrame for Prophet (requires 'ds' and 'y' columns)
        # For simplicity, we'll use the index as the date column
        # In a real implementation, you would extract the date from X
        df = pd.DataFrame()

        # If X is a pandas DataFrame with a DatetimeIndex, use it
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            df['ds'] = X.index
            self.date_column = 'index'
        else:
            # Create a synthetic date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(X))
            df['ds'] = pd.date_range(start=start_date, periods=len(X), freq=self.freq)
            self.date_column = 'synthetic'

        # Add the target variable
        df['y'] = y

        # Store the last date for future predictions
        self.last_date = df['ds'].max()

        # Add regressor columns if X has more than one feature
        self.feature_columns = []
        if X.shape[1] > 0:
            if isinstance(X, pd.DataFrame):
                for col in X.columns:
                    col_name = str(col)
                    df[col_name] = X[col].values
                    self.feature_columns.append(col_name)
            else:
                for i in range(X.shape[1]):
                    col_name = f'feature_{i}'
                    df[col_name] = X[:, i]
                    self.feature_columns.append(col_name)

        # Initialize and fit the Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )

        # Add regressors
        for col in self.feature_columns:
            self.model.add_regressor(col)

        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)

        return self

    def predict(self, X, horizon=None):
        """Make predictions with the Prophet model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Use specified horizon or default
        pred_horizon = horizon if horizon is not None else self.prediction_horizon

        # Create future DataFrame for prediction
        if isinstance(X, pd.DataFrame) and len(X) >= pred_horizon:
            # If X contains the future data points
            future = pd.DataFrame()

            if self.date_column == 'index' and isinstance(X.index, pd.DatetimeIndex):
                future['ds'] = X.index
            else:
                # Create a date range starting from the day after the last training date
                future['ds'] = pd.date_range(
                    start=self.last_date + pd.Timedelta(days=1),
                    periods=pred_horizon,
                    freq=self.freq
                )

            # Add regressor values if available
            if self.feature_columns and len(self.feature_columns) > 0:
                if isinstance(X, pd.DataFrame):
                    for col in self.feature_columns:
                        if col in X.columns:
                            future[col] = X[col].values[:pred_horizon]
                        else:
                            # Use zeros if column not found
                            future[col] = np.zeros(pred_horizon)
                else:
                    for i, col in enumerate(self.feature_columns):
                        if i < X.shape[1]:
                            future[col] = X[:pred_horizon, i]
                        else:
                            future[col] = np.zeros(pred_horizon)
        else:
            # Create a future DataFrame with the specified horizon
            future = self.model.make_future_dataframe(periods=pred_horizon, freq=self.freq)

            # Add regressor values (use the last values from training if not provided)
            if self.feature_columns and len(self.feature_columns) > 0:
                for col in self.feature_columns:
                    future[col] = future[col].fillna(method='ffill')

        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)

        # Return the predicted values
        return forecast['yhat'].values[-pred_horizon:]

    def predict_with_intervals(self, X, horizon=None):
        """Make predictions with uncertainty intervals"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Use specified horizon or default
        pred_horizon = horizon if horizon is not None else self.prediction_horizon

        # Create future DataFrame for prediction (similar to predict method)
        if isinstance(X, pd.DataFrame) and len(X) >= pred_horizon:
            future = pd.DataFrame()

            if self.date_column == 'index' and isinstance(X.index, pd.DatetimeIndex):
                future['ds'] = X.index
            else:
                future['ds'] = pd.date_range(
                    start=self.last_date + pd.Timedelta(days=1),
                    periods=pred_horizon,
                    freq=self.freq
                )

            # Add regressor values if available
            if self.feature_columns and len(self.feature_columns) > 0:
                if isinstance(X, pd.DataFrame):
                    for col in self.feature_columns:
                        if col in X.columns:
                            future[col] = X[col].values[:pred_horizon]
                        else:
                            future[col] = np.zeros(pred_horizon)
                else:
                    for i, col in enumerate(self.feature_columns):
                        if i < X.shape[1]:
                            future[col] = X[:pred_horizon, i]
                        else:
                            future[col] = np.zeros(pred_horizon)
        else:
            future = self.model.make_future_dataframe(periods=pred_horizon, freq=self.freq)

            if self.feature_columns and len(self.feature_columns) > 0:
                for col in self.feature_columns:
                    future[col] = future[col].fillna(method='ffill')

        # Make predictions with uncertainty intervals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)

        # Extract the relevant part of the forecast
        forecast = forecast.iloc[-pred_horizon:]

        # Return a dictionary with predictions and intervals
        return {
            'predictions': forecast['yhat'].values,
            'lower_bound': forecast['yhat_lower'].values,
            'upper_bound': forecast['yhat_upper'].values,
            'dates': forecast['ds'].values
        }

    def get_components(self):
        """Get the trend and seasonal components of the time series"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create a DataFrame covering the historical period
        historical = self.model.history

        # Make predictions for the historical period
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            components = self.model.predict(historical)

        # Extract components
        trend = components['trend']
        seasonality = components['yearly'] if 'yearly' in components else None
        weekly = components['weekly'] if 'weekly' in components else None

        return {
            'trend': trend.values,
            'seasonality': seasonality.values if seasonality is not None else None,
            'weekly': weekly.values if weekly is not None else None,
            'dates': components['ds'].values
        }


class HierarchicalTimeSeriesModel:
    """Hierarchical time series forecasting model"""

    def __init__(self, base_model_class=ProphetModel, reconciliation_method='bottom_up'):
        """
        Initialize the hierarchical time series model

        Args:
            base_model_class: The model class to use for individual time series
            reconciliation_method: Method for reconciling forecasts ('bottom_up', 'top_down', or 'middle_out')
        """
        self.base_model_class = base_model_class
        self.reconciliation_method = reconciliation_method
        self.models = {}
        self.hierarchy = {}
        self.n_features_in_ = None

    def add_time_series(self, name, parent=None):
        """
        Add a time series to the hierarchy

        Args:
            name (str): Name of the time series
            parent (str): Name of the parent time series (None for top level)
        """
        if parent is not None and parent not in self.hierarchy:
            raise ValueError(f"Parent time series '{parent}' not found in hierarchy")

        self.hierarchy[name] = {
            'parent': parent,
            'children': []
        }

        # Update parent's children list
        if parent is not None:
            self.hierarchy[parent]['children'].append(name)

    def fit(self, data_dict, y_dict):
        """
        Fit models for all time series in the hierarchy

        Args:
            data_dict (dict): Dictionary of DataFrames with features for each time series
            y_dict (dict): Dictionary of target values for each time series
        """
        for name in self.hierarchy:
            if name in data_dict and name in y_dict:
                X = data_dict[name]
                y = y_dict[name]

                # Create and fit model for this time series
                model = self.base_model_class()
                model.fit(X, y)

                # Store the model
                self.models[name] = model

                # Store number of features
                if self.n_features_in_ is None:
                    self.n_features_in_ = X.shape[1]

        return self

    def predict(self, data_dict, horizon=5):
        """
        Make hierarchically reconciled predictions

        Args:
            data_dict (dict): Dictionary of DataFrames with features for each time series
            horizon (int): Prediction horizon

        Returns:
            dict: Dictionary of reconciled predictions for each time series
        """
        # Make individual predictions
        individual_predictions = {}
        for name, model in self.models.items():
            if name in data_dict:
                X = data_dict[name]
                individual_predictions[name] = model.predict(X, horizon=horizon)

        # Reconcile predictions based on the selected method
        if self.reconciliation_method == 'bottom_up':
            return self._reconcile_bottom_up(individual_predictions, horizon)
        elif self.reconciliation_method == 'top_down':
            return self._reconcile_top_down(individual_predictions, horizon)
        elif self.reconciliation_method == 'middle_out':
            return self._reconcile_middle_out(individual_predictions, horizon)
        else:
            return individual_predictions

    def _reconcile_bottom_up(self, predictions, horizon):
        """Reconcile predictions using the bottom-up approach"""
        reconciled = predictions.copy()

        # Find leaf nodes (time series with no children)
        leaf_nodes = [name for name in self.hierarchy if not self.hierarchy[name]['children']]

        # Process nodes from bottom to top
        processed = set(leaf_nodes)
        while len(processed) < len(self.hierarchy):
            for name in self.hierarchy:
                if name in processed:
                    continue

                # Check if all children have been processed
                children = self.hierarchy[name]['children']
                if all(child in processed for child in children):
                    # Aggregate children's predictions
                    if children:
                        child_sum = np.zeros(horizon)
                        for child in children:
                            if child in reconciled:
                                child_sum += reconciled[child]

                        # Replace this node's prediction with the sum of its children
                        reconciled[name] = child_sum

                    processed.add(name)

        return reconciled

    def _reconcile_top_down(self, predictions, horizon):
        """Reconcile predictions using the top-down approach"""
        reconciled = predictions.copy()

        # Find the root node (time series with no parent)
        root_nodes = [name for name in self.hierarchy if self.hierarchy[name]['parent'] is None]

        # Process nodes from top to bottom
        for root in root_nodes:
            if root not in reconciled:
                continue

            # Process this subtree
            self._distribute_top_down(root, reconciled, horizon)

        return reconciled

    def _distribute_top_down(self, node, reconciled, horizon):
        """Distribute forecasts from a node to its children"""
        children = self.hierarchy[node]['children']
        if not children:
            return

        # Calculate proportions based on historical data
        total_children = np.zeros(horizon)
        for child in children:
            if child in reconciled:
                total_children += reconciled[child]

        # Distribute the parent's forecast to children based on proportions
        if np.sum(total_children) > 0:
            for child in children:
                if child in reconciled:
                    proportion = reconciled[child] / total_children
                    reconciled[child] = reconciled[node] * proportion

                    # Recursively distribute to this child's children
                    self._distribute_top_down(child, reconciled, horizon)

    def _reconcile_middle_out(self, predictions, horizon):
        """Reconcile predictions using the middle-out approach"""
        # For simplicity, we'll implement a basic version that combines bottom-up and top-down
        # First, identify middle level nodes (e.g., nodes with both parents and children)
        middle_nodes = [
            name for name in self.hierarchy 
            if self.hierarchy[name]['parent'] is not None and self.hierarchy[name]['children']
        ]

        reconciled = predictions.copy()

        # Apply bottom-up from leaf nodes to middle nodes
        leaf_nodes = [name for name in self.hierarchy if not self.hierarchy[name]['children']]
        processed = set(leaf_nodes)

        while not all(node in processed for node in middle_nodes):
            for name in self.hierarchy:
                if name in processed or name not in middle_nodes:
                    continue

                # Check if all children have been processed
                children = self.hierarchy[name]['children']
                if all(child in processed for child in children):
                    # Aggregate children's predictions
                    if children:
                        child_sum = np.zeros(horizon)
                        for child in children:
                            if child in reconciled:
                                child_sum += reconciled[child]

                        # Replace this node's prediction with the sum of its children
                        reconciled[name] = child_sum

                    processed.add(name)

        # Apply top-down from middle nodes to their parents
        for middle_node in middle_nodes:
            current = middle_node
            parent = self.hierarchy[current]['parent']

            while parent is not None:
                # Update parent based on this node and its siblings
                siblings = self.hierarchy[parent]['children']
                sibling_sum = np.zeros(horizon)

                for sibling in siblings:
                    if sibling in reconciled:
                        sibling_sum += reconciled[sibling]

                reconciled[parent] = sibling_sum

                # Move up the hierarchy
                current = parent
                parent = self.hierarchy[current]['parent'] if current in self.hierarchy else None

        return reconciled


class NeuralProphetModel(BaseEstimator, RegressorMixin):
    """NeuralProphet model for time series forecasting, compatible with scikit-learn"""

    def __init__(self, n_changepoints=10, yearly_seasonality=True, weekly_seasonality=True,
                 daily_seasonality=False, batch_size=64, epochs=100, learning_rate=0.001,
                 prediction_horizon=5):
        """Initialize the NeuralProphet model with parameters"""
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.n_features_in_ = None
        self.date_column = None
        self.feature_columns = None
        self.last_date = None
        self.freq = 'D'  # Default to daily data

    def fit(self, X, y):
        """Fit the NeuralProphet model to the data"""
        try:
            from neuralprophet import NeuralProphet
        except ImportError:
            raise ImportError("NeuralProphet is not installed. Install it with: pip install neuralprophet")

        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Create a DataFrame for NeuralProphet (requires 'ds' and 'y' columns)
        df = pd.DataFrame()

        # If X is a pandas DataFrame with a DatetimeIndex, use it
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            df['ds'] = X.index
            self.date_column = 'index'
        else:
            # Create a synthetic date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(X))
            df['ds'] = pd.date_range(start=start_date, periods=len(X), freq=self.freq)
            self.date_column = 'synthetic'

        # Add the target variable
        df['y'] = y

        # Store the last date for future predictions
        self.last_date = df['ds'].max()

        # Add regressor columns if X has more than one feature
        self.feature_columns = []
        if X.shape[1] > 0:
            if isinstance(X, pd.DataFrame):
                for col in X.columns:
                    col_name = str(col)
                    df[col_name] = X[col].values
                    self.feature_columns.append(col_name)
            else:
                for i in range(X.shape[1]):
                    col_name = f'feature_{i}'
                    df[col_name] = X[:, i]
                    self.feature_columns.append(col_name)

        # Initialize the NeuralProphet model
        self.model = NeuralProphet(
            n_changepoints=self.n_changepoints,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate
        )

        # Add regressors
        for col in self.feature_columns:
            self.model.add_regressor(col)

        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df, freq=self.freq)

        return self

    def predict(self, X, horizon=None):
        """Make predictions with the NeuralProphet model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Use specified horizon or default
        pred_horizon = horizon if horizon is not None else self.prediction_horizon

        # Create future DataFrame for prediction
        future = self.model.make_future_dataframe(
            df=pd.DataFrame({'ds': [self.last_date]}),
            periods=pred_horizon,
            n_historic_predictions=False
        )

        # Add regressor values if available
        if self.feature_columns and len(self.feature_columns) > 0:
            if isinstance(X, pd.DataFrame) and len(X) >= pred_horizon:
                for col in self.feature_columns:
                    if col in X.columns:
                        future[col] = X[col].values[:pred_horizon]
                    else:
                        future[col] = np.zeros(pred_horizon)
            elif X.shape[0] >= pred_horizon and X.shape[1] >= len(self.feature_columns):
                for i, col in enumerate(self.feature_columns):
                    future[col] = X[:pred_horizon, i]
            else:
                # Use the last known values if not enough data provided
                for col in self.feature_columns:
                    future[col] = future[col].fillna(method='ffill')

        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)

        # Return the predicted values
        return forecast['yhat1'].values

    def predict_with_intervals(self, X, horizon=None):
        """Make predictions with uncertainty intervals"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Use specified horizon or default
        pred_horizon = horizon if horizon is not None else self.prediction_horizon

        # Create future DataFrame for prediction
        future = self.model.make_future_dataframe(
            df=pd.DataFrame({'ds': [self.last_date]}),
            periods=pred_horizon,
            n_historic_predictions=False
        )

        # Add regressor values if available (similar to predict method)
        if self.feature_columns and len(self.feature_columns) > 0:
            if isinstance(X, pd.DataFrame) and len(X) >= pred_horizon:
                for col in self.feature_columns:
                    if col in X.columns:
                        future[col] = X[col].values[:pred_horizon]
                    else:
                        future[col] = np.zeros(pred_horizon)
            elif X.shape[0] >= pred_horizon and X.shape[1] >= len(self.feature_columns):
                for i, col in enumerate(self.feature_columns):
                    future[col] = X[:pred_horizon, i]
            else:
                for col in self.feature_columns:
                    future[col] = future[col].fillna(method='ffill')

        # Make predictions with uncertainty intervals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)

        # Return a dictionary with predictions and intervals
        return {
            'predictions': forecast['yhat1'].values,
            'lower_bound': forecast['yhat1_lower'].values if 'yhat1_lower' in forecast.columns else None,
            'upper_bound': forecast['yhat1_upper'].values if 'yhat1_upper' in forecast.columns else None,
            'dates': forecast['ds'].values
        }

    def get_components(self):
        """Get the trend and seasonal components of the time series"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create a DataFrame covering the historical period
        historical = pd.DataFrame({'ds': pd.date_range(
            start=self.last_date - pd.Timedelta(days=365),
            end=self.last_date,
            freq=self.freq
        )})

        # Add regressor values (use the last values from training)
        if self.feature_columns and len(self.feature_columns) > 0:
            for col in self.feature_columns:
                historical[col] = 0  # Simplified for demonstration

        # Make predictions for the historical period
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            components = self.model.predict(historical)
            components_dict = self.model.get_components(components)

        # Extract components (structure depends on NeuralProphet implementation)
        return components_dict


class RLTradingModel(BaseEstimator, ClassifierMixin):
    """Reinforcement Learning model for adaptive trading strategies"""

    def __init__(self, state_size=10, action_size=3, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001,
                 batch_size=32, memory_size=1000, train_episodes=100):
        """Initialize the RL model with parameters"""
        self.state_size = state_size
        self.action_size = action_size  # 0: SELL, 1: HOLD, 2: BUY
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.train_episodes = train_episodes

        self.memory = []
        self.model = None
        self.target_model = None
        self.n_features_in_ = None
        self.scaler = MinMaxScaler()
        self.classes_ = np.array([0, 1, 2])  # SELL, HOLD, BUY

    def _build_model(self):
        """Build a neural network model for Q-learning"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def _act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def _replay(self):
        """Train the model with experiences from memory"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.random.choice(self.memory, self.batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _update_target_model(self):
        """Update the target model with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())

    def _create_environment(self, X, y):
        """Create a simple trading environment for training"""
        class TradingEnv:
            def __init__(self, data, targets, window_size):
                self.data = data
                self.targets = targets
                self.window_size = window_size
                self.reset()

            def reset(self):
                self.position = self.window_size
                self.done = False
                self.profit = 0
                return self._get_state()

            def _get_state(self):
                return self.data[self.position-self.window_size:self.position]

            def step(self, action):
                # Move to the next position
                self.position += 1

                # Check if episode is done
                if self.position >= len(self.data) - 1:
                    self.done = True

                # Calculate reward based on action and actual price movement
                actual_movement = np.sign(self.targets[self.position])

                # Convert action to direction: 0 (SELL) -> -1, 1 (HOLD) -> 0, 2 (BUY) -> 1
                action_direction = action - 1

                # Reward is positive if action matches actual movement, negative otherwise
                if action_direction == actual_movement:
                    reward = 1.0
                elif action_direction == 0 or actual_movement == 0:  # HOLD or no movement
                    reward = 0.1
                else:
                    reward = -1.0

                # Update profit (simplified)
                self.profit += reward

                return self._get_state(), reward, self.done, {"profit": self.profit}

        return TradingEnv(X, y, self.state_size)

    def fit(self, X, y):
        """Train the RL model using the data"""
        # Store number of features for prediction
        self.n_features_in_ = X.shape[1]

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Convert y to direction: -1 (down), 0 (no change), 1 (up)
        y_direction = np.sign(y)

        # Create environment
        env = self._create_environment(X_scaled, y_direction)

        # Initialize models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

        # Training loop
        for episode in range(self.train_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            while True:
                action = self._act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self._remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

            self._replay()

            # Update target model periodically
            if episode % 10 == 0:
                self._update_target_model()

        return self

    def predict(self, X):
        """Predict the action (SELL, HOLD, BUY) for the given data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # For prediction with a single sample
        if X_scaled.shape[0] < self.state_size:
            # Pad with zeros to reach state_size
            padding = np.zeros((self.state_size - X_scaled.shape[0], X_scaled.shape[1]))
            X_padded = np.vstack((padding, X_scaled))
            state = X_padded.reshape(1, self.state_size)
        else:
            # Use the last state_size samples
            state = X_scaled[-self.state_size:].reshape(1, self.state_size)

        # Get Q-values
        q_values = self.model.predict(state, verbose=0)

        # Return the action with the highest Q-value
        return np.argmax(q_values, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities for the given data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # For prediction with a single sample
        if X_scaled.shape[0] < self.state_size:
            # Pad with zeros to reach state_size
            padding = np.zeros((self.state_size - X_scaled.shape[0], X_scaled.shape[1]))
            X_padded = np.vstack((padding, X_scaled))
            state = X_padded.reshape(1, self.state_size)
        else:
            # Use the last state_size samples
            state = X_scaled[-self.state_size:].reshape(1, self.state_size)

        # Get Q-values
        q_values = self.model.predict(state, verbose=0)

        # Convert Q-values to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        probabilities = exp_q / np.sum(exp_q, axis=1, keepdims=True)

        return probabilities


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
            print(f"DEBUG: Starting prepare_data for {self.stock_symbol}")
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
            print(f"DEBUG: Got {len(df)} data points for {self.stock_symbol}")

            # Convert Decimal fields to float before any calculations
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # SPY-Daten laden und mit df mergen
            spy_stock = Stock.objects.get(symbol='SPY')
            spy_data = StockData.objects.filter(
                stock=spy_stock,
                date__range=[start_date, end_date]
            ).values('date', 'close_price')

            spy_df = pd.DataFrame(list(spy_data)).rename(columns={'close_price': 'spy_close'})

            if not spy_df.empty:
                df = df.merge(spy_df, on='date', how='left')
                if 'spy_close' in df.columns:
                    df['spy_close'] = pd.to_numeric(df['spy_close'], errors='coerce')

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
            features.loc[:, 'future_return'] = features['close_price'].pct_change(self.prediction_days, fill_method=None).fillna(0).shift(
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

            print(f"DEBUG: Prepared data for {self.stock_symbol}: X shape={X_scaled.shape}, feature count={X_scaled.shape[1]}")
            return X_scaled, y_price, y_signal

        except Exception as e:
            logger.error(f"Error preparing data for {self.stock_symbol}: {str(e)}")
            return None, None, None

    def detect_market_regime(self):
        """
        Detect the current market regime using the MarketRegimeDetector.

        Returns:
            dict: Dictionary containing the current regime information
        """
        try:
            # Create a MarketRegimeDetector instance
            regime_detector = MarketRegimeDetector(
                stock_symbol=self.stock_symbol,
                lookback_period=252,  # Use 1 year of data for regime detection
                n_regimes=4,
                algorithm='kmeans'
            )

            # Detect the current regime
            regime_detector.detect_regimes()

            # Get the current regime
            current_regime = regime_detector.predict_current_regime()

            logger.info(f"Detected market regime for {self.stock_symbol}: {current_regime}")

            return current_regime
        except Exception as e:
            logger.error(f"Error detecting market regime for {self.stock_symbol}: {str(e)}")
            return None

    def _adjust_training_window(self, base_window):
        """Dynamically adjust training window based on stock volatility"""
        try:
            from .models import Stock, StockData

            stock = Stock.objects.get(symbol=self.stock_symbol)
            recent_data = StockData.objects.filter(stock=stock).order_by('-date')[:60]

            if recent_data.count() < 30:
                return base_window

            df = pd.DataFrame(list(recent_data.values()))
            if 'close_price' in df.columns:
                df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
            if 'high_price' in df.columns:
                df['high_price'] = pd.to_numeric(df['high_price'], errors='coerce')
            if 'low_price' in df.columns:
                df['low_price'] = pd.to_numeric(df['low_price'], errors='coerce')

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
        """
        Calculate technical indicators and other features for ML models.
        Implements adaptive feature calculation based on available data amount.
        """
        # Get caller information for debugging
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_line = caller_frame.f_lineno
        print(f"DEBUG: _calculate_features called from {caller_function} at line {caller_line}")

        # Basiskonfiguration für Feature-Sets
        SHORT_WINDOW = 5
        MEDIUM_WINDOW = 10
        STANDARD_WINDOW = 20
        EXTENDED_WINDOW = 50  # Nur wenn genügend Daten vorhanden
        LONG_WINDOW = 200  # Nur wenn genügend Daten vorhanden

        # Initialisierung und Datenkonvertierung
        df_features = df.copy()
        data_length = len(df_features)

        print(f"DEBUG: Feature-Berechnung mit {data_length} Datenpunkten gestartet für {self.stock_symbol}")

        # Prüfe, welche Feature-Sets berechnet werden können
        can_calculate_short = data_length >= SHORT_WINDOW + 5
        can_calculate_medium = data_length >= MEDIUM_WINDOW + 5
        can_calculate_standard = data_length >= STANDARD_WINDOW + 5
        can_calculate_extended = data_length >= EXTENDED_WINDOW + 5
        can_calculate_long = data_length >= LONG_WINDOW + 5

        # Spaltentypen konvertieren
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        # Feature calculation configuration
        feature_config = {
            'windows': {
                'short': SHORT_WINDOW,
                'medium': MEDIUM_WINDOW,
                'standard': STANDARD_WINDOW,
                'extended': EXTENDED_WINDOW,
                'long': LONG_WINDOW
            },
            'can_calculate': {
                'short': can_calculate_short,
                'medium': can_calculate_medium,
                'standard': can_calculate_standard,
                'extended': can_calculate_extended,
                'long': can_calculate_long
            },
            'data_length': data_length
        }

        # Calculate features by category
        df_features = self._calculate_basic_features(df_features)

        # Continue with the rest of the feature calculations
        if feature_config['can_calculate']['short']:
            df_features = self._calculate_short_term_features(df_features, feature_config)

        if feature_config['can_calculate']['medium']:
            df_features = self._calculate_medium_term_features(df_features, feature_config)

        if feature_config['can_calculate']['standard']:
            df_features = self._calculate_standard_features(df_features, feature_config)

        if 'spy_close' in df_features.columns:
            df_features = self._calculate_spy_correlation_features(df_features, feature_config)

        if feature_config['can_calculate']['extended']:
            df_features = self._calculate_extended_features(df_features, feature_config)

        if feature_config['can_calculate']['long']:
            df_features = self._calculate_long_term_features(df_features, feature_config)

        # Calculate volatility categories
        df_features = self._calculate_volatility_categories(df_features, feature_config)

        # Add sentiment features if available
        df_features = self._calculate_sentiment_features(df_features)

        # Clean data and handle NaN values
        df_features = self._clean_and_handle_nan(df_features, feature_config)

        return df_features

    def _calculate_basic_features(self, df):
        """Calculate basic price and candle features that don't depend on window sizes."""
        # Preis- und Kerzen-Features
        df['hl_ratio'] = df['high_price'] / df['low_price']
        df['co_ratio'] = df['close_price'] / df['open_price']
        df['candle_body'] = df['close_price'] - df['open_price']
        df['upper_shadow'] = df['high_price'] - np.maximum(df['close_price'], df['open_price'])
        df['lower_shadow'] = np.minimum(df['close_price'], df['open_price']) - df['low_price']
        df['is_bullish'] = (df['close_price'] > df['open_price']).astype(int)
        df['is_doji'] = (abs(df['candle_body']) < 0.1 * (df['high_price'] - df['low_price'])).astype(int)

        # Heikin-Ashi Kerzen - reduzieren Rauschen und zeigen Trends deutlicher
        try:
            # Heikin-Ashi Berechnung
            ha_close = (df['open_price'] + df['high_price'] + df['low_price'] + df['close_price']) / 4

            # Ersten Wert für HA Open setzen
            ha_open = df['open_price'].copy()

            # Iterativ berechnen (erfordert Schleife)
            for i in range(1, len(df)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

            # HA High und Low
            ha_high = df[['high_price', 'open_price', 'close_price']].max(axis=1)
            ha_low = df[['low_price', 'open_price', 'close_price']].min(axis=1)

            # Als Features speichern
            df['ha_open'] = ha_open
            df['ha_close'] = ha_close
            df['ha_high'] = ha_high
            df['ha_low'] = ha_low

            # Heikin-Ashi Kerzen-Features
            df['ha_body'] = ha_close - ha_open
            df['ha_is_bullish'] = (ha_close > ha_open).astype(int)

            # Trendstärke basierend auf Heikin-Ashi
            # Mehrere aufeinanderfolgende gleichfarbige Kerzen deuten auf starken Trend hin
            df['ha_trend_strength'] = df['ha_is_bullish'].rolling(window=3).sum()
            df.loc[df['ha_trend_strength'] == 0, 'ha_trend_strength'] = -3  # Alle 3 bearish

            # Trendwechsel-Signale
            df['ha_trend_change'] = df['ha_is_bullish'].diff().fillna(0)
            df['ha_bullish_reversal'] = (df['ha_trend_change'] > 0).astype(int)
            df['ha_bearish_reversal'] = (df['ha_trend_change'] < 0).astype(int)

        except Exception as e:
            print(f"Fehler bei Heikin-Ashi-Berechnung: {str(e)}")

        # Kurzfristige Renditen - maximal 1-Tages-Lag
        df['daily_return'] = df['close_price'].pct_change(fill_method=None).fillna(0)

        return df

    def _calculate_short_term_features(self, df, config):
        """Calculate short-term features that require a small amount of data."""
        SHORT_WINDOW = config['windows']['short']
        data_length = config['data_length']

        # Moving average
        df[f'ma_{SHORT_WINDOW}'] = df['close_price'].rolling(window=SHORT_WINDOW, min_periods=3).mean()
        if 'ma_5' in df.columns:  # Sicherheitsprüfung
            df[f'ma_{SHORT_WINDOW}_dist'] = (df['close_price'] - df[f'ma_{SHORT_WINDOW}']) / df[f'ma_{SHORT_WINDOW}'].replace(0, np.nan)

        # Volatility and volume
        df['volatility_5'] = df['daily_return'].rolling(window=SHORT_WINDOW, min_periods=3).std()
        df['volume_ma_5'] = df['volume'].rolling(window=SHORT_WINDOW, min_periods=3).mean()

        # EMA für MACD (kurze Komponente)
        df['ema_12'] = df['close_price'].ewm(span=12, min_periods=5, adjust=False).mean()

        # Kurzfristige Momentum-Features
        df[f'momentum_{SHORT_WINDOW}'] = df['close_price'] / df['close_price'].shift(SHORT_WINDOW).replace(0, np.nan) - 1

        # RSI mit kleinerem Fenster
        delta = df['close_price'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(com=SHORT_WINDOW, min_periods=3, adjust=False).mean()
        ema_down = down.ewm(com=SHORT_WINDOW, min_periods=3, adjust=False).mean()
        rs = ema_up / (ema_down.replace(0, np.nan))  # Vermeidung von Division durch Null
        df['rsi'] = 100 - (100 / (1 + rs))

        # Close-Lags für kurzfristige Vergleiche
        df['close_lag_1'] = df['close_price'].shift(1)

        # Preis-Geschwindigkeit über kurze Perioden
        df['price_velocity_3'] = df['close_price'].diff(min(3, data_length - 1)) / min(3, data_length - 1)

        return df

    def _calculate_medium_term_features(self, df, config):
        """Calculate medium-term features that require a moderate amount of data."""
        MEDIUM_WINDOW = config['windows']['medium']
        data_length = config['data_length']

        # Moving average
        df[f'ma_{MEDIUM_WINDOW}'] = df['close_price'].rolling(window=MEDIUM_WINDOW, min_periods=5).mean()
        if f'ma_{MEDIUM_WINDOW}' in df.columns:  # Sicherheitsprüfung
            df[f'ma_{MEDIUM_WINDOW}_dist'] = (df['close_price'] - df[f'ma_{MEDIUM_WINDOW}']) / df[f'ma_{MEDIUM_WINDOW}'].replace(0, np.nan)

        # Mittelfristige Renditen
        df['weekly_return'] = df['close_price'].pct_change(min(5, data_length - 1), fill_method=None).fillna(0)

        # Mittelfristige Momentum-Features
        df[f'momentum_{MEDIUM_WINDOW}'] = df['close_price'] / df['close_price'].shift(MEDIUM_WINDOW).replace(0, np.nan) - 1

        # MACD-Komponenten (wenn möglich)
        if 'ema_12' in df.columns:
            df['ema_26'] = df['close_price'].ewm(span=26, min_periods=10, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=4, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # SMA-Kreuzungen (falls möglich)
        if 'ma_5' in df.columns and f'ma_{MEDIUM_WINDOW}' in df.columns:
            df['ma_5_10_cross'] = df['ma_5'] - df[f'ma_{MEDIUM_WINDOW}']

        # Weitere Close-Lags
        df['close_lag_2'] = df['close_price'].shift(2)
        if 'rsi' in df.columns:
            df['rsi_lag_1'] = df['rsi'].shift(1)

        # Mittelfristige Trendstärke
        df['trend_strength_10'] = df['close_price'].diff(min(MEDIUM_WINDOW, data_length - 1))

        # Beschleunigung der Rendite
        if 'daily_return' in df.columns:
            df['return_acceleration'] = df['daily_return'].diff()

        return df

    def _calculate_standard_features(self, df, config):
        """Calculate standard features that require a moderate amount of data."""
        STANDARD_WINDOW = config['windows']['standard']
        MEDIUM_WINDOW = config['windows']['medium']
        data_length = config['data_length']
        can_calculate_medium = config['can_calculate']['medium']

        # Moving average
        df[f'ma_{STANDARD_WINDOW}'] = df['close_price'].rolling(window=STANDARD_WINDOW, min_periods=10).mean()
        if f'ma_{STANDARD_WINDOW}' in df.columns:  # Sicherheitsprüfung
            df[f'ma_{STANDARD_WINDOW}_dist'] = (df['close_price'] - df[f'ma_{STANDARD_WINDOW}']) / df[f'ma_{STANDARD_WINDOW}'].replace(0, np.nan)

        # Längerfristige Renditen
        df['monthly_return'] = df['close_price'].pct_change(min(STANDARD_WINDOW, data_length - 1), fill_method=None).fillna(0)

        # Volatilitätsfeatures
        df['volatility_20'] = df['daily_return'].rolling(window=STANDARD_WINDOW, min_periods=10).std()

        # Volume-Features
        df['volume_ma_20'] = df['volume'].rolling(window=STANDARD_WINDOW, min_periods=10).mean()
        if 'volume_ma_20' in df.columns and df['volume_ma_20'].max() > 0:
            df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, np.nan)

        # Bollinger Bands
        if f'ma_{STANDARD_WINDOW}' in df.columns:
            std = df['close_price'].rolling(window=STANDARD_WINDOW, min_periods=10).std()
            df['bb_middle'] = df[f'ma_{STANDARD_WINDOW}']
            df['bb_upper'] = df['bb_middle'] + 2 * std
            df['bb_lower'] = df['bb_middle'] - 2 * std

            # Vermeidung von Division durch Null
            bb_width_divisor = df['bb_middle'].replace(0, np.nan)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_width_divisor

            # BB Position nur berechnen, wenn nicht durch Null geteilt wird
            bb_denominator = (df['bb_upper'] - df['bb_lower'])
            bb_position_valid = bb_denominator > 0
            df['bb_position'] = np.nan  # Default-Wert
            if bb_position_valid.any():
                df.loc[bb_position_valid, 'bb_position'] = (
                        (df.loc[bb_position_valid, 'close_price'] -
                         df.loc[bb_position_valid, 'bb_lower']) /
                        bb_denominator[bb_position_valid]
                )
            # Fallback für ungültige Positionen
            df['bb_position'] = df['bb_position'].fillna(0.5)

        # Z-Score
        if f'ma_{STANDARD_WINDOW}' in df.columns:
            std20 = df['close_price'].rolling(window=STANDARD_WINDOW, min_periods=10).std()
            std20_valid = std20 > 0
            df['zscore_20'] = np.nan
            if std20_valid.any():
                df.loc[std20_valid, 'zscore_20'] = (
                        (df.loc[std20_valid, 'close_price'] -
                         df.loc[std20_valid, f'ma_{STANDARD_WINDOW}']) /
                        std20[std20_valid]
                )

        # Standardfeatures für längerfristige Momentum-Features
        df[f'momentum_{STANDARD_WINDOW}'] = df['close_price'] / df['close_price'].shift(STANDARD_WINDOW).replace(0, np.nan) - 1

        # Bullish Signal Count basierend auf verfügbaren Indikatoren
        bullish_indicators = []
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            bullish_indicators.append((df['macd'] > df['macd_signal']).astype(int))
        if 'rsi' in df.columns:
            bullish_indicators.append((df['rsi'] < 30).astype(int))
        if f'ma_{STANDARD_WINDOW}' in df.columns:
            bullish_indicators.append((df['close_price'] > df[f'ma_{STANDARD_WINDOW}']).astype(int))

        if bullish_indicators:
            df['bullish_signals'] = sum(bullish_indicators)

        # Close Lag 3 für längere Trends
        df['close_lag_3'] = df['close_price'].shift(3)

        # MACD Lags für Trendveränderungen
        if 'macd' in df.columns:
            df['macd_lag_1'] = df['macd'].shift(1)

        # SMA-Kreuzungen und Veränderungen
        if f'ma_{STANDARD_WINDOW}' in df.columns and can_calculate_medium and f'ma_{MEDIUM_WINDOW}' in df.columns:
            df[f'ma_{MEDIUM_WINDOW}_{STANDARD_WINDOW}_cross'] = df[f'ma_{MEDIUM_WINDOW}'] - df[f'ma_{STANDARD_WINDOW}']

        # Bullish Engulfing Muster
        if 'is_bullish' in df.columns and 'candle_body' in df.columns:
            df['is_bullish_engulfing'] = ((df['is_bullish'] == 1) &
                                           (df['candle_body'] > df['candle_body'].shift())).astype(int)

        # ADX (Average Directional Index) - Trendstärke-Indikator
        try:
            # Berechnung der Richtungsbewegungen
            plus_dm = df['high_price'].diff()
            minus_dm = df['low_price'].diff(-1).abs()

            # Positive/Negative Richtungsbewegung
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

            # True Range
            high_low = df['high_price'] - df['low_price']
            high_close = (df['high_price'] - df['close_price'].shift()).abs()
            low_close = (df['low_price'] - df['close_price'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Smoothed Werte (14-Perioden)
            window = min(14, data_length - 5)
            if window >= 5:  # Mindestens 5 Perioden für sinnvolle Berechnung
                smoothed_tr = tr.rolling(window=window, min_periods=5).sum()
                smoothed_plus_dm = plus_dm.rolling(window=window, min_periods=5).sum()
                smoothed_minus_dm = minus_dm.rolling(window=window, min_periods=5).sum()

                # Direktionale Indikatoren
                plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
                minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

                # Direktionaler Index
                di_diff = (plus_di - minus_di).abs()
                di_sum = plus_di + minus_di
                dx = 100 * di_diff / di_sum.replace(0, np.nan)

                # ADX (Average Directional Index) - geglätteter DX
                df['adx'] = dx.rolling(window=window, min_periods=5).mean()
                df['+di'] = plus_di
                df['-di'] = minus_di

                # ADX Interpretation als Features
                df['strong_trend'] = (df['adx'] > 25).astype(int)
                df['very_strong_trend'] = (df['adx'] > 50).astype(int)
                df['bullish_trend'] = ((df['+di'] > df['-di']) & (df['adx'] > 20)).astype(int)
                df['bearish_trend'] = ((df['+di'] < df['-di']) & (df['adx'] > 20)).astype(int)
        except Exception as e:
            print(f"Fehler bei ADX-Berechnung: {str(e)}")

        return df

    def _calculate_spy_correlation_features(self, df, config):
        """Calculate features related to SPY correlation and market beta."""
        MEDIUM_WINDOW = config['windows']['medium']
        STANDARD_WINDOW = config['windows']['standard']
        data_length = config['data_length']
        can_calculate_medium = config['can_calculate']['medium']
        can_calculate_standard = config['can_calculate']['standard']

        # Grundlegende SPY-Renditen berechnen
        df['spy_daily_return'] = df['spy_close'].pct_change(fill_method=None).fillna(0)

        # Kurzfristige relative Stärke
        if 'daily_return' in df.columns:
            df['rel_strength_daily'] = df['daily_return'] - df['spy_daily_return']

        # Medium-Term SPY Features
        if can_calculate_medium:
            df['spy_return_10d'] = df['spy_close'].pct_change(MEDIUM_WINDOW, fill_method=None).fillna(0)
            if 'weekly_return' in df.columns:
                df['rel_strength_10d'] = df['weekly_return'] - df['spy_return_10d']

        # Standard SPY Features
        if can_calculate_standard:
            # 20-Tage Korrelation - kritisch, dass genügend Datenpunkte vorhanden sind
            min_corr_periods = min(15, data_length - 5)
            if min_corr_periods >= 10:  # Mindestens 10 Punkte für sinnvolle Korrelation
                df['corr_with_spy_20d'] = df['close_price'].rolling(STANDARD_WINDOW,
                                                                    min_periods=min_corr_periods).corr(
                    df['spy_close'])

            # SPY und Stock Rolling Return
            min_periods_spy = min(15, data_length - 5)
            if can_calculate_standard and min_periods_spy >= 10:
                spy_ret_window = min(30, data_length - 5)
                df['spy_return_30d'] = df['spy_close'].pct_change(fill_method=None).fillna(0).rolling(window=spy_ret_window,
                                                                                          min_periods=min_periods_spy).sum()
                df['stock_return_30d'] = df['close_price'].pct_change(fill_method=None).fillna(0).rolling(
                    window=spy_ret_window, min_periods=min_periods_spy).sum()

                # Alpha und Beta nur berechnen, wenn genügend nicht-NaN Werte vorhanden sind
                df['rolling_alpha'] = np.nan
                df['rolling_beta'] = np.nan

                valid_indices = ~(df['spy_return_30d'].isna() | df['stock_return_30d'].isna())

                # Nur Alpha/Beta für einzelne Datenpunkte berechnen, wenn genug Vergangenheitsdaten verfügbar sind
                if valid_indices.sum() >= spy_ret_window:
                    window = min(30, valid_indices.sum() - 5)

                    for i in range(window, len(df)):
                        # Prüfe, ob genügend Daten für eine Regression vorhanden sind
                        valid_window = ~(df['spy_return_30d'].iloc[i - window:i].isna() |
                                         df['stock_return_30d'].iloc[i - window:i].isna())

                        if valid_window.sum() >= 10:  # Mindestens 10 Punkte für Regression
                            x = df['spy_return_30d'].iloc[i - window:i][valid_window].values.reshape(-1, 1)
                            y = df['stock_return_30d'].iloc[i - window:i][valid_window].values

                            try:
                                model = LinearRegression()
                                model.fit(x, y)
                                df.loc[df.index[i], 'rolling_alpha'] = model.intercept_
                                df.loc[df.index[i], 'rolling_beta'] = model.coef_[0]
                            except:
                                # Fallback bei Fehlern in der Regression
                                pass

        return df

    def _calculate_extended_features(self, df, config):
        """Calculate extended features that require a larger amount of data."""
        EXTENDED_WINDOW = config['windows']['extended']
        STANDARD_WINDOW = config['windows']['standard']
        data_length = config['data_length']

        # Extended moving average
        df[f'ma_{EXTENDED_WINDOW}'] = df['close_price'].rolling(window=EXTENDED_WINDOW, min_periods=30).mean()

        # SMA Kreuzungen mit Extended Windows
        if f'ma_{STANDARD_WINDOW}' in df.columns and f'ma_{EXTENDED_WINDOW}' in df.columns:
            df[f'ma_{STANDARD_WINDOW}_{EXTENDED_WINDOW}_cross'] = np.sign(df[f'ma_{STANDARD_WINDOW}'] - df[f'ma_{EXTENDED_WINDOW}'])
            df['sma_cross_change'] = df[f'ma_{STANDARD_WINDOW}_{EXTENDED_WINDOW}_cross'].diff().fillna(0)
            df['sma_bullish_cross'] = (df['sma_cross_change'] > 0).astype(int)
            df['sma_bearish_cross'] = (df['sma_cross_change'] < 0).astype(int)

        # EMA Kreuzungen
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_12_26_cross'] = np.sign(df['ema_12'] - df['ema_26'])
            df['ema_cross_change'] = df['ema_12_26_cross'].diff().fillna(0)
            df['ema_bullish_cross'] = (df['ema_cross_change'] > 0).astype(int)
            df['ema_bearish_cross'] = (df['ema_cross_change'] < 0).astype(int)

        # Ichimoku Cloud - umfassendes Indikator-System
        try:
            # Parameter für Ichimoku (angepasst an verfügbare Datenmenge)
            tenkan_period = min(9, data_length // 5)  # Conversion Line
            kijun_period = min(26, data_length // 3)  # Base Line
            senkou_b_period = min(52, data_length // 2)  # Leading Span B

            if tenkan_period >= 3 and kijun_period >= 5 and senkou_b_period >= 10:
                # Tenkan-sen (Conversion Line): (n-period high + n-period low) / 2
                period_high = df['high_price'].rolling(window=tenkan_period).max()
                period_low = df['low_price'].rolling(window=tenkan_period).min()
                df['tenkan_sen'] = (period_high + period_low) / 2

                # Kijun-sen (Base Line): (n-period high + n-period low) / 2
                period_high = df['high_price'].rolling(window=kijun_period).max()
                period_low = df['low_price'].rolling(window=kijun_period).min()
                df['kijun_sen'] = (period_high + period_low) / 2

                # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
                df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2)

                # Senkou Span B (Leading Span B): (n-period high + n-period low) / 2
                period_high = df['high_price'].rolling(window=senkou_b_period).max()
                period_low = df['low_price'].rolling(window=senkou_b_period).min()
                df['senkou_span_b'] = (period_high + period_low) / 2

                # Chikou Span (Lagging Span): Aktuelle Schlusskurse
                df['chikou_span'] = df['close_price']

                # Ichimoku-Signale als Features
                # Preis über der Cloud (bullish)
                df['price_above_cloud'] = (df['close_price'] > df['senkou_span_a']) & \
                                          (df['close_price'] > df['senkou_span_b'])
                df['price_above_cloud'] = df['price_above_cloud'].astype(int)

                # Preis unter der Cloud (bearish)
                df['price_below_cloud'] = (df['close_price'] < df['senkou_span_a']) & \
                                          (df['close_price'] < df['senkou_span_b'])
                df['price_below_cloud'] = df['price_below_cloud'].astype(int)

                # Collect all new columns in a dictionary
                new_columns = {
                    'price_in_cloud': (~(df['price_above_cloud'] | df['price_below_cloud'])).astype(int),

                    # TK Cross (Tenkan kreuzt Kijun) - starkes Signal
                    'tk_cross': np.sign(df['tenkan_sen'] - df['kijun_sen']),
                }

                # Add tk_cross_change after tk_cross is calculated
                new_columns['tk_cross_change'] = new_columns['tk_cross'].diff().fillna(0)
                new_columns['bullish_tk_cross'] = (new_columns['tk_cross_change'] > 0).astype(int)
                new_columns['bearish_tk_cross'] = (new_columns['tk_cross_change'] < 0).astype(int)

                # Kijun Cross (Preis kreuzt Kijun) - wichtiges Signal
                new_columns['kijun_cross'] = np.sign(df['close_price'] - df['kijun_sen'])
                new_columns['kijun_cross_change'] = new_columns['kijun_cross'].diff().fillna(0)
                new_columns['bullish_kijun_cross'] = (new_columns['kijun_cross_change'] > 0).astype(int)
                new_columns['bearish_kijun_cross'] = (new_columns['kijun_cross_change'] < 0).astype(int)

                # Cloud-Twist (Senkou Span A kreuzt Senkou Span B) - langfristiges Signal
                new_columns['cloud_twist'] = np.sign(df['senkou_span_a'] - df['senkou_span_b'])
                new_columns['cloud_twist_change'] = new_columns['cloud_twist'].diff().fillna(0)
                new_columns['bullish_cloud_twist'] = (new_columns['cloud_twist_change'] > 0).astype(int)
                new_columns['bearish_cloud_twist'] = (new_columns['cloud_twist_change'] < 0).astype(int)

                # Add all new columns at once
                df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        except Exception as e:
            print(f"Fehler bei Ichimoku-Berechnung: {str(e)}")

        return df

    def _calculate_long_term_features(self, df, config):
        """Calculate long-term features that require a very large amount of data."""
        LONG_WINDOW = config['windows']['long']
        EXTENDED_WINDOW = config['windows']['extended']

        # Collect long-term features in a dictionary
        long_term_columns = {
            f'ma_{LONG_WINDOW}': df['close_price'].rolling(window=LONG_WINDOW, min_periods=120).mean()
        }

        # SMA Kreuzungen mit Long Windows
        if f'ma_{EXTENDED_WINDOW}' in df.columns and f'ma_{LONG_WINDOW}' in df.columns:
            long_term_columns[f'ma_{EXTENDED_WINDOW}_{LONG_WINDOW}_cross'] = df[f'ma_{EXTENDED_WINDOW}'] - df[f'ma_{LONG_WINDOW}']

        # Add all long-term columns at once
        df = pd.concat([df, pd.DataFrame(long_term_columns, index=df.index)], axis=1)

        return df

    def _calculate_volatility_categories(self, df, config):
        """Calculate volatility categories and flags."""
        # Volatilitäts-Kategorisierung (nur wenn vorhanden)
        new_columns = {}

        if 'volatility_20' in df.columns and df['volatility_20'].count() >= 10:
            try:
                # Quantile-Berechnung könnte fehlschlagen bei zu wenigen Werten
                new_columns['volatility_category'] = pd.qcut(
                    df['volatility_20'].rank(method='first'),  # Ranking für gleiche Werte
                    q=3,
                    labels=[0, 1, 2]
                ).astype(int)
            except:
                # Fallback: Manuelle Kategorisierung
                if df['volatility_20'].max() > 0:
                    thresholds = [
                        df['volatility_20'].quantile(0.33, interpolation='nearest'),
                        df['volatility_20'].quantile(0.67, interpolation='nearest')
                    ]

                    # Create a Series with default value 1
                    volatility_category = pd.Series(1, index=df.index)
                    # Update values based on thresholds
                    volatility_category[df['volatility_20'] <= thresholds[0]] = 0
                    volatility_category[df['volatility_20'] > thresholds[1]] = 2
                    new_columns['volatility_category'] = volatility_category

        # Bei High Volatility Feature, ein einfacheres Fallback verwenden
        if 'volatility_20' in df.columns and df['volatility_20'].count() >= 5:
            # Verwende Median statt Quantil bei wenigen Datenpunkten
            vol_thresh = df['volatility_20'].median() * 1.5
            new_columns['high_volatility_flag'] = (df['volatility_20'] > vol_thresh).astype(int)

        # Add all new columns at once if there are any
        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

        return df

    def _calculate_sentiment_features(self, df):
        """
        Calculate sentiment-based features using the SentimentAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame with stock data

        Returns:
            pd.DataFrame: DataFrame with added sentiment features
        """
        try:
            from .sentiment_analyzer import SentimentAnalyzer

            # Create a SentimentAnalyzer instance
            analyzer = SentimentAnalyzer(self.stock_symbol)

            # Check if we have already calculated sentiment features in this session
            if hasattr(self, '_cached_sentiment_features') and self._cached_sentiment_features is not None:
                logger.info(f"Using cached sentiment features for {self.stock_symbol}")
                sentiment_features = self._cached_sentiment_features
            else:
                # Get sentiment features
                sentiment_features = analyzer.create_sentiment_features()
                # Cache the sentiment features for future use
                self._cached_sentiment_features = sentiment_features

            if sentiment_features is not None and not sentiment_features.empty:
                logger.info(f"Adding sentiment features for {self.stock_symbol}")

                # Merge sentiment features with stock data
                df_with_dates = df.copy()
                if 'date' not in df_with_dates.columns and df_with_dates.index.name != 'date':
                    # If date is not a column or index, try to get it from the database
                    stock = Stock.objects.get(symbol=self.stock_symbol)
                    dates = StockData.objects.filter(stock=stock).order_by('date').values_list('date', flat=True)
                    if len(dates) == len(df_with_dates):
                        df_with_dates['date'] = dates
                    else:
                        logger.warning(f"Date mismatch for {self.stock_symbol}, cannot add sentiment features")
                        return df

                # Ensure date is in the right format for merging
                if 'date' in df_with_dates.columns:
                    df_with_dates['date'] = pd.to_datetime(df_with_dates['date'])
                    sentiment_features['date'] = pd.to_datetime(sentiment_features['date'])

                    # Merge with stock data
                    merged_df = pd.merge(df_with_dates, sentiment_features, on='date', how='left')

                    # Fill missing sentiment values with neutral values
                    for col in sentiment_features.columns:
                        if col != 'date' and col in merged_df.columns:
                            if col.startswith('sentiment_compound'):
                                merged_df[col] = merged_df[col].fillna(0)
                            elif col.startswith('sentiment_positive'):
                                merged_df[col] = merged_df[col].fillna(0.33)
                            elif col.startswith('sentiment_negative'):
                                merged_df[col] = merged_df[col].fillna(0.33)
                            elif col.startswith('sentiment_neutral'):
                                merged_df[col] = merged_df[col].fillna(0.34)
                            elif col.startswith('sentiment_volume'):
                                merged_df[col] = merged_df[col].fillna(0)
                            elif col.startswith('sentiment_momentum'):
                                merged_df[col] = merged_df[col].fillna(0)
                            elif col == 'sentiment_regime':
                                merged_df[col] = merged_df[col].fillna('neutral')
                            elif col == 'sentiment_price_divergence':
                                merged_df[col] = merged_df[col].fillna(0)
                            else:
                                merged_df[col] = merged_df[col].fillna(0)

                    # Create a visualization of sentiment impact on predictions
                    self._visualize_sentiment_impact(sentiment_features)

                    return merged_df
                else:
                    logger.warning(f"No date column found for {self.stock_symbol}, cannot add sentiment features")
            else:
                logger.warning(f"No sentiment features available for {self.stock_symbol}")
        except Exception as e:
            logger.error(f"Error calculating sentiment features for {self.stock_symbol}: {str(e)}")

        return df

    def _visualize_sentiment_impact(self, sentiment_df):
        """
        Create a visualization of sentiment impact on stock price.

        Args:
            sentiment_df (pd.DataFrame): DataFrame with sentiment features

        Returns:
            str: Path to the saved visualization or None if not saved
        """
        try:
            if sentiment_df is None or sentiment_df.empty:
                return None

            # Get stock data
            stock = Stock.objects.get(symbol=self.stock_symbol)
            stock_data = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            if stock_data.empty:
                return None

            # Convert date to datetime
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

            # Merge with stock data
            merged_df = pd.merge(stock_data, sentiment_df, on='date', how='inner')

            if merged_df.empty:
                return None

            # Convert price columns to float
            for col in ['open_price', 'high_price', 'low_price', 'close_price']:
                if col in merged_df.columns:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

            # Create a figure with multiple subplots
            fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

            # Plot stock price
            axs[0].plot(merged_df['date'], merged_df['close_price'], 'k-', label='Close Price')
            axs[0].set_title(f"Sentiment Impact on {self.stock_symbol}", fontsize=16)
            axs[0].set_ylabel('Price', fontsize=12)
            axs[0].grid(True, alpha=0.3)

            # Create a twin axis for sentiment
            ax1b = axs[0].twinx()
            ax1b.plot(merged_df['date'], merged_df['sentiment_compound'], 'b-', alpha=0.7, label='Sentiment')
            ax1b.fill_between(merged_df['date'], merged_df['sentiment_compound'], 0, 
                             where=(merged_df['sentiment_compound'] >= 0), 
                             color='green', alpha=0.3)
            ax1b.fill_between(merged_df['date'], merged_df['sentiment_compound'], 0, 
                             where=(merged_df['sentiment_compound'] < 0), 
                             color='red', alpha=0.3)
            ax1b.set_ylabel('Sentiment Score', fontsize=12)
            ax1b.set_ylim(-1, 1)

            # Combine legends
            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Plot sentiment-price divergence
            if 'sentiment_price_divergence' in merged_df.columns:
                axs[1].bar(merged_df['date'], merged_df['sentiment_price_divergence'], color='purple', alpha=0.7)
                axs[1].set_title("Sentiment-Price Divergence", fontsize=14)
                axs[1].set_ylabel('Divergence', fontsize=12)
                axs[1].set_xlabel('Date', fontsize=12)
                axs[1].grid(True, alpha=0.3)
            elif 'sentiment_momentum_1d' in merged_df.columns:
                axs[1].plot(merged_df['date'], merged_df['sentiment_momentum_1d'], 'g-', label='Sentiment Momentum')
                axs[1].set_title("Sentiment Momentum", fontsize=14)
                axs[1].set_ylabel('Momentum', fontsize=12)
                axs[1].set_xlabel('Date', fontsize=12)
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()

            # Format dates on x-axis
            for ax in axs:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            # Create directory if it doesn't exist
            sentiment_dir = os.path.join('static', 'images', 'sentiment')
            os.makedirs(sentiment_dir, exist_ok=True)

            # Save the figure
            save_path = os.path.join(sentiment_dir, f'{self.stock_symbol}_sentiment_impact.png')
            plt.savefig(save_path)
            plt.close(fig)

            logger.info(f"Saved sentiment impact visualization for {self.stock_symbol}")

            # Return the web path to the visualization
            return f'/static/images/sentiment/{self.stock_symbol}_sentiment_impact.png'

        except Exception as e:
            logger.error(f"Error visualizing sentiment impact for {self.stock_symbol}: {str(e)}")
            return None

    def _get_sentiment_info(self):
        """
        Get sentiment information for the stock.

        Returns:
            dict: Dictionary with sentiment information
        """
        try:
            from .sentiment_analyzer import SentimentAnalyzer

            # Create a SentimentAnalyzer instance
            analyzer = SentimentAnalyzer(self.stock_symbol)

            # Analyze sentiment
            sentiment_data = analyzer.analyze_sentiment()

            if sentiment_data is None:
                logger.warning(f"No sentiment data available for {self.stock_symbol}")
                return None

            # Check if we have already calculated sentiment features in this session
            if hasattr(self, '_cached_sentiment_features') and self._cached_sentiment_features is not None:
                logger.info(f"Using cached sentiment features for {self.stock_symbol}")
                sentiment_features = self._cached_sentiment_features
            else:
                # Get sentiment features
                sentiment_features = analyzer.create_sentiment_features()
                # Cache the sentiment features for future use
                self._cached_sentiment_features = sentiment_features

            # Create visualization
            visualization_path = self._visualize_sentiment_impact(sentiment_features)

            # Get the most recent sentiment
            combined_sentiment = sentiment_data['combined']
            dates = sorted(combined_sentiment.keys())

            if not dates:
                logger.warning(f"No sentiment dates available for {self.stock_symbol}")
                return None

            latest_date = dates[-1]
            current_sentiment = combined_sentiment[latest_date]

            # Create sentiment info dictionary
            sentiment_info = {
                'current_sentiment': {
                    'compound': float(current_sentiment['compound']),
                    'positive': float(current_sentiment['positive']),
                    'negative': float(current_sentiment['negative']),
                    'neutral': float(current_sentiment['neutral']),
                    'sentiment': current_sentiment['sentiment'],
                    'volume': int(current_sentiment['volume']),
                    'date': latest_date
                },
                'visualization_path': visualization_path
            }

            # Add top positive and negative items if available
            if 'news' in sentiment_data and 'sentiment' in sentiment_data['news']:
                # Sort by compound score
                positive_items = sorted([item for item in sentiment_data['news']['sentiment'] if item['compound'] > 0], 
                                      key=lambda x: x['compound'], reverse=True)[:3]
                negative_items = sorted([item for item in sentiment_data['news']['sentiment'] if item['compound'] < 0], 
                                      key=lambda x: x['compound'])[:3]

                sentiment_info['top_positive'] = positive_items
                sentiment_info['top_negative'] = negative_items

            return sentiment_info

        except Exception as e:
            logger.error(f"Error getting sentiment info for {self.stock_symbol}: {str(e)}")
            return None

    def _clean_and_handle_nan(self, df, config):
        """Clean data and handle NaN values."""
        # Bereinigen der Daten
        df = df.replace([np.inf, -np.inf], np.nan)

        # Bestimme wichtige Feature-Spalten die komplett sein müssen (keine NaN erlaubt)
        critical_features = ['close_price', 'open_price', 'high_price', 'low_price']

        # Wichtige Features, die berechnet werden sollten, aber NaN haben dürfen
        useful_features = []

        if config['can_calculate']['short']:
            useful_features.extend(['daily_return', f'ma_{config["windows"]["short"]}', 'ema_12', 'rsi'])

        if config['can_calculate']['medium']:
            useful_features.extend(['weekly_return', f'ma_{config["windows"]["medium"]}', 'macd', 'macd_signal'])

        if config['can_calculate']['standard']:
            useful_features.extend(['monthly_return', f'ma_{config["windows"]["standard"]}', 'bb_middle', 'bb_upper', 'bb_lower'])

        # Prüfe, ob kritische Features NaN-Werte enthalten
        missing_critical = [col for col in critical_features if
                            col in df.columns and df[col].isna().any()]
        if missing_critical:
            print(f"DEBUG: Kritische Features mit NaN: {missing_critical}")

        # NaN-Werte in nicht-kritischen Features mit Durchschnittswerten füllen
        for col in df.columns:
            if col not in critical_features and df[col].isna().any():
                # Für Features mit signifikanten NaN-Werten, aber ausreichend Nicht-NaN-Werten
                non_nan_count = df[col].count()
                if non_nan_count >= 5 and non_nan_count >= len(df) * 0.3:
                    df[col] = df[col].fillna(df[col].mean())

        # Fallback: Wenn nach all diesen Berechnungen immer noch alle Zeilen NaN-Werte enthalten
        original_len = len(df)
        df_no_nan = df.dropna()

        if len(df_no_nan) == 0:
            print(f"DEBUG: Nach allen Berechnungen immer noch keine vollständigen Zeilen! Führe Minimal-Fallback aus.")

            # Nur minimale kritische Features behalten
            min_features = critical_features.copy()
            if 'daily_return' in df.columns:
                min_features.append('daily_return')

            # Auch nützliche Features hinzufügen, die berechnet wurden
            available_useful = []
            for f in useful_features:
                if f in df.columns:
                    non_nan_ratio = df[f].count() / len(df)
                    if non_nan_ratio > 0.8:  # If at least 80% of values are not NaN
                        available_useful.append(f)

            min_features.extend(available_useful)

            # Reduziertes DataFrame erstellen
            available_min_features = [col for col in min_features if col in df.columns]
            if not available_min_features:
                # If no features are available, at least keep the price columns
                available_min_features = [col for col in ['open_price', 'high_price', 'low_price', 'close_price']
                                          if col in df.columns]

            df_minimal = df[available_min_features].copy()

            # Verbleibende NaN-Werte auffüllen
            for col in df_minimal.columns:
                # Check if column contains any NaN values before attempting to fill
                if df_minimal[col].isna().any():
                    # Calculate mean safely
                    non_nan_values = df_minimal[col].dropna()
                    if len(non_nan_values) > 0:
                        mean_value = non_nan_values.mean()
                        # Use proper check for scalar value
                        if not pd.isna(mean_value):
                            df_minimal[col] = df_minimal[col].fillna(mean_value)
                        else:
                            df_minimal[col] = df_minimal[col].fillna(0)
                    else:
                        df_minimal[col] = df_minimal[col].fillna(0)

            feature_count = len(df_minimal.columns)
            print(f"DEBUG: Minimal-Feature-Set verwendet: {list(df_minimal.columns)}")
            print(f"DEBUG: _calculate_features returning minimal set with {feature_count} features for {self.stock_symbol}")
            print(f"DEBUG: WARNING - Using minimal feature set due to data quality issues")
            return df_minimal

        # Normale Rückgabe: Entferne NaN-Zeilen
        if len(df_no_nan) < original_len:
            print(f"DEBUG: {original_len - len(df_no_nan)} Zeilen mit NaN-Werten entfernt")

        # Log the number of features being returned
        feature_count = len(df_no_nan.columns)
        print(f"DEBUG: _calculate_features returning {feature_count} features for {self.stock_symbol}")
        if feature_count < 10:
            print(f"DEBUG: WARNING - Very few features calculated: {list(df_no_nan.columns)}")

        return df_no_nan

    def _load_or_train_model(self, model_type):
        """Load a saved model or train a new one if no saved model exists or if feature count has changed"""
        model_path = self.price_model_path if model_type == 'price' else self.signal_model_path
        feature_names_path = os.path.join(self.models_dir, f'{self.stock_symbol}_feature_names.pkl')

        try:
            # Try to load the model
            if os.path.exists(model_path):
                model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))

                # Retrain model if it's older than 30 days
                if model_age.days > 30:
                    logger.info(f"Model for {self.stock_symbol} is older than 30 days, retraining")
                    return self._train_model(model_type)

                # Load the model
                logger.info(f"Loading existing {model_type} model for {self.stock_symbol}")
                model = joblib.load(model_path)

                # Check if the model is compatible with the current feature set
                # First, get the current feature count
                X, _, _ = self.prepare_data()
                if X is None:
                    logger.error(f"Could not prepare data for {self.stock_symbol}, cannot check feature compatibility")
                    return model

                # Check if the model has n_features_in_ attribute (most sklearn models do)
                if hasattr(model, 'n_features_in_'):
                    expected_n_features = model.n_features_in_
                    current_n_features = X.shape[1]

                    if expected_n_features != current_n_features:
                        logger.warning(f"Feature count mismatch for {self.stock_symbol} {model_type} model: "
                                      f"Model expects {expected_n_features} features, but current data has {current_n_features}")

                        # If we have more features than the model expects, we can still use the model
                        # The predict method will handle the feature selection
                        if current_n_features > expected_n_features:
                            logger.info(f"Current data has more features than model expects. "
                                       f"The predict method will handle feature selection.")
                            return model
                        else:
                            # If we have fewer features than expected, we need to retrain
                            logger.warning(f"Current data has fewer features than model expects. Retraining model.")
                            return self._train_model(model_type)

                # Load feature names if available
                if os.path.exists(feature_names_path):
                    try:
                        self.feature_names = joblib.load(feature_names_path)
                        logger.info(f"Loaded feature names for {self.stock_symbol} {model_type} model")
                    except Exception as e:
                        logger.error(f"Error loading feature names for {self.stock_symbol} {model_type} model: {str(e)}")

                return model
            else:
                # Train a new model if none exists
                logger.info(f"No existing model found for {self.stock_symbol}, training new model")
                return self._train_model(model_type)

        except Exception as e:
            logger.error(f"Error loading {model_type} model for {self.stock_symbol}: {str(e)}")
            # Train a new model if there was an error loading
            return self._train_model(model_type)

    def select_model_based_on_characteristics(self, model_type):
        """
        Select the best model type based on stock characteristics and market regime

        Args:
            model_type (str): 'price' or 'signal'

        Returns:
            tuple: (base_model, param_grid, scoring)
        """
        try:
            # Get stock data for analysis
            stock = Stock.objects.get(symbol=self.stock_symbol)
            df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            # Calculate characteristics
            data_size = len(df)
            std_value = df['close_price'].pct_change().std()
            # Ensure std_value is a float before using with np.sqrt
            std_value = float(std_value)
            volatility = std_value * np.sqrt(252.0)  # Annualized volatility

            # Get trading volume characteristics
            avg_volume = df['volume'].mean()
            volume_volatility = df['volume'].std() / avg_volume if avg_volume > 0 else 0

            # Calculate trend strength using linear regression on prices
            df['day_index'] = range(len(df))
            if len(df) > 30:
                trend_model = LinearRegression()
                X_trend = df['day_index'].values.reshape(-1, 1)[-30:]
                y_trend = df['close_price'].values[-30:]
                trend_model.fit(X_trend, y_trend)
                trend_strength = abs(trend_model.coef_[0]) / df['close_price'].mean()
            else:
                trend_strength = 0

            # Calculate seasonality and cyclicality
            if len(df) > 60:  # Need enough data to detect patterns
                # Simple seasonality detection using autocorrelation
                price_series = df['close_price'].values
                autocorr = np.correlate(price_series, price_series, mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Take only the positive lags
                # Normalize
                autocorr = autocorr / autocorr[0]
                # Check for peaks in autocorrelation (excluding lag 0)
                peaks = []
                for i in range(2, min(len(autocorr)-1, 30)):  # Look for peaks in first 30 lags
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.2:
                        peaks.append((i, autocorr[i]))

                has_seasonality = len(peaks) > 0
                seasonality_strength = max([p[1] for p in peaks]) if peaks else 0
            else:
                has_seasonality = False
                seasonality_strength = 0

            # Detect market regime
            current_regime = self.detect_market_regime()
            regime_type = current_regime.get('regime_type', 'unknown') if current_regime else 'unknown'

            logger.info(f"Stock characteristics for {self.stock_symbol}: "
                       f"data_size={data_size}, volatility={volatility:.4f}, "
                       f"volume_volatility={volume_volatility:.4f}, trend_strength={trend_strength:.4f}, "
                       f"seasonality_strength={seasonality_strength:.4f}, "
                       f"market_regime={regime_type}")

            # Select regime-specific models
            if regime_type in ['volatile_bull', 'volatile_bear', 'volatile_sideways']:
                # For volatile regimes, use models that handle volatility well
                if model_type == 'price':
                    # For price prediction in volatile regimes, ensemble methods work well
                    base_model = VotingRegressor([
                        ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)),
                        ('ridge', Ridge(alpha=1.0)),
                        ('lr', LinearRegression())
                    ])
                    param_grid = {
                        'gbr__n_estimators': [50, 100, 200],
                        'gbr__learning_rate': [0.05, 0.1, 0.2],
                        'gbr__max_depth': [2, 3, 4],
                        'ridge__alpha': [0.5, 1.0, 2.0]
                    }
                    logger.info(f"Selected VotingRegressor for {self.stock_symbol} (volatile regime)")
                    return base_model, param_grid, 'neg_mean_squared_error'
                else:  # signal model
                    # For signal prediction in volatile regimes, use a robust classifier
                    base_model = VotingClassifier([
                        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5)),
                        ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)),
                        ('lr', LogisticRegression(C=1.0))
                    ], voting='soft')
                    param_grid = {
                        'rf__n_estimators': [50, 100, 200],
                        'rf__max_depth': [3, 5, 7],
                        'xgb__n_estimators': [50, 100, 200],
                        'xgb__learning_rate': [0.05, 0.1, 0.2],
                        'lr__C': [0.5, 1.0, 2.0]
                    }
                    logger.info(f"Selected VotingClassifier for {self.stock_symbol} (volatile regime)")
                    return base_model, param_grid, 'f1_weighted'

            elif regime_type in ['steady_bull', 'steady_bear']:
                # For steady trend regimes, use models that capture trends well
                if model_type == 'price':
                    # For price prediction in steady trend regimes, LSTM works well
                    base_model = LSTMModel(
                        sequence_length=10,
                        units=50,
                        dropout=0.2
                    )
                    param_grid = {
                        'sequence_length': [5, 10, 15],
                        'units': [30, 50, 70],
                        'dropout': [0.1, 0.2, 0.3]
                    }
                    logger.info(f"Selected LSTMModel for {self.stock_symbol} (steady trend regime)")
                    return base_model, param_grid, 'neg_mean_squared_error'
                else:  # signal model
                    # For signal prediction in steady trend regimes, gradient boosting works well
                    base_model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3
                    )
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.05, 0.1, 0.2],
                        'max_depth': [2, 3, 4]
                    }
                    logger.info(f"Selected XGBClassifier for {self.stock_symbol} (steady trend regime)")
                    return base_model, param_grid, 'f1_weighted'

            elif regime_type == 'low_volatility_sideways':
                # For low volatility sideways regimes, use simpler models
                if model_type == 'price':
                    # For price prediction in low volatility sideways regimes, linear models work well
                    base_model = LinearRegression()
                    param_grid = {}  # Linear regression doesn't have hyperparameters to tune
                    logger.info(f"Selected LinearRegression for {self.stock_symbol} (low volatility sideways regime)")
                    return base_model, param_grid, 'neg_mean_squared_error'
                else:  # signal model
                    # For signal prediction in low volatility sideways regimes, logistic regression works well
                    base_model = LogisticRegression(C=1.0)
                    param_grid = {
                        'C': [0.1, 0.5, 1.0, 2.0, 5.0]
                    }
                    logger.info(f"Selected LogisticRegression for {self.stock_symbol} (low volatility sideways regime)")
                    return base_model, param_grid, 'f1_weighted'

            # If regime-specific model selection didn't return a model, fall back to the default selection
            if model_type == 'price':
                # For price prediction

                # Check if we have enough data for deep learning models
                if data_size >= 500:
                    # For stocks with strong seasonality, Prophet works well
                    if has_seasonality and seasonality_strength > 0.3:
                        base_model = ProphetModel(
                            changepoint_prior_scale=0.05,
                            seasonality_mode='additive',
                            prediction_horizon=self.prediction_days
                        )
                        param_grid = {
                            'changepoint_prior_scale': [0.01, 0.05, 0.1],
                            'seasonality_mode': ['additive', 'multiplicative']
                        }
                        logger.info(f"Selected ProphetModel for {self.stock_symbol} (seasonal patterns)")

                    # For stocks with complex patterns and enough data, NeuralProphet works well
                    elif data_size >= 500 and (has_seasonality or volatility > 0.25):
                        base_model = NeuralProphetModel(
                            n_changepoints=10,
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            prediction_horizon=self.prediction_days
                        )
                        param_grid = {
                            'n_changepoints': [5, 10, 20],
                            'yearly_seasonality': [True, False],
                            'weekly_seasonality': [True, False]
                        }
                        logger.info(f"Selected NeuralProphetModel for {self.stock_symbol} (complex patterns)")

                    # For stocks with strong seasonality or cyclical patterns, Transformer models work well
                    elif has_seasonality and seasonality_strength > 0.4:
                        base_model = TransformerModel(
                            sequence_length=20,
                            embed_dim=32,
                            num_heads=2,
                            dropout=0.2
                        )
                        param_grid = {
                            'sequence_length': [10, 20, 30],
                            'embed_dim': [16, 32, 64],
                            'num_heads': [1, 2, 4],
                            'dropout': [0.1, 0.2, 0.3]
                        }
                        logger.info(f"Selected TransformerModel for {self.stock_symbol} (seasonal patterns)")

                    # For stocks with high volatility, LSTM models work well
                    elif volatility > 0.3:
                        base_model = LSTMModel(
                            sequence_length=20,
                            units=50,
                            dropout=0.2
                        )
                        param_grid = {
                            'sequence_length': [10, 20, 30],
                            'units': [32, 50, 64],
                            'dropout': [0.1, 0.2, 0.3]
                        }
                        logger.info(f"Selected LSTMModel for {self.stock_symbol} (high volatility)")

                    # For stocks with both technical and fundamental factors, hybrid models work well
                    elif data_size >= 1000:
                        base_model = HybridModel(
                            technical_model='lstm' if volatility > 0.2 else 'transformer',
                            sequence_length=20
                        )
                        param_grid = {
                            'sequence_length': [10, 20, 30],
                            'technical_model': ['lstm', 'transformer'],
                            'fundamental_weight': [0.2, 0.3, 0.4]
                        }
                        logger.info(f"Selected HybridModel for {self.stock_symbol} (large dataset)")

                    # Default to traditional models for other cases
                    else:
                        if data_size < 100:
                            # For small datasets, simpler models work better
                            base_model = LinearRegression()
                            param_grid = {}
                            logger.info(f"Selected LinearRegression for {self.stock_symbol} (small dataset)")
                        elif volatility > 0.4:  # High volatility
                            # For high volatility stocks, gradient boosting works well
                            base_model = GradientBoostingRegressor(random_state=42)
                            param_grid = {
                                'n_estimators': [100, 200, 300],
                                'learning_rate': [0.01, 0.05, 0.1],
                                'max_depth': [3, 4, 5],
                                'subsample': [0.8, 0.9, 1.0]
                            }
                            logger.info(f"Selected GradientBoostingRegressor for {self.stock_symbol} (high volatility)")
                        elif trend_strength > 0.01:  # Strong trend
                            # For trending stocks, XGBoost works well
                            base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
                            param_grid = {
                                'n_estimators': [100, 200, 300],
                                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                                'max_depth': [3, 4, 5, 6],
                                'subsample': [0.8, 0.9, 1.0],
                                'colsample_bytree': [0.8, 0.9, 1.0]
                            }
                            logger.info(f"Selected XGBRegressor for {self.stock_symbol} (strong trend)")
                        else:
                            # Default case - ensemble of models
                            lr = LinearRegression()
                            gbr = GradientBoostingRegressor(random_state=42, n_estimators=100)
                            xgbr = xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100)

                            base_model = VotingRegressor([
                                ('lr', lr),
                                ('gbr', gbr),
                                ('xgbr', xgbr)
                            ])
                            param_grid = {}  # No hyperparameter tuning for ensemble
                            logger.info(f"Selected VotingRegressor ensemble for {self.stock_symbol}")
                else:
                    # For small datasets, simpler models work better
                    if data_size < 100:
                        base_model = LinearRegression()
                        param_grid = {}
                        logger.info(f"Selected LinearRegression for {self.stock_symbol} (small dataset)")
                    elif volatility > 0.4:  # High volatility
                        # For high volatility stocks, gradient boosting works well
                        base_model = GradientBoostingRegressor(random_state=42)
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'max_depth': [3, 4, 5],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        logger.info(f"Selected GradientBoostingRegressor for {self.stock_symbol} (high volatility)")
                    elif trend_strength > 0.01:  # Strong trend
                        # For trending stocks, XGBoost works well
                        base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
                        param_grid = {
                            'n_estimators': [100, 200, 300],
                            'learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'max_depth': [3, 4, 5, 6],
                            'subsample': [0.8, 0.9, 1.0],
                            'colsample_bytree': [0.8, 0.9, 1.0]
                        }
                        logger.info(f"Selected XGBRegressor for {self.stock_symbol} (strong trend)")
                    else:
                        # Default case - ensemble of models
                        lr = LinearRegression()
                        gbr = GradientBoostingRegressor(random_state=42, n_estimators=100)
                        xgbr = xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100)

                        base_model = VotingRegressor([
                            ('lr', lr),
                            ('gbr', gbr),
                            ('xgbr', xgbr)
                        ])
                        param_grid = {}  # No hyperparameter tuning for ensemble
                        logger.info(f"Selected VotingRegressor ensemble for {self.stock_symbol}")

                scoring = 'neg_mean_squared_error'

            else:  # signal model
                # For signal prediction

                # Check if we have enough data for reinforcement learning
                if data_size >= 1000 and volatility > 0.25:
                    # For stocks with high volatility and enough data, RL works well for trading signals
                    base_model = RLTradingModel(
                        state_size=20,
                        train_episodes=100
                    )
                    param_grid = {
                        'state_size': [10, 20, 30],
                        'gamma': [0.9, 0.95, 0.99],
                        'train_episodes': [50, 100, 150]
                    }
                    logger.info(f"Selected RLTradingModel for {self.stock_symbol} (large dataset with volatility)")
                elif data_size < 100:
                    # For small datasets, logistic regression works better
                    base_model = LogisticRegression(random_state=42, class_weight='balanced')
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear', 'saga']
                    }
                    logger.info(f"Selected LogisticRegression for {self.stock_symbol} (small dataset)")
                elif volume_volatility > 0.5:  # High volume volatility
                    # For stocks with volatile volume, RandomForest works well
                    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                    param_grid = {
                        'n_estimators': [100, 150, 200],
                        'max_depth': [4, 6, 8, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    logger.info(f"Selected RandomForestClassifier for {self.stock_symbol} (high volume volatility)")
                elif volatility > 0.3:  # High price volatility
                    # For volatile stocks, XGBoost works well
                    base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 4, 5],
                        'subsample': [0.8, 0.9],
                        'scale_pos_weight': [1, 3, 5]  # For imbalanced classes
                    }
                    logger.info(f"Selected XGBClassifier for {self.stock_symbol} (high price volatility)")
                else:
                    # Default case - ensemble of models
                    lr = LogisticRegression(random_state=42, class_weight='balanced')
                    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
                    xgbc = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

                    base_model = VotingClassifier([
                        ('lr', lr),
                        ('rf', rf),
                        ('xgbc', xgbc)
                    ], voting='soft')
                    param_grid = {}  # No hyperparameter tuning for ensemble
                    logger.info(f"Selected VotingClassifier ensemble for {self.stock_symbol}")

                scoring = 'f1_weighted'  # Better than accuracy for imbalanced classes

            return base_model, param_grid, scoring

        except Exception as e:
            logger.error(f"Error selecting model based on characteristics for {self.stock_symbol}: {str(e)}")
            # Fallback to default models
            if model_type == 'price':
                base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
                scoring = 'neg_mean_squared_error'
            else:
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 8]
                }
                scoring = 'f1_weighted'

            return base_model, param_grid, scoring

    def get_time_series_cv(self, y=None, is_classification=False):
        """
        Get appropriate cross-validation strategy for time series data

        Args:
            y (pd.Series): Target variable (for classification)
            is_classification (bool): Whether this is a classification problem

        Returns:
            object: Cross-validation strategy
        """
        if is_classification and y is not None:
            class_counts = y.value_counts()
            min_class_samples = class_counts.min()

            if min_class_samples < 3:
                logger.warning(f"Too few class samples for '{self.stock_symbol}' - min class has only {min_class_samples} example(s)")
                return None

            # For classification with sufficient samples, use purged time series CV
            # This avoids look-ahead bias by ensuring train/test splits respect time
            if min_class_samples >= 20:
                n_splits = 5
                gap_size = 5  # Gap between train and test to avoid leakage

                # Custom time series split with gap
                indices = np.arange(len(y))
                test_size = len(y) // (n_splits + 1)  # Approximate test size

                cv_splits = []
                for i in range(n_splits):
                    test_start = i * test_size
                    test_end = (i + 1) * test_size
                    train_end = max(0, test_start - gap_size)

                    # Ensure we have enough training data
                    if train_end < 30:
                        continue

                    train_indices = indices[:train_end]
                    test_indices = indices[test_start:test_end]

                    # Check if we have samples from each class in both train and test
                    train_classes = set(y.iloc[train_indices])
                    test_classes = set(y.iloc[test_indices])

                    if len(train_classes) < 2 or len(test_classes) < 2:
                        continue

                    cv_splits.append((train_indices, test_indices))

                if len(cv_splits) >= 2:
                    logger.info(f"[{self.stock_symbol}] Using custom purged time series CV with {len(cv_splits)} splits")
                    return cv_splits

                # Fallback to standard TimeSeriesSplit if custom splits didn't work
                logger.info(f"[{self.stock_symbol}] Falling back to TimeSeriesSplit")
                return TimeSeriesSplit(n_splits=min(5, min_class_samples // 4))
            else:
                # For limited samples, use fewer splits
                n_splits = min(3, min_class_samples)
                logger.info(f"[{self.stock_symbol}] Using TimeSeriesSplit with {n_splits} splits (limited samples)")
                return TimeSeriesSplit(n_splits=n_splits)
        else:
            # For regression or when y is not provided
            n_splits = 5
            logger.info(f"[{self.stock_symbol}] Using TimeSeriesSplit with {n_splits} splits for regression")
            return TimeSeriesSplit(n_splits=n_splits)

    def statistical_significance_test(self, model1_preds, model2_preds, y_true, model_type='signal'):
        """
        Perform statistical significance testing to compare two models

        Args:
            model1_preds: Predictions from first model
            model2_preds: Predictions from second model
            y_true: True values
            model_type: 'price' or 'signal'

        Returns:
            dict: Results of statistical tests
        """
        results = {}

        try:
            if model_type == 'price':
                # For regression, compare MSE using paired t-test
                errors1 = (model1_preds - y_true) ** 2
                errors2 = (model2_preds - y_true) ** 2

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(errors1, errors2)
                results['t_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'better_model': 'model1' if t_stat < 0 else 'model2'
                }

                # Wilcoxon signed-rank test (non-parametric alternative)
                w_stat, p_value = stats.wilcoxon(errors1, errors2)
                results['wilcoxon_test'] = {
                    'statistic': float(w_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'better_model': 'model1' if w_stat < len(errors1) * (len(errors1) + 1) / 4 else 'model2'
                }

            else:  # signal model
                # For classification, use McNemar's test
                # This tests whether the disagreements between models are statistically significant
                contingency_table = np.zeros((2, 2))

                for i in range(len(y_true)):
                    correct1 = model1_preds[i] == y_true[i]
                    correct2 = model2_preds[i] == y_true[i]
                    contingency_table[int(correct1), int(correct2)] += 1

                # McNemar's test
                try:
                    chi2, p_value = stats.mcnemar(contingency_table, exact=True)
                    results['mcnemar_test'] = {
                        'chi2': float(chi2),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'better_model': 'model1' if contingency_table[1, 0] > contingency_table[0, 1] else 'model2'
                    }
                except ValueError as e:
                    logger.warning(f"McNemar test failed: {str(e)}")
                    results['mcnemar_test'] = {
                        'error': str(e),
                        'significant': False
                    }

                # Compare F1 scores
                f1_model1 = f1_score(y_true, model1_preds, average='weighted')
                f1_model2 = f1_score(y_true, model2_preds, average='weighted')
                results['f1_comparison'] = {
                    'f1_model1': float(f1_model1),
                    'f1_model2': float(f1_model2),
                    'difference': float(f1_model1 - f1_model2),
                    'better_model': 'model1' if f1_model1 > f1_model2 else 'model2'
                }

            return results

        except Exception as e:
            logger.error(f"Error in statistical significance test: {str(e)}")
            return {'error': str(e)}

    def improve_model_robustness(self, base_model, param_grid, model_type):
        """
        Improve model robustness using regularization, early stopping, ensembles, and adversarial training

        Args:
            base_model: The base model to improve
            param_grid: Parameter grid for hyperparameter tuning
            model_type (str): 'price' or 'signal'

        Returns:
            tuple: (improved_model, updated_param_grid)
        """
        try:
            # 1. Add regularization techniques to prevent overfitting
            if model_type == 'price':
                # For regression models
                if isinstance(base_model, xgb.XGBRegressor):
                    # XGBoost already has regularization parameters in param_grid
                    # Add L1 and L2 regularization parameters if not already present
                    if 'reg_alpha' not in param_grid:
                        param_grid['reg_alpha'] = [0, 0.001, 0.01, 0.1, 1.0]  # L1 regularization
                    if 'reg_lambda' not in param_grid:
                        param_grid['reg_lambda'] = [0.01, 0.1, 1.0, 10.0]  # L2 regularization

                    logger.info(f"Added L1/L2 regularization to XGBRegressor for {self.stock_symbol}")

                elif isinstance(base_model, GradientBoostingRegressor):
                    # Add alpha parameter for GradientBoostingRegressor
                    if 'alpha' not in param_grid:
                        param_grid['alpha'] = [0.1, 0.5, 0.9]  # Quantile regression alpha

                    logger.info(f"Added alpha regularization to GradientBoostingRegressor for {self.stock_symbol}")

                elif isinstance(base_model, LinearRegression):
                    # Replace LinearRegression with Ridge or ElasticNet for regularization
                    from sklearn.linear_model import ElasticNet

                    # Use Ridge regression (L2 regularization)
                    base_model = Ridge(random_state=42)
                    param_grid = {
                        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                    }

                    logger.info(f"Replaced LinearRegression with Ridge for {self.stock_symbol}")

                elif isinstance(base_model, VotingRegressor):
                    # For ensemble, apply regularization to each base estimator
                    regularized_estimators = []

                    for name, estimator in base_model.estimators:
                        if isinstance(estimator, LinearRegression):
                            # Replace with Ridge
                            regularized_estimators.append((name, Ridge(alpha=1.0, random_state=42)))
                        elif isinstance(estimator, xgb.XGBRegressor):
                            # Add regularization
                            estimator.set_params(reg_alpha=0.01, reg_lambda=1.0)
                            regularized_estimators.append((name, estimator))
                        else:
                            regularized_estimators.append((name, estimator))

                    base_model = VotingRegressor(regularized_estimators)
                    logger.info(f"Applied regularization to VotingRegressor components for {self.stock_symbol}")

            else:
                # For classification models
                if isinstance(base_model, RandomForestClassifier):
                    # RandomForest has implicit regularization through max_depth and min_samples_split
                    # Ensure these parameters are in the grid
                    if 'max_depth' not in param_grid:
                        param_grid['max_depth'] = [3, 5, 7, None]
                    if 'min_samples_split' not in param_grid:
                        param_grid['min_samples_split'] = [2, 5, 10]
                    if 'min_samples_leaf' not in param_grid:
                        param_grid['min_samples_leaf'] = [1, 2, 4]

                    logger.info(f"Added tree-based regularization to RandomForestClassifier for {self.stock_symbol}")

                elif isinstance(base_model, xgb.XGBClassifier):
                    # Add regularization parameters
                    if 'reg_alpha' not in param_grid:
                        param_grid['reg_alpha'] = [0, 0.001, 0.01, 0.1, 1.0]
                    if 'reg_lambda' not in param_grid:
                        param_grid['reg_lambda'] = [0.01, 0.1, 1.0, 10.0]

                    logger.info(f"Added L1/L2 regularization to XGBClassifier for {self.stock_symbol}")

                elif isinstance(base_model, LogisticRegression):
                    # Add regularization parameters
                    if 'C' not in param_grid:
                        param_grid['C'] = [0.001, 0.01, 0.1, 1.0, 10.0]  # Inverse of regularization strength
                    if 'penalty' not in param_grid:
                        param_grid['penalty'] = ['l1', 'l2', 'elasticnet']
                    if 'solver' not in param_grid:
                        param_grid['solver'] = ['saga']  # Only saga supports all penalties
                    if 'l1_ratio' not in param_grid:
                        param_grid['l1_ratio'] = [0.2, 0.5, 0.8]  # For elasticnet

                    logger.info(f"Added regularization parameters to LogisticRegression for {self.stock_symbol}")

                elif isinstance(base_model, VotingClassifier):
                    # For ensemble, apply regularization to each base estimator
                    regularized_estimators = []

                    for name, estimator in base_model.estimators:
                        if isinstance(estimator, LogisticRegression):
                            # Add regularization
                            estimator.set_params(C=0.1, penalty='l2')
                            regularized_estimators.append((name, estimator))
                        elif isinstance(estimator, xgb.XGBClassifier):
                            # Add regularization
                            estimator.set_params(reg_alpha=0.01, reg_lambda=1.0)
                            regularized_estimators.append((name, estimator))
                        else:
                            regularized_estimators.append((name, estimator))

                    base_model = VotingClassifier(regularized_estimators, voting='soft')
                    logger.info(f"Applied regularization to VotingClassifier components for {self.stock_symbol}")

            # 2. Implement early stopping for iterative models
            if isinstance(base_model, xgb.XGBRegressor) or isinstance(base_model, xgb.XGBClassifier):
                # Add early stopping parameters
                base_model.set_params(early_stopping_rounds=10)
                logger.info(f"Added early stopping to XGBoost model for {self.stock_symbol}")

            # 3. Create model ensembles to reduce prediction variance
            # Only create ensembles if the base model is not already an ensemble
            if not isinstance(base_model, VotingRegressor) and not isinstance(base_model, VotingClassifier):
                if model_type == 'price':
                    # For regression, create a stacking ensemble
                    from sklearn.ensemble import StackingRegressor

                    # Create base estimators
                    estimators = []

                    # Add the original model
                    estimators.append(('original', base_model))

                    # Add a Ridge regressor
                    estimators.append(('ridge', Ridge(alpha=1.0, random_state=42)))

                    # Add a different type of model
                    if not isinstance(base_model, GradientBoostingRegressor):
                        estimators.append(('gbr', GradientBoostingRegressor(
                            n_estimators=100, 
                            learning_rate=0.1, 
                            random_state=42
                        )))

                    # Create the stacking ensemble with a final estimator
                    stacking_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(random_state=42),
                        cv=3
                    )

                    # Replace the base model with the ensemble
                    base_model = stacking_model
                    param_grid = {}  # No hyperparameter tuning for the ensemble

                    logger.info(f"Created StackingRegressor ensemble for {self.stock_symbol}")

                else:
                    # For classification, create a stacking ensemble
                    from sklearn.ensemble import StackingClassifier

                    # Create base estimators
                    estimators = []

                    # Add the original model
                    estimators.append(('original', base_model))

                    # Add a LogisticRegression classifier
                    estimators.append(('lr', LogisticRegression(
                        C=1.0, 
                        class_weight='balanced', 
                        random_state=42
                    )))

                    # Add a different type of model
                    if not isinstance(base_model, RandomForestClassifier):
                        estimators.append(('rf', RandomForestClassifier(
                            n_estimators=100, 
                            class_weight='balanced', 
                            random_state=42
                        )))

                    # Create the stacking ensemble with a final estimator
                    stacking_model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(random_state=42),
                        cv=3
                    )

                    # Replace the base model with the ensemble
                    base_model = stacking_model
                    param_grid = {}  # No hyperparameter tuning for the ensemble

                    logger.info(f"Created StackingClassifier ensemble for {self.stock_symbol}")

            # 4. Add adversarial training for robustness to market shocks
            # This is a simplified version of adversarial training
            # We'll add random noise to the training data during fitting
            if model_type == 'price':
                # For regression models, wrap in a custom model that adds noise during training
                original_model = base_model

                class NoiseRobustRegressor:
                    def __init__(self, base_estimator, noise_level=0.01):
                        self.base_estimator = base_estimator
                        self.noise_level = noise_level

                    def fit(self, X, y):
                        # Add random noise to features
                        noise = np.random.normal(0, self.noise_level, X.shape)
                        X_noisy = X + noise

                        # Fit the base estimator
                        self.base_estimator.fit(X_noisy, y)
                        return self

                    def predict(self, X):
                        return self.base_estimator.predict(X)

                    def get_params(self, deep=True):
                        return {'base_estimator': self.base_estimator, 'noise_level': self.noise_level}

                    def set_params(self, **params):
                        for key, value in params.items():
                            setattr(self, key, value)
                        return self

                # Create the noise-robust model
                base_model = NoiseRobustRegressor(original_model, noise_level=0.01)
                logger.info(f"Added adversarial training (noise) to regression model for {self.stock_symbol}")

            else:
                # For classification models, wrap in a custom model that adds noise during training
                original_model = base_model

                class NoiseRobustClassifier:
                    def __init__(self, base_estimator, noise_level=0.01):
                        self.base_estimator = base_estimator
                        self.noise_level = noise_level

                    def fit(self, X, y):
                        # Add random noise to features
                        noise = np.random.normal(0, self.noise_level, X.shape)
                        X_noisy = X + noise

                        # Fit the base estimator
                        self.base_estimator.fit(X_noisy, y)
                        return self

                    def predict(self, X):
                        return self.base_estimator.predict(X)

                    def predict_proba(self, X):
                        if hasattr(self.base_estimator, 'predict_proba'):
                            return self.base_estimator.predict_proba(X)
                        else:
                            raise AttributeError("Base estimator does not have predict_proba method")

                    def get_params(self, deep=True):
                        return {'base_estimator': self.base_estimator, 'noise_level': self.noise_level}

                    def set_params(self, **params):
                        for key, value in params.items():
                            setattr(self, key, value)
                        return self

                # Create the noise-robust model
                base_model = NoiseRobustClassifier(original_model, noise_level=0.01)
                logger.info(f"Added adversarial training (noise) to classification model for {self.stock_symbol}")

            return base_model, param_grid

        except Exception as e:
            logger.error(f"Error improving model robustness for {self.stock_symbol}: {str(e)}")
            # Return the original model and param_grid if there's an error
            return base_model, param_grid

    def handle_class_imbalance(self, X, y, model_type='signal'):
        """
        Handle class imbalance using advanced techniques

        Args:
            X (numpy.ndarray): Feature matrix
            y (pd.Series): Target variable
            model_type (str): 'price' or 'signal'

        Returns:
            tuple: (X_resampled, y_resampled, class_weights, recommended_model)
        """
        try:
            if model_type != 'signal':
                # For regression, we don't need to handle class imbalance
                return X, y, None, None

            # Check class distribution
            class_counts = y.value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

            logger.info(f"Class distribution for {self.stock_symbol}: {dict(class_counts)}")
            logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")

            # If the imbalance is not severe, just use class weights
            if imbalance_ratio < 3:
                logger.info(f"Imbalance not severe for {self.stock_symbol}, using class weights only")
                # Calculate class weights inversely proportional to class frequencies
                class_weights = {
                    cls: len(y) / (len(class_counts) * count)
                    for cls, count in class_counts.items()
                }
                return X, y, class_weights, None

            # For moderate imbalance (3-10), use SMOTE
            elif imbalance_ratio < 10:
                logger.info(f"Moderate imbalance for {self.stock_symbol}, using SMOTE")

                # Check if we have enough samples for SMOTE
                if min_class >= 5:
                    try:
                        # Use SMOTE with appropriate k_neighbors
                        k_neighbors = min(min_class - 1, 5)  # k must be <= n_minority_samples - 1
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                        X_resampled, y_resampled = smote.fit_resample(X, y)

                        logger.info(f"SMOTE resampling successful for {self.stock_symbol}: {X.shape} -> {X_resampled.shape}")

                        # Still use some class weights for robustness
                        class_weights = {
                            cls: len(y_resampled) / (len(class_counts) * count)
                            for cls, count in pd.Series(y_resampled).value_counts().items()
                        }

                        return X_resampled, pd.Series(y_resampled), class_weights, None
                    except Exception as e:
                        logger.warning(f"SMOTE failed for {self.stock_symbol}: {str(e)}, falling back to class weights")
                        # Fallback to class weights
                        class_weights = {
                            cls: len(y) / (len(class_counts) * count)
                            for cls, count in class_counts.items()
                        }
                        return X, y, class_weights, None
                else:
                    logger.warning(f"Not enough samples for SMOTE for {self.stock_symbol}, using class weights")
                    # Fallback to class weights
                    class_weights = {
                        cls: len(y) / (len(class_counts) * count)
                        for cls, count in class_counts.items()
                    }
                    return X, y, class_weights, None

            # For severe imbalance (>10), use ADASYN and ensemble methods
            else:
                logger.info(f"Severe imbalance for {self.stock_symbol}, using ADASYN and ensemble methods")

                # Check if we have enough samples for ADASYN
                if min_class >= 5:
                    try:
                        # Use ADASYN with appropriate n_neighbors
                        n_neighbors = min(min_class - 1, 5)  # n must be <= n_minority_samples - 1
                        adasyn = ADASYN(random_state=42, n_neighbors=n_neighbors)
                        X_resampled, y_resampled = adasyn.fit_resample(X, y)

                        logger.info(f"ADASYN resampling successful for {self.stock_symbol}: {X.shape} -> {X_resampled.shape}")

                        # Create an ensemble specifically for imbalanced data
                        # 1. Random Forest with balanced class weights
                        rf = RandomForestClassifier(
                            n_estimators=100, 
                            class_weight='balanced',
                            random_state=42
                        )

                        # 2. XGBoost with scale_pos_weight
                        # Calculate scale_pos_weight as ratio of negative to positive samples
                        neg_count = sum(y_resampled == 0)
                        pos_count = sum(y_resampled == 1)
                        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

                        xgb_clf = xgb.XGBClassifier(
                            n_estimators=100,
                            scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )

                        # 3. Logistic Regression with balanced class weights
                        lr = LogisticRegression(
                            class_weight='balanced',
                            random_state=42,
                            max_iter=1000
                        )

                        # Create a voting ensemble
                        ensemble = VotingClassifier(
                            estimators=[
                                ('rf', rf),
                                ('xgb', xgb_clf),
                                ('lr', lr)
                            ],
                            voting='soft'
                        )

                        # Calculate class weights for any other models
                        class_weights = {
                            cls: len(y_resampled) / (len(class_counts) * count)
                            for cls, count in pd.Series(y_resampled).value_counts().items()
                        }

                        return X_resampled, pd.Series(y_resampled), class_weights, ensemble
                    except Exception as e:
                        logger.warning(f"ADASYN failed for {self.stock_symbol}: {str(e)}, falling back to cost-sensitive learning")
                        # Fallback to cost-sensitive learning
                        # Calculate class weights with higher penalty for minority class
                        class_weights = {
                            cls: (len(y) / (len(class_counts) * count)) ** 1.5  # Exponential scaling
                            for cls, count in class_counts.items()
                        }

                        # Create a cost-sensitive ensemble
                        rf = RandomForestClassifier(
                            n_estimators=100, 
                            class_weight=class_weights,
                            random_state=42
                        )

                        return X, y, class_weights, rf
                else:
                    logger.warning(f"Not enough samples for ADASYN for {self.stock_symbol}, using cost-sensitive learning")
                    # Fallback to cost-sensitive learning
                    # Calculate class weights with higher penalty for minority class
                    class_weights = {
                        cls: (len(y) / (len(class_counts) * count)) ** 1.5  # Exponential scaling
                        for cls, count in class_counts.items()
                    }

                    # Create a cost-sensitive model
                    rf = RandomForestClassifier(
                        n_estimators=100, 
                        class_weight=class_weights,
                        random_state=42
                    )

                    return X, y, class_weights, rf

        except Exception as e:
            logger.error(f"Error handling class imbalance for {self.stock_symbol}: {str(e)}")
            # Return original data if there's an error
            return X, y, None, None

    def _train_model(self, model_type):
        """Train a new model with hyperparameter search and optional calibration, then save it."""
        X, y_price, y_signal = self.prepare_data()

        if X is None or len(X) < 30:
            logger.error(f"Insufficient data to train {model_type} model for {self.stock_symbol}")
            return None

        # Determine target variable
        if model_type == 'signal':
            y = y_signal
            class_counts = y.value_counts()
            min_class_samples = class_counts.min()

            if min_class_samples < 3:
                logger.warning(
                    f"Abbruch: Zu wenige Klassendaten für '{self.stock_symbol}' – minimalste Klasse hat nur {min_class_samples} Beispiel(e)"
                )
                return None

            # Handle class imbalance
            X_resampled, y_resampled, class_weights, imbalance_model = self.handle_class_imbalance(X, y, model_type)

            # Use the resampled data if available
            if X_resampled is not None and y_resampled is not None:
                X = X_resampled
                y = y_resampled

            # If we have a specific model for imbalanced data, use it
            if imbalance_model is not None:
                logger.info(f"Using specialized model for imbalanced data for {self.stock_symbol}")
                # Train the model
                imbalance_model.fit(X, y)
                return imbalance_model
        else:
            y = y_price

        # Get appropriate cross-validation strategy
        cv_strategy = self.get_time_series_cv(y, is_classification=(model_type == 'signal'))
        if cv_strategy is None:
            return None

        try:
            # Select model based on stock characteristics
            base_model, param_grid, scoring = self.select_model_based_on_characteristics(model_type)

            # Apply robustness improvements
            base_model, param_grid = self.improve_model_robustness(base_model, param_grid, model_type)

            # Set model path
            if model_type == 'price':
                model_path = self.price_model_path
            else:
                model_path = self.signal_model_path

            best_score = float('-inf')
            best_model = None

            # If cv_strategy is a list of tuples, it's our custom CV
            if isinstance(cv_strategy, list):
                cv_splits = cv_strategy
            else:
                # Otherwise use the sklearn CV object
                cv_splits = list(cv_strategy.split(X, y if model_type == 'signal' else None))

            for train_idx, test_idx in cv_splits:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # For classification, determine CV folds based on class distribution
                if model_type == 'signal':
                    cv_folds = min(3, y_train.value_counts().min())
                else:
                    cv_folds = 3

                # Skip if we don't have enough data for cross-validation
                if cv_folds < 2:
                    continue

                # Hyperparameter search
                if param_grid:  # Only do search if we have parameters to tune
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
                else:
                    # If no param_grid (e.g., for ensembles), just fit the model
                    model = base_model
                    model.fit(X_train, y_train)

                # Calibration for classification models
                if model_type == 'signal' and cv_folds >= 2:
                    # Only calibrate if the model supports predict_proba
                    if hasattr(model, 'predict_proba'):
                        model = CalibratedClassifierCV(model, method='isotonic', cv=cv_folds)
                        model.fit(X_train, y_train)
                    else:
                        logger.warning(f"Calibration skipped for {self.stock_symbol} - model doesn't support predict_proba")
                elif model_type == 'signal':
                    logger.warning(f"Calibration skipped for {self.stock_symbol} - insufficient data")

                # Evaluation
                if model_type == 'price':
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)

                    # Calculate directional accuracy
                    direction_actual = np.sign(y_test)
                    direction_pred = np.sign(preds)
                    direction_accuracy = np.mean(direction_actual == direction_pred)

                    # Combine metrics into a single score (weighted average)
                    # Higher is better, so negate MSE-based metrics
                    score = -0.4*rmse + 0.3*r2 + 0.3*direction_accuracy

                    logger.info(f"Fold evaluation for {self.stock_symbol} (price): "
                               f"RMSE={rmse:.4f}, R²={r2:.4f}, Direction Acc={direction_accuracy:.4f}, Score={score:.4f}")
                else:
                    preds = model.predict(X_test)
                    accuracy = accuracy_score(y_test, preds)
                    precision = precision_score(y_test, preds, average='weighted')
                    recall = recall_score(y_test, preds, average='weighted')
                    f1 = f1_score(y_test, preds, average='weighted')

                    # For models that support probability predictions
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba_preds = model.predict_proba(X_test)
                            # For binary classification
                            if proba_preds.shape[1] == 2:
                                auc_score = roc_auc_score(y_test, proba_preds[:, 1])
                            else:
                                # For multiclass, use one-vs-rest approach
                                auc_score = roc_auc_score(y_test, proba_preds, multi_class='ovr')
                        except Exception as e:
                            logger.warning(f"AUC calculation failed: {str(e)}")
                            auc_score = 0
                    else:
                        auc_score = 0

                    # Combine metrics into a single score (weighted average)
                    score = 0.25*accuracy + 0.25*precision + 0.25*recall + 0.25*f1

                    logger.info(f"Fold evaluation for {self.stock_symbol} (signal): "
                               f"Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}, Score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model

            if best_model is not None:
                joblib.dump(best_model, model_path)

                # Save feature names for consistent feature selection during prediction
                feature_names_path = os.path.join(self.models_dir, f'{self.stock_symbol}_feature_names.pkl')

                # Get feature names from original dataframe
                stock = Stock.objects.get(symbol=self.stock_symbol)
                df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

                # Ensure numeric conversion
                for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                    df[col] = df[col].astype(float)

                df = self._calculate_features(df)
                feature_columns = [col for col in df.columns if col not in
                                  ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

                # Load existing feature names if available
                if hasattr(self, 'feature_names') and self.feature_names:
                    feature_names = self.feature_names
                else:
                    feature_names = {}

                # Update feature names for current model type
                feature_names[model_type] = feature_columns

                # Store feature names
                self.feature_names = feature_names
                joblib.dump(self.feature_names, feature_names_path)
                logger.info(f"Saved feature names for {self.stock_symbol} {model_type} model")

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

    def calibrate_predictions(self, X, model_type='signal'):
        """
        Calibrate predictions and provide uncertainty estimates

        Args:
            X (numpy.ndarray): Feature matrix
            model_type (str): 'price' or 'signal'

        Returns:
            tuple: (predictions, confidence_intervals, calibrated_probabilities)
        """
        try:
            if model_type == 'price' and self.price_model is not None:
                # For regression models, we estimate prediction intervals
                # using quantile regression or bootstrap

                # Method 1: Simple approach using residual standard deviation
                preds = self.price_model.predict(X)

                # Get historical residuals from training data
                X_train, y_price, _ = self.prepare_data()
                if X_train is None or y_price is None:
                    logger.error(f"No training data available for {self.stock_symbol}")
                    return preds, None, None

                train_preds = self.price_model.predict(X_train)
                residuals = y_price - train_preds
                residual_std = np.std(residuals)

                # Calculate prediction intervals (95% confidence)
                lower_bound = preds - 1.96 * residual_std
                upper_bound = preds + 1.96 * residual_std

                confidence_intervals = np.column_stack((lower_bound, upper_bound))

                # Method 2: Bootstrap for more robust intervals (if we have enough data)
                if len(X_train) >= 100:
                    try:
                        from sklearn.base import clone
                        n_bootstraps = 100
                        bootstrap_predictions = np.zeros((len(X), n_bootstraps))

                        # Create bootstrap samples and predict
                        for i in range(n_bootstraps):
                            # Sample with replacement
                            bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
                            X_bootstrap = X_train[bootstrap_indices]
                            y_bootstrap = y_price.iloc[bootstrap_indices]

                            # Train a model on the bootstrap sample
                            if hasattr(self.price_model, 'fit'):
                                bootstrap_model = clone(self.price_model)
                                bootstrap_model.fit(X_bootstrap, y_bootstrap)
                                bootstrap_predictions[:, i] = bootstrap_model.predict(X)
                            else:
                                # If model doesn't support fit (e.g., it's an ensemble), use original predictions
                                bootstrap_predictions[:, i] = preds

                        # Calculate bootstrap confidence intervals
                        lower_percentile = np.percentile(bootstrap_predictions, 2.5, axis=1)
                        upper_percentile = np.percentile(bootstrap_predictions, 97.5, axis=1)
                        bootstrap_intervals = np.column_stack((lower_percentile, upper_percentile))

                        # Use bootstrap intervals if available
                        confidence_intervals = bootstrap_intervals
                        logger.info(f"Using bootstrap confidence intervals for {self.stock_symbol}")
                    except Exception as e:
                        logger.warning(f"Bootstrap confidence intervals failed: {str(e)}. Using simple intervals.")

                # Create visualization of uncertainty
                try:
                    if len(X) <= 30:  # Only visualize for reasonable number of predictions
                        plt.figure(figsize=(10, 6))
                        x_values = range(len(preds))
                        plt.plot(x_values, preds, 'b-', label='Prediction')
                        plt.fill_between(x_values, 
                                        confidence_intervals[:, 0], 
                                        confidence_intervals[:, 1], 
                                        color='b', alpha=0.2, label='95% Confidence Interval')
                        plt.title(f'Price Predictions with Uncertainty for {self.stock_symbol}')
                        plt.xlabel('Prediction Index')
                        plt.ylabel('Predicted Price Change')
                        plt.legend()

                        # Save the plot
                        uncertainty_plot_path = os.path.join('static', 'images', 'uncertainty')
                        os.makedirs(uncertainty_plot_path, exist_ok=True)
                        plt.savefig(os.path.join(uncertainty_plot_path, f'{self.stock_symbol}_price_uncertainty.png'))
                        plt.close()
                except Exception as e:
                    logger.warning(f"Error creating uncertainty visualization: {str(e)}")

                return preds, confidence_intervals, None

            elif model_type == 'signal' and self.signal_model is not None:
                # For classification models, we calibrate probabilities

                # Check if model supports probability predictions
                if not hasattr(self.signal_model, 'predict_proba'):
                    logger.warning(f"Model for {self.stock_symbol} doesn't support probability predictions")
                    return self.signal_model.predict(X), None, None

                # Get raw probability predictions
                raw_probs = self.signal_model.predict_proba(X)
                predictions = self.signal_model.predict(X)

                # Check if we already have a calibrated model
                if hasattr(self.signal_model, 'calibrated_classifiers_'):
                    logger.info(f"Using already calibrated model for {self.stock_symbol}")
                    calibrated_probs = raw_probs
                else:
                    # Try to calibrate the model
                    try:
                        # Get training data
                        X_train, _, y_signal = self.prepare_data()
                        if X_train is None or y_signal is None:
                            logger.error(f"No training data available for {self.stock_symbol}")
                            return predictions, None, raw_probs

                        # Split data for calibration
                        train_size = int(0.8 * len(X_train))
                        X_calib = X_train[train_size:]
                        y_calib = y_signal.iloc[train_size:]

                        # Create a calibrated model
                        calibrator = CalibratedClassifierCV(self.signal_model, cv='prefit', method='isotonic')
                        calibrator.fit(X_calib, y_calib)

                        # Get calibrated probabilities
                        calibrated_probs = calibrator.predict_proba(X)

                        logger.info(f"Successfully calibrated probabilities for {self.stock_symbol}")
                    except Exception as e:
                        logger.warning(f"Probability calibration failed: {str(e)}. Using raw probabilities.")
                        calibrated_probs = raw_probs

                # Calculate confidence intervals for probabilities
                # For classification, this is based on the Beta distribution
                n_samples = 100  # Number of samples in calibration set
                alpha = 0.05  # For 95% confidence

                confidence_intervals = []

                for i in range(len(calibrated_probs)):
                    class_intervals = []
                    for j in range(calibrated_probs.shape[1]):
                        p = calibrated_probs[i, j]
                        # Beta distribution parameters
                        a = n_samples * p + 1
                        b = n_samples * (1 - p) + 1

                        # Confidence interval
                        lower = max(0, stats.beta.ppf(alpha/2, a, b))
                        upper = min(1, stats.beta.ppf(1 - alpha/2, a, b))
                        class_intervals.append((lower, upper))

                    confidence_intervals.append(class_intervals)

                # Create visualization of calibration
                try:
                    # Only for binary classification
                    if calibrated_probs.shape[1] == 2:

                        # Get calibration curve
                        X_train, _, y_signal = self.prepare_data()
                        if X_train is not None and y_signal is not None:
                            # Use a separate validation set
                            train_size = int(0.8 * len(X_train))
                            X_val = X_train[train_size:]
                            y_val = y_signal.iloc[train_size:]

                            # Get predictions
                            val_probs = self.signal_model.predict_proba(X_val)[:, 1]

                            # Calculate calibration curve
                            prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10)

                            # Plot calibration curve
                            plt.figure(figsize=(10, 6))
                            plt.plot(prob_pred, prob_true, 's-', label='Calibration curve')
                            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
                            plt.title(f'Probability Calibration for {self.stock_symbol}')
                            plt.xlabel('Mean predicted probability')
                            plt.ylabel('Fraction of positives')
                            plt.legend()

                            # Save the plot
                            calibration_plot_path = os.path.join('static', 'images', 'calibration')
                            os.makedirs(calibration_plot_path, exist_ok=True)
                            plt.savefig(os.path.join(calibration_plot_path, f'{self.stock_symbol}_calibration.png'))
                            plt.close()
                except Exception as e:
                    logger.warning(f"Error creating calibration visualization: {str(e)}")

                return predictions, confidence_intervals, calibrated_probs

            else:
                logger.warning(f"No model available for {model_type} predictions for {self.stock_symbol}")
                return None, None, None

        except Exception as e:
            logger.error(f"Error in calibrate_predictions for {self.stock_symbol}: {str(e)}")
            return None, None, None

    def predict(self, use_feature_importance=True, feature_importance_threshold=0.01):
        """
        Make predictions with the trained models, optionally using feature importance analysis

        Args:
            use_feature_importance (bool): If True, analyze feature importance and use only important features
            feature_importance_threshold (float): Threshold for feature importance analysis

        Returns:
            dict: Prediction results including recommendation, confidence, and adaptive thresholds
        """
        try:
            print(f"DEBUG: Starting prediction for {self.stock_symbol}")

            # Detect current market regime
            current_regime = self.detect_market_regime()
            regime_type = current_regime.get('regime_type', 'unknown') if current_regime else 'unknown'
            regime_name = current_regime.get('regime_name', 'Unknown') if current_regime else 'Unknown'

            logger.info(f"Current market regime for {self.stock_symbol}: {regime_name} ({regime_type})")

            # Check if we need to retrain models based on the current regime
            if hasattr(self, 'last_regime_type') and self.last_regime_type != regime_type:
                logger.info(f"Market regime changed from {self.last_regime_type} to {regime_type}. Retraining models.")
                # Retrain models for the new regime
                self.price_model = self._train_model('price')
                self.signal_model = self._train_model('signal')

            # Store current regime type for future comparisons
            self.last_regime_type = regime_type

            X, _, _ = self.prepare_data()

            if X is None:
                print(f"DEBUG: No data available for prediction for {self.stock_symbol}")
                return None

            print(f"DEBUG: After prepare_data, X shape: {X.shape}")

            if X is None or X.shape[0] == 0:
                logger.error(f"No data available for prediction for {self.stock_symbol}")
                return None

            # Feature importance analysis to remove unimportant features
            feature_importance_results = None
            feature_indices_to_keep = None

            if use_feature_importance:
                try:
                    logger.info(f"Analyzing feature importance for {self.stock_symbol}")
                    feature_importance_results = self.analyze_feature_importance(
                        threshold=feature_importance_threshold,
                        remove_least_important=True
                    )

                    if feature_importance_results and 'feature_indices_to_keep' in feature_importance_results:
                        feature_indices_to_keep = feature_importance_results['feature_indices_to_keep']
                        logger.info(f"Using {len(feature_indices_to_keep)}/{X.shape[1]} features for prediction")

                        # Filter features based on importance, but ensure we have at least some features
                        if len(feature_indices_to_keep) > 0:
                            X = X[:, feature_indices_to_keep]
                            logger.info(f"Reduced feature set shape: {X.shape}")
                        else:
                            logger.warning(f"No features deemed important enough! Using all features instead.")
                            feature_indices_to_keep = list(range(X.shape[1]))
                except Exception as e:
                    logger.error(f"Error in feature importance analysis: {str(e)}")
                    # Continue with all features if there's an error
                    feature_indices_to_keep = None

            latest_features = X[-1].reshape(1, -1)

            predicted_return = 0.0
            confidence = 0.0

            # Price Prediction
            if self.price_model is not None:
                # Check for feature count mismatch
                expected_n_features = self.price_model.n_features_in_ if hasattr(self.price_model, 'n_features_in_') else None

                if expected_n_features is not None and latest_features.shape[1] != expected_n_features:
                    logger.warning(f"Feature count mismatch for {self.stock_symbol}: Model expects {expected_n_features} features, but got {latest_features.shape[1]}")

                    # Check if we have saved feature names
                    if hasattr(self, 'feature_names') and self.feature_names and 'price' in self.feature_names:
                        logger.info(f"Using saved feature names for consistent feature selection")

                        # Get the original feature names from the current data
                        stock = Stock.objects.get(symbol=self.stock_symbol)
                        df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

                        # Ensure numeric conversion
                        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                            df[col] = df[col].astype(float)

                        df = self._calculate_features(df)
                        current_features = [col for col in df.columns if col not in
                                          ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

                        # Get the expected feature names from the saved model
                        expected_features = self.feature_names['price']

                        print(f"DEBUG: Current features count: {len(current_features)}")
                        print(f"DEBUG: Expected features count: {len(expected_features)}")

                        # Create a mapping from current features to expected features
                        feature_mapping = {}
                        for i, feature in enumerate(current_features):
                            if feature in expected_features:
                                feature_mapping[i] = expected_features.index(feature)

                        print(f"DEBUG: Mapped {len(feature_mapping)} features out of {len(current_features)}")

                        # If we have enough mappings, use them to rearrange features
                        if len(feature_mapping) >= expected_n_features * 0.8:  # At least 80% of features can be mapped
                            print(f"DEBUG: Have enough mappings ({len(feature_mapping)} >= {expected_n_features * 0.8}), using feature alignment")
                            # Create a new feature array with the correct order
                            print(f"DEBUG: Before alignment, X shape: {X.shape}")
                            X_aligned = np.zeros((X.shape[0], expected_n_features))
                            print(f"DEBUG: Created X_aligned with shape: {X_aligned.shape}")

                            # Log some of the mapping for debugging
                            mapping_sample = list(feature_mapping.items())[:5]
                            print(f"DEBUG: Mapping sample (first 5): {mapping_sample}")

                            try:
                                # Filter the mapping to only include valid indices for X
                                valid_mapping = {k: v for k, v in feature_mapping.items() if k < X.shape[1]}
                                print(f"DEBUG: Filtered mapping to {len(valid_mapping)} valid features (out of {len(feature_mapping)})")

                                for current_idx, expected_idx in valid_mapping.items():
                                    if expected_idx < expected_n_features:
                                        print(f"DEBUG: Mapping feature {current_idx} to {expected_idx}")
                                        X_aligned[:, expected_idx] = X[:, current_idx]

                                # Fill remaining features with zeros (they're already zeros, but log it)
                                missing_features = set(range(expected_n_features)) - set(valid_mapping.values())
                                print(f"DEBUG: {len(missing_features)} features will be zeros")
                            except Exception as e:
                                import traceback
                                print(f"DEBUG: Error during feature mapping: {str(e)}")
                                print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
                                raise

                            # Use the aligned features
                            X = X_aligned
                            latest_features = X[-1].reshape(1, -1)
                            print(f"DEBUG: After alignment, feature shape: {latest_features.shape}")
                            logger.info(f"Successfully aligned features using feature names mapping")
                        else:
                            print(f"DEBUG: Not enough mappings ({len(feature_mapping)} < {expected_n_features * 0.8}), falling back to retraining")
                            logger.warning(f"Could not align enough features using names, falling back to retraining")
                            # Fall back to retraining
                            self.price_model = self._train_model('price')
                            if self.price_model is None:
                                logger.error(f"Failed to retrain price model for {self.stock_symbol}")
                                print(f"DEBUG: Retraining failed, latest_features shape: {latest_features.shape}")
                                return None

                            # After retraining, recalculate features
                            X, _, _ = self.prepare_data()
                            if X is None or X.shape[0] == 0:
                                logger.error(f"No data available for prediction after retraining for {self.stock_symbol}")
                                return None

                            latest_features = X[-1].reshape(1, -1)
                    else:
                        # If we don't have saved feature names, use the old approach
                        if latest_features.shape[1] > expected_n_features:
                            # If we have more features than expected, use only the first n_features that the model expects
                            logger.info(f"Using only the first {expected_n_features} features for prediction")
                            latest_features = latest_features[:, :expected_n_features]
                        else:
                            # If we have fewer features than expected, we need to retrain the model
                            logger.warning(f"Retraining model for {self.stock_symbol} due to feature count mismatch")
                            self.price_model = self._train_model('price')
                            if self.price_model is None:
                                logger.error(f"Failed to retrain price model for {self.stock_symbol}")

                                # Extreme feature count mismatch fallback (e.g., GME with 6 vs 113 features)
                                # If retraining fails and we have a significant feature count mismatch, pad with zeros
                                print(f"DEBUG: Checking extreme feature count mismatch for {self.stock_symbol}: {latest_features.shape[1]} vs {expected_n_features}")
                                # Always use the fallback for extreme mismatches
                                if latest_features.shape[1] != expected_n_features:
                                    logger.warning(f"Extreme feature count mismatch for {self.stock_symbol}: {latest_features.shape[1]} vs {expected_n_features}")
                                    print(f"DEBUG: Using fallback padding for {self.stock_symbol}")

                                    # Create padded feature array
                                    padded_features = np.zeros((latest_features.shape[0], expected_n_features))
                                    # Copy available features (only up to the minimum of both dimensions)
                                    min_features = min(latest_features.shape[1], expected_n_features)
                                    padded_features[:, :min_features] = latest_features[:, :min_features]
                                    latest_features = padded_features

                                    logger.info(f"Padded features from {latest_features.shape[1]} to {expected_n_features}")
                                    print(f"DEBUG: After padding, feature shape: {latest_features.shape}")
                                else:
                                    return None

                            else:
                                # After retraining, we need to recalculate features to match the new models
                                X, _, _ = self.prepare_data()
                                if X is None or X.shape[0] == 0:
                                    logger.error(f"No data available for prediction after retraining for {self.stock_symbol}")
                                    return None

                                # If we're using feature importance, we need to recalculate it after retraining
                                if use_feature_importance:
                                    try:
                                        logger.info(f"Recalculating feature importance after retraining for {self.stock_symbol}")
                                        feature_importance_results = self.analyze_feature_importance(
                                            threshold=feature_importance_threshold,
                                            remove_least_important=True
                                        )

                                        if feature_importance_results and 'feature_indices_to_keep' in feature_importance_results:
                                            feature_indices_to_keep = feature_importance_results['feature_indices_to_keep']
                                            logger.info(f"Using {len(feature_indices_to_keep)}/{X.shape[1]} features for prediction after retraining")

                                            # Filter features based on importance, but ensure we have at least some features
                                            if len(feature_indices_to_keep) > 0:
                                                X = X[:, feature_indices_to_keep]
                                                logger.info(f"Reduced feature set shape after retraining: {X.shape}")
                                            else:
                                                logger.warning(f"No features deemed important enough after retraining! Using all features instead.")
                                                feature_indices_to_keep = list(range(X.shape[1]))
                                    except Exception as e:
                                        logger.error(f"Error in feature importance analysis after retraining: {str(e)}")
                                        # Continue with all features if there's an error
                                        feature_indices_to_keep = None

                                latest_features = X[-1].reshape(1, -1)

                try:
                    print(f"DEBUG: About to predict with price model, feature shape: {latest_features.shape}")
                    # Check if the feature count matches what the model expects
                    if hasattr(self.price_model, 'n_features_in_'):
                        print(f"DEBUG: Model expects {self.price_model.n_features_in_} features, got {latest_features.shape[1]}")
                        if latest_features.shape[1] != self.price_model.n_features_in_:
                            print(f"DEBUG: WARNING - Feature count mismatch right before prediction!")

                    # Use calibrated predictions with confidence intervals
                    preds, confidence_intervals, _ = self.calibrate_predictions(latest_features, model_type='price')
                    predicted_return = float(preds[0])

                    # Store confidence intervals for the result
                    if confidence_intervals is not None:
                        lower_bound = float(confidence_intervals[0, 0])
                        upper_bound = float(confidence_intervals[0, 1])
                        prediction_uncertainty = {
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'interval_width': upper_bound - lower_bound
                        }
                    else:
                        prediction_uncertainty = None

                    print(f"DEBUG: Price prediction successful: {predicted_return}")
                    print(f"DEBUG: Confidence interval: {prediction_uncertainty}")
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error in price prediction: {str(e)}")
                    print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
                    # Fallback to standard prediction if calibration fails
                    try:
                        predicted_return = float(self.price_model.predict(latest_features)[0])
                        prediction_uncertainty = None
                        print(f"DEBUG: Fallback price prediction successful: {predicted_return}")
                    except Exception as e2:
                        print(f"DEBUG: Fallback prediction also failed: {str(e2)}")
                        raise

            # Signal Prediction
            if self.signal_model is not None and hasattr(self.signal_model, 'predict_proba'):
                # Check for feature count mismatch
                expected_n_features = self.signal_model.n_features_in_ if hasattr(self.signal_model, 'n_features_in_') else None

                if expected_n_features is not None and latest_features.shape[1] != expected_n_features:
                    logger.warning(f"Feature count mismatch for signal model: Model expects {expected_n_features} features, but got {latest_features.shape[1]}")

                    # Check if we have saved feature names
                    if hasattr(self, 'feature_names') and self.feature_names and 'signal' in self.feature_names:
                        logger.info(f"Using saved feature names for consistent feature selection in signal model")

                        # Get the original feature names from the current data
                        stock = Stock.objects.get(symbol=self.stock_symbol)
                        df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

                        # Ensure numeric conversion
                        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                            df[col] = df[col].astype(float)

                        df = self._calculate_features(df)
                        current_features = [col for col in df.columns if col not in
                                          ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

                        # Get the expected feature names from the saved model
                        expected_features = self.feature_names['signal']

                        # Create a mapping from current features to expected features
                        feature_mapping = {}
                        for i, feature in enumerate(current_features):
                            if feature in expected_features:
                                feature_mapping[i] = expected_features.index(feature)

                        # If we have enough mappings, use them to rearrange features
                        if len(feature_mapping) >= expected_n_features * 0.8:  # At least 80% of features can be mapped
                            # Create a new feature array with the correct order
                            X_aligned = np.zeros((X.shape[0], expected_n_features))
                            for current_idx, expected_idx in feature_mapping.items():
                                if expected_idx < expected_n_features:
                                    X_aligned[:, expected_idx] = X[:, current_idx]

                            # Use the aligned features
                            X = X_aligned
                            latest_features = X[-1].reshape(1, -1)
                            latest_features_signal = latest_features
                            logger.info(f"Successfully aligned features using feature names mapping for signal model")
                        else:
                            logger.warning(f"Could not align enough features using names for signal model, falling back to retraining")
                            # Fall back to retraining
                            self.signal_model = self._train_model('signal')
                            if self.signal_model is None:
                                logger.error(f"Failed to retrain signal model for {self.stock_symbol}")
                                return None

                            # After retraining, recalculate features
                            X, _, _ = self.prepare_data()
                            if X is None or X.shape[0] == 0:
                                logger.error(f"No data available for prediction after retraining for {self.stock_symbol}")
                                return None

                            latest_features = X[-1].reshape(1, -1)
                            latest_features_signal = latest_features
                    else:
                        # If we don't have saved feature names, use the old approach
                        if latest_features.shape[1] > expected_n_features:
                            # If we have more features than expected, use only the first n_features that the model expects
                            logger.info(f"Using only the first {expected_n_features} features for signal prediction")
                            latest_features_signal = latest_features[:, :expected_n_features]
                        else:
                            # If we have fewer features than expected, we need to retrain the model
                            logger.warning(f"Retraining signal model for {self.stock_symbol} due to feature count mismatch")
                            self.signal_model = self._train_model('signal')
                            if self.signal_model is None:
                                logger.error(f"Failed to retrain signal model for {self.stock_symbol}")

                                # Extreme feature count mismatch fallback (e.g., GME with 6 vs 113 features)
                                # If retraining fails and we have a significant feature count mismatch, pad with zeros
                                print(f"DEBUG: Checking extreme feature count mismatch for signal model: {latest_features.shape[1]} vs {expected_n_features}")
                                # Always use the fallback for extreme mismatches
                                if latest_features.shape[1] != expected_n_features:
                                    logger.warning(f"Extreme feature count mismatch for signal model: {latest_features.shape[1]} vs {expected_n_features}")
                                    print(f"DEBUG: Using fallback padding for signal model")

                                    # Create padded feature array
                                    padded_features = np.zeros((latest_features.shape[0], expected_n_features))
                                    # Copy available features (only up to the minimum of both dimensions)
                                    min_features = min(latest_features.shape[1], expected_n_features)
                                    padded_features[:, :min_features] = latest_features[:, :min_features]
                                    latest_features_signal = padded_features

                                    logger.info(f"Padded features from {latest_features.shape[1]} to {expected_n_features}")
                                    print(f"DEBUG: After padding, signal feature shape: {latest_features_signal.shape}")
                                else:
                                    return None
                            else:
                                # After retraining, we need to recalculate features to match the new model
                                X, _, _ = self.prepare_data()
                                if X is None or X.shape[0] == 0:
                                    logger.error(f"No data available for prediction after retraining for {self.stock_symbol}")
                                    return None

                                # If we're using feature importance, we need to recalculate it after retraining
                                if use_feature_importance:
                                    try:
                                        logger.info(f"Recalculating feature importance after retraining signal model for {self.stock_symbol}")
                                        feature_importance_results = self.analyze_feature_importance(
                                            threshold=feature_importance_threshold,
                                            remove_least_important=True
                                        )

                                        if feature_importance_results and 'feature_indices_to_keep' in feature_importance_results:
                                            feature_indices_to_keep = feature_importance_results['feature_indices_to_keep']
                                            logger.info(f"Using {len(feature_indices_to_keep)}/{X.shape[1]} features for signal prediction after retraining")

                                            # Filter features based on importance, but ensure we have at least some features
                                            if len(feature_indices_to_keep) > 0:
                                                X = X[:, feature_indices_to_keep]
                                                logger.info(f"Reduced feature set shape after retraining: {X.shape}")
                                            else:
                                                logger.warning(f"No features deemed important enough for signal model after retraining! Using all features instead.")
                                                feature_indices_to_keep = list(range(X.shape[1]))
                                    except Exception as e:
                                        logger.error(f"Error in feature importance analysis after retraining signal model: {str(e)}")
                                        # Continue with all features if there's an error
                                        feature_indices_to_keep = None

                                latest_features = X[-1].reshape(1, -1)
                                latest_features_signal = latest_features
                else:
                    latest_features_signal = latest_features

                try:
                    print(f"DEBUG: About to predict with signal model, feature shape: {latest_features_signal.shape}")

                    # Use calibrated predictions with confidence intervals
                    signal_pred, confidence_intervals, calibrated_probs = self.calibrate_predictions(latest_features_signal, model_type='signal')

                    if calibrated_probs is not None:
                        # Use calibrated probabilities
                        probas = calibrated_probs
                        confidence = float(max(probas[0]))

                        # Get the predicted class
                        predicted_class = int(signal_pred[0])

                        # Store calibration information
                        signal_calibration = {
                            'predicted_class': predicted_class,
                            'class_probabilities': {i: float(p) for i, p in enumerate(probas[0])},
                        }

                        # Add confidence intervals if available
                        if confidence_intervals is not None:
                            signal_calibration['confidence_intervals'] = {}
                            for i, interval in enumerate(confidence_intervals[0]):
                                signal_calibration['confidence_intervals'][i] = {
                                    'lower': float(interval[0]),
                                    'upper': float(interval[1])
                                }
                    else:
                        # Fallback to standard prediction
                        probas = self.signal_model.predict_proba(latest_features_signal)
                        confidence = float(max(probas[0]))
                        signal_calibration = None

                    print(f"DEBUG: Signal prediction successful, confidence: {confidence}")
                    print(f"DEBUG: Signal calibration: {signal_calibration}")
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error in signal prediction: {str(e)}")
                    print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
                    # Fallback to standard prediction if calibration fails
                    try:
                        probas = self.signal_model.predict_proba(latest_features_signal)
                        confidence = float(max(probas[0]))
                        signal_calibration = None
                        print(f"DEBUG: Fallback signal prediction successful, confidence: {confidence}")
                    except Exception as e2:
                        print(f"DEBUG: Fallback signal prediction also failed: {str(e2)}")
                        raise

            # Aktueller Kurs
            stock = Stock.objects.get(symbol=self.stock_symbol)
            current_price_obj = StockData.objects.filter(stock=stock).order_by('-date').first().close_price
            current_price = float(current_price_obj)

            # Vorhergesagter Kurs
            predicted_price = current_price * (1 + predicted_return)

            # Adaptive Schwellenwerte basierend auf Volatilität
            # ATR-basierte Volatilitätsberechnung für adaptive Schwellenwerte
            volatility_info = {}
            try:
                # Letzten 20 Tage für Volatilitätsberechnung verwenden
                recent_data = StockData.objects.filter(stock=stock).order_by('-date')[:20]

                if recent_data.count() >= 14:  # Mindestens 14 Tage für ATR-Berechnung
                    df_vol = pd.DataFrame(list(recent_data.values()))
                    df_vol['close_price'] = df_vol['close_price'].astype(float)
                    df_vol['high_price'] = df_vol['high_price'].astype(float)
                    df_vol['low_price'] = df_vol['low_price'].astype(float)

                    # ATR-Berechnung
                    high_low = df_vol['high_price'] - df_vol['low_price']
                    high_close = abs(df_vol['high_price'] - df_vol['close_price'].shift())
                    low_close = abs(df_vol['low_price'] - df_vol['close_price'].shift())

                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)

                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    last_close = df_vol['close_price'].iloc[0]  # Neuester Schlusskurs
                    atr_pct = (atr / last_close) * 100 if last_close != 0 else 5

                    # Adaptive Schwellenwerte basierend auf ATR
                    # Höhere Volatilität = höhere Schwellenwerte
                    base_threshold = 0.02  # 2% Basis-Schwellenwert

                    # Volatilitätsbasierte Anpassung
                    volatility_category = ""
                    if atr_pct < 1.5:  # Niedrige Volatilität
                        volatility_factor = 0.8
                        volatility_category = "niedrig"
                        logger.info(f"Niedrige Volatilität für {self.stock_symbol}: {atr_pct:.2f}% ATR")
                    elif atr_pct < 3.0:  # Mittlere Volatilität
                        volatility_factor = 1.0
                        volatility_category = "mittel"
                        logger.info(f"Mittlere Volatilität für {self.stock_symbol}: {atr_pct:.2f}% ATR")
                    elif atr_pct < 5.0:  # Hohe Volatilität
                        volatility_factor = 1.5
                        volatility_category = "hoch"
                        logger.info(f"Hohe Volatilität für {self.stock_symbol}: {atr_pct:.2f}% ATR")
                    else:  # Sehr hohe Volatilität
                        volatility_factor = 2.0
                        volatility_category = "sehr hoch"
                        logger.info(f"Sehr hohe Volatilität für {self.stock_symbol}: {atr_pct:.2f}% ATR")

                    min_return_for_buy = base_threshold * volatility_factor
                    min_return_for_sell = -base_threshold * volatility_factor

                    # Store volatility information for frontend display
                    volatility_info = {
                        'atr': float(atr),
                        'atr_pct': float(atr_pct),
                        'volatility_category': volatility_category,
                        'volatility_factor': float(volatility_factor),
                        'base_threshold': float(base_threshold),
                        'buy_threshold': float(min_return_for_buy),
                        'sell_threshold': float(min_return_for_sell)
                    }

                    logger.info(f"Adaptive Schwellenwerte für {self.stock_symbol}: Buy={min_return_for_buy:.2%}, Sell={min_return_for_sell:.2%}")
                else:
                    # Fallback auf Standardwerte, wenn nicht genug Daten
                    min_return_for_buy = 0.02
                    min_return_for_sell = -0.02
                    volatility_info = {
                        'atr': None,
                        'atr_pct': None,
                        'volatility_category': "standard",
                        'volatility_factor': 1.0,
                        'base_threshold': 0.02,
                        'buy_threshold': float(min_return_for_buy),
                        'sell_threshold': float(min_return_for_sell)
                    }
                    logger.warning(f"Nicht genug Daten für adaptive Schwellenwerte bei {self.stock_symbol}, verwende Standardwerte")
            except Exception as e:
                # Fallback bei Fehlern
                min_return_for_buy = 0.02
                min_return_for_sell = -0.02
                volatility_info = {
                    'atr': None,
                    'atr_pct': None,
                    'volatility_category': "standard (Fehler)",
                    'volatility_factor': 1.0,
                    'base_threshold': 0.02,
                    'buy_threshold': float(min_return_for_buy),
                    'sell_threshold': float(min_return_for_sell)
                }
                logger.error(f"Fehler bei Berechnung adaptiver Schwellenwerte für {self.stock_symbol}: {str(e)}")

            recommendation = 'HOLD'  # Default

            if predicted_return >= min_return_for_buy:
                recommendation = 'BUY'
            elif predicted_return <= min_return_for_sell:
                recommendation = 'SELL'

            # Zusätzlich: Sicherheit bei Penny Stocks
            if current_price < 1.0 and abs(predicted_return) < 0.05:
                recommendation = 'HOLD'

            # Prepare feature importance summary for the result
            feature_importance_summary = None
            if feature_importance_results:
                # Get the reduction stats
                reduction_stats = feature_importance_results.get('feature_reduction', {})

                # Get the top price and signal features
                top_price_features = dict(list(feature_importance_results.get('important_features', {}).get('price', []))[:5])
                top_signal_features = dict(list(feature_importance_results.get('important_features', {}).get('signal', []))[:5])

                # If no important features were found, get the top features by importance
                if not top_price_features and 'price_importance' in feature_importance_results:
                    price_importance = feature_importance_results['price_importance']
                    top_price_features = dict(sorted(price_importance.items(), key=lambda x: x[1], reverse=True)[:5])

                if not top_signal_features and 'signal_importance' in feature_importance_results:
                    signal_importance = feature_importance_results['signal_importance']
                    top_signal_features = dict(sorted(signal_importance.items(), key=lambda x: x[1], reverse=True)[:5])

                # Create the summary
                feature_importance_summary = {
                    'reduction_stats': reduction_stats,
                    'top_price_features': top_price_features,
                    'top_signal_features': top_signal_features
                }

                # Log the summary for debugging
                logger.info(f"Feature importance summary for {self.stock_symbol}:")
                logger.info(f"  Reduction stats: {reduction_stats}")
                logger.info(f"  Top price features: {list(top_price_features.keys())}")
                logger.info(f"  Top signal features: {list(top_signal_features.keys())}")

            # Add uncertainty information to the result
            uncertainty_info = {}

            # Add price prediction uncertainty if available
            if 'prediction_uncertainty' in locals() and prediction_uncertainty is not None:
                uncertainty_info['price'] = {
                    'lower_bound': round(current_price * (1 + prediction_uncertainty['lower_bound']), 2),
                    'upper_bound': round(current_price * (1 + prediction_uncertainty['upper_bound']), 2),
                    'interval_width': round(prediction_uncertainty['interval_width'] * 100, 2),  # as percentage
                    'confidence_level': 95  # 95% confidence interval
                }

            # Add signal prediction calibration if available
            if 'signal_calibration' in locals() and signal_calibration is not None:
                uncertainty_info['signal'] = signal_calibration

                # Add visualization paths if they exist
                uncertainty_plot_path = os.path.join('static', 'images', 'uncertainty', f'{self.stock_symbol}_price_uncertainty.png')
                if os.path.exists(uncertainty_plot_path):
                    uncertainty_info['price_uncertainty_plot'] = f'/static/images/uncertainty/{self.stock_symbol}_price_uncertainty.png'

                calibration_plot_path = os.path.join('static', 'images', 'calibration', f'{self.stock_symbol}_calibration.png')
                if os.path.exists(calibration_plot_path):
                    uncertainty_info['calibration_plot'] = f'/static/images/calibration/{self.stock_symbol}_calibration.png'

            # Get regime visualization path if available
            regime_visualization_path = None
            if current_regime and 'visualization_path' in current_regime:
                regime_visualization_path = current_regime['visualization_path']

            # Get sentiment information if available
            sentiment_info = self._get_sentiment_info()

            result = {
                'stock_symbol': self.stock_symbol,
                'current_price': current_price,
                'predicted_return': round(predicted_return * 100, 2),  # in Prozent
                'predicted_price': round(predicted_price, 2),
                'recommendation': recommendation,
                'confidence': round(confidence, 2),
                'prediction_days': self.prediction_days,
                'adaptive_thresholds': volatility_info,
                'feature_importance': feature_importance_summary,
                'uncertainty': uncertainty_info,
                'market_regime': {
                    'regime_type': regime_type,
                    'regime_name': regime_name,
                    'visualization_path': regime_visualization_path,
                    'details': current_regime
                },
                'sentiment': sentiment_info
            }

            print(
                f"[MLPredictor] {self.stock_symbol}: pred_return={predicted_return:.4f}, confidence={confidence:.2f}, recommendation={recommendation}")

            self._save_prediction(result)

            return result

        except Exception as e:
            logger.error(f"Error making prediction for {self.stock_symbol}: {str(e)}")
            return None

    def perform_seasonal_decomposition(self, data=None):
        """
        Perform seasonal decomposition of time series data

        Args:
            data (pd.DataFrame, optional): Data to decompose. If None, uses stock data from database.

        Returns:
            dict: Dictionary containing trend, seasonal, and residual components
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise ImportError("statsmodels is not installed. Install it with: pip install statsmodels")

        # Get data if not provided
        if data is None:
            stock = Stock.objects.get(symbol=self.stock_symbol)
            data = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            # Ensure numeric conversion
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                data[col] = data[col].astype(float)

        # Convert to time series with date index
        if 'date' in data.columns:
            data = data.set_index('date')

        # Perform decomposition on close price
        if 'close_price' in data.columns:
            # Determine period based on data frequency
            if len(data) >= 365:  # Daily data with at least a year
                period = 252  # Approximate number of trading days in a year
            elif len(data) >= 30:  # Daily data with at least a month
                period = 21   # Approximate number of trading days in a month
            else:
                period = 7    # Weekly pattern

            # Perform decomposition
            decomposition = seasonal_decompose(
                data['close_price'], 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )

            # Create visualization
            plt.figure(figsize=(12, 10))

            plt.subplot(411)
            plt.plot(decomposition.observed)
            plt.title('Observed')
            plt.grid(True)

            plt.subplot(412)
            plt.plot(decomposition.trend)
            plt.title('Trend')
            plt.grid(True)

            plt.subplot(413)
            plt.plot(decomposition.seasonal)
            plt.title('Seasonal')
            plt.grid(True)

            plt.subplot(414)
            plt.plot(decomposition.resid)
            plt.title('Residual')
            plt.grid(True)

            plt.tight_layout()

            # Save the plot
            decomposition_plot_path = os.path.join('static', 'images', 'decomposition')
            os.makedirs(decomposition_plot_path, exist_ok=True)
            plt.savefig(os.path.join(decomposition_plot_path, f'{self.stock_symbol}_decomposition.png'))
            plt.close()

            return {
                'observed': decomposition.observed.values,
                'trend': decomposition.trend.values,
                'seasonal': decomposition.seasonal.values,
                'residual': decomposition.resid.values,
                'dates': decomposition.observed.index.values,
                'visualization_path': f'/static/images/decomposition/{self.stock_symbol}_decomposition.png'
            }
        else:
            return None

    def analyze_trend(self, data=None):
        """
        Analyze trend strength and direction

        Args:
            data (pd.DataFrame, optional): Data to analyze. If None, uses stock data from database.

        Returns:
            dict: Dictionary containing trend analysis results
        """
        # Get data if not provided
        if data is None:
            stock = Stock.objects.get(symbol=self.stock_symbol)
            data = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            # Ensure numeric conversion
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                data[col] = data[col].astype(float)

        # Convert to time series with date index if needed
        if 'date' in data.columns:
            data = data.set_index('date')

        # Analyze trend using linear regression
        if 'close_price' in data.columns:
            # Add time index
            data['time_idx'] = np.arange(len(data))

            # Fit linear regression to detect overall trend
            trend_model = LinearRegression()
            X_trend = data['time_idx'].values.reshape(-1, 1)
            y_trend = data['close_price'].values
            trend_model.fit(X_trend, y_trend)

            # Calculate trend metrics
            trend_slope = trend_model.coef_[0]
            trend_direction = "upward" if trend_slope > 0 else "downward"
            trend_strength = abs(trend_slope) / data['close_price'].mean()

            # Calculate R-squared to measure trend strength
            y_pred = trend_model.predict(X_trend)
            r_squared = r2_score(y_trend, y_pred)

            # Analyze short-term vs long-term trends
            if len(data) >= 200:
                # Short-term trend (last 30 days)
                short_term_data = data.iloc[-30:]
                short_term_data['time_idx'] = np.arange(len(short_term_data))

                short_term_model = LinearRegression()
                X_short = short_term_data['time_idx'].values.reshape(-1, 1)
                y_short = short_term_data['close_price'].values
                short_term_model.fit(X_short, y_short)

                short_term_slope = short_term_model.coef_[0]
                short_term_direction = "upward" if short_term_slope > 0 else "downward"

                # Medium-term trend (last 90 days)
                medium_term_data = data.iloc[-90:]
                medium_term_data['time_idx'] = np.arange(len(medium_term_data))

                medium_term_model = LinearRegression()
                X_medium = medium_term_data['time_idx'].values.reshape(-1, 1)
                y_medium = medium_term_data['close_price'].values
                medium_term_model.fit(X_medium, y_medium)

                medium_term_slope = medium_term_model.coef_[0]
                medium_term_direction = "upward" if medium_term_slope > 0 else "downward"

                # Trend consistency
                trend_consistency = "consistent" if (
                    (trend_slope > 0 and short_term_slope > 0 and medium_term_slope > 0) or
                    (trend_slope < 0 and short_term_slope < 0 and medium_term_slope < 0)
                ) else "inconsistent"

                # Create visualization
                plt.figure(figsize=(12, 8))

                # Plot price and trends
                plt.plot(data.index, data['close_price'], label='Close Price')
                plt.plot(data.index, y_pred, 'r--', label='Long-term Trend')

                # Plot short and medium term trends
                short_term_pred = short_term_model.predict(X_short)
                medium_term_pred = medium_term_model.predict(X_medium)

                plt.plot(short_term_data.index, short_term_pred, 'g--', label='Short-term Trend (30 days)')
                plt.plot(medium_term_data.index, medium_term_pred, 'b--', label='Medium-term Trend (90 days)')

                plt.title(f'Trend Analysis for {self.stock_symbol}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)

                # Save the plot
                trend_plot_path = os.path.join('static', 'images', 'trends')
                os.makedirs(trend_plot_path, exist_ok=True)
                plt.savefig(os.path.join(trend_plot_path, f'{self.stock_symbol}_trend_analysis.png'))
                plt.close()

                return {
                    'overall_trend_direction': trend_direction,
                    'overall_trend_strength': float(trend_strength),
                    'overall_trend_slope': float(trend_slope),
                    'r_squared': float(r_squared),
                    'short_term_direction': short_term_direction,
                    'short_term_slope': float(short_term_slope),
                    'medium_term_direction': medium_term_direction,
                    'medium_term_slope': float(medium_term_slope),
                    'trend_consistency': trend_consistency,
                    'visualization_path': f'/static/images/trends/{self.stock_symbol}_trend_analysis.png'
                }
            else:
                # For shorter time series, just return the overall trend
                return {
                    'overall_trend_direction': trend_direction,
                    'overall_trend_strength': float(trend_strength),
                    'overall_trend_slope': float(trend_slope),
                    'r_squared': float(r_squared)
                }
        else:
            return None

    def predict_multiple_horizons(self, horizons=[5, 10, 30], use_feature_importance=True, feature_importance_threshold=0.01):
        """
        Make predictions for multiple time horizons

        Args:
            horizons (list): List of time horizons (in days) to predict
            use_feature_importance (bool): If True, analyze feature importance and use only important features
            feature_importance_threshold (float): Threshold for feature importance analysis

        Returns:
            dict: Dictionary of predictions for each horizon
        """
        results = {}

        for horizon in horizons:
            # Store original prediction days
            original_prediction_days = self.prediction_days

            # Set prediction days to current horizon
            self.prediction_days = horizon

            # Make prediction for this horizon
            prediction = self.predict(
                use_feature_importance=use_feature_importance,
                feature_importance_threshold=feature_importance_threshold
            )

            if prediction:
                results[horizon] = prediction

            # Restore original prediction days
            self.prediction_days = original_prediction_days

        return results

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

    def analyze_feature_importance_with_shap(self, model_type='signal', sample_size=100):
        """
        Analyze feature importance using SHAP (SHapley Additive exPlanations)

        Args:
            model_type (str): 'price' or 'signal'
            sample_size (int): Number of samples to use for SHAP analysis

        Returns:
            dict: Dictionary containing SHAP values and feature importance
        """
        try:
            X, y_price, y_signal = self.prepare_data()

            if X is None or len(X) == 0:
                logger.error(f"No data available for SHAP analysis for {self.stock_symbol}")
                return None

            # Select model and target
            if model_type == 'price':
                model = self.price_model
                y = y_price
            else:
                model = self.signal_model
                y = y_signal

            if model is None:
                logger.error(f"No {model_type} model available for SHAP analysis for {self.stock_symbol}")
                return None

            # Get feature names
            stock = Stock.objects.get(symbol=self.stock_symbol)
            df = pd.DataFrame(list(StockData.objects.filter(stock=stock).order_by('date').values()))

            # Ensure numeric conversion
            for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
                df[col] = df[col].astype(float)

            df = self._calculate_features(df)
            feature_columns = [col for col in df.columns if col not in
                              ['id', 'stock_id', 'date', 'future_return', 'signal_target']]

            # Limit sample size for performance
            if len(X) > sample_size:
                # Use stratified sampling for classification
                if model_type == 'signal':
                    _, X_sample, _, y_sample = train_test_split(
                        X, y, test_size=sample_size/len(X), 
                        stratify=y, random_state=42
                    )
                else:
                    # For regression, use random sampling
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X[indices]
                    y_sample = y.iloc[indices]
            else:
                X_sample = X
                y_sample = y

            # Create SHAP explainer based on model type
            try:
                # For tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    # For multi-class classification, shap_values is a list of arrays
                    if isinstance(shap_values, list) and model_type == 'signal':
                        # Get the mean absolute SHAP value for each feature across all classes
                        mean_abs_shap = np.mean([np.abs(shap_values[i]) for i in range(len(shap_values))], axis=0)
                        mean_abs_shap = np.mean(mean_abs_shap, axis=0)
                    else:
                        # For regression or binary classification
                        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                else:
                    # For other models, use KernelExplainer (slower but more general)
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                # Create feature importance dictionary
                feature_importance = {}
                for i, feature in enumerate(feature_columns[:len(mean_abs_shap)]):
                    feature_importance[feature] = float(mean_abs_shap[i])

                # Sort by importance
                feature_importance = {k: v for k, v in sorted(
                    feature_importance.items(), key=lambda item: item[1], reverse=True
                )}

                # Create visualization
                plt.figure(figsize=(10, 6))

                # Get top 20 features
                top_features = list(feature_importance.keys())[:20]
                top_importance = list(feature_importance.values())[:20]

                # Create bar plot
                plt.barh(range(len(top_features)), top_importance, align='center')
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('SHAP Value (Feature Importance)')
                plt.title(f'Top 20 Features by SHAP Value for {self.stock_symbol} ({model_type} model)')

                # Save the plot
                shap_plot_path = os.path.join('static', 'images', 'shap')
                os.makedirs(shap_plot_path, exist_ok=True)
                plt.savefig(os.path.join(shap_plot_path, f'{self.stock_symbol}_{model_type}_shap.png'))
                plt.close()

                # Create SHAP summary plot
                plt.figure(figsize=(10, 8))
                if isinstance(shap_values, list) and model_type == 'signal':
                    # For multi-class, use the first class (usually the positive class)
                    shap.summary_plot(
                        shap_values[1], 
                        X_sample, 
                        feature_names=feature_columns[:X_sample.shape[1]], 
                        show=False
                    )
                else:
                    shap.summary_plot(
                        shap_values, 
                        X_sample, 
                        feature_names=feature_columns[:X_sample.shape[1]], 
                        show=False
                    )

                # Save the summary plot
                plt.savefig(os.path.join(shap_plot_path, f'{self.stock_symbol}_{model_type}_shap_summary.png'))
                plt.close()

                # Create time-varying feature importance analysis
                if len(X) >= 60:  # Need enough data for time series analysis
                    # Split data into time windows
                    window_size = min(30, len(X) // 3)
                    n_windows = len(X) // window_size

                    time_varying_importance = {}

                    for i in range(n_windows):
                        start_idx = i * window_size
                        end_idx = (i + 1) * window_size

                        X_window = X[start_idx:end_idx]

                        # Calculate SHAP values for this window
                        window_shap_values = explainer.shap_values(X_window)

                        # Calculate mean absolute SHAP values
                        if isinstance(window_shap_values, list) and model_type == 'signal':
                            window_mean_abs_shap = np.mean([np.abs(window_shap_values[j]) for j in range(len(window_shap_values))], axis=0)
                            window_mean_abs_shap = np.mean(window_mean_abs_shap, axis=0)
                        else:
                            window_mean_abs_shap = np.mean(np.abs(window_shap_values), axis=0)

                        # Store importance for this window
                        window_importance = {}
                        for j, feature in enumerate(feature_columns[:len(window_mean_abs_shap)]):
                            window_importance[feature] = float(window_mean_abs_shap[j])

                        # Sort by importance
                        window_importance = {k: v for k, v in sorted(
                            window_importance.items(), key=lambda item: item[1], reverse=True
                        )}

                        # Store for this time window
                        time_varying_importance[f'window_{i+1}'] = window_importance

                    # Create visualization of time-varying importance
                    plt.figure(figsize=(12, 8))

                    # Get top 5 features across all windows
                    all_features = set()
                    for window in time_varying_importance.values():
                        all_features.update(list(window.keys())[:5])

                    top_features = list(all_features)[:10]  # Limit to 10 features

                    # Create line plot for each feature
                    for feature in top_features:
                        importance_values = []
                        for window in time_varying_importance.values():
                            importance_values.append(window.get(feature, 0))

                        plt.plot(range(1, n_windows + 1), importance_values, marker='o', label=feature)

                    plt.xlabel('Time Window')
                    plt.ylabel('SHAP Value (Feature Importance)')
                    plt.title(f'Time-Varying Feature Importance for {self.stock_symbol} ({model_type} model)')
                    plt.legend(loc='best')
                    plt.grid(True)

                    # Save the plot
                    plt.savefig(os.path.join(shap_plot_path, f'{self.stock_symbol}_{model_type}_time_varying_importance.png'))
                    plt.close()
                else:
                    time_varying_importance = None

                # Create feature attribution for individual predictions
                # Get the most recent data point for individual attribution
                latest_features = X[-1].reshape(1, -1)

                # Calculate SHAP values for this point
                individual_shap_values = explainer.shap_values(latest_features)

                # Create individual attribution dictionary
                individual_attribution = {}

                if isinstance(individual_shap_values, list) and model_type == 'signal':
                    # For multi-class, use the predicted class
                    pred_class = model.predict(latest_features)[0]
                    individual_values = individual_shap_values[pred_class][0]
                else:
                    individual_values = individual_shap_values[0]

                for i, feature in enumerate(feature_columns[:len(individual_values)]):
                    individual_attribution[feature] = float(individual_values[i])

                # Sort by absolute importance
                individual_attribution = {k: v for k, v in sorted(
                    individual_attribution.items(), key=lambda item: abs(item[1]), reverse=True
                )}

                # Create visualization of individual attribution
                plt.figure(figsize=(10, 6))

                # Get top 20 features
                top_features = list(individual_attribution.keys())[:20]
                top_values = [individual_attribution[f] for f in top_features]

                # Create bar plot with positive/negative coloring
                colors = ['red' if x < 0 else 'blue' for x in top_values]
                plt.barh(range(len(top_features)), top_values, align='center', color=colors)
                plt.yticks(range(len(top_features)), top_features)
                plt.xlabel('SHAP Value (Feature Attribution)')
                plt.title(f'Feature Attribution for Latest Prediction - {self.stock_symbol} ({model_type} model)')

                # Save the plot
                plt.savefig(os.path.join(shap_plot_path, f'{self.stock_symbol}_{model_type}_individual_attribution.png'))
                plt.close()

                # Create waterfall plot for individual prediction
                plt.figure(figsize=(10, 8))

                if isinstance(individual_shap_values, list) and model_type == 'signal':
                    # For multi-class, use the predicted class
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=individual_shap_values[pred_class][0], 
                            base_values=explainer.expected_value[pred_class],
                            data=latest_features[0],
                            feature_names=feature_columns[:latest_features.shape[1]]
                        ),
                        show=False
                    )
                else:
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=individual_shap_values[0], 
                            base_values=explainer.expected_value,
                            data=latest_features[0],
                            feature_names=feature_columns[:latest_features.shape[1]]
                        ),
                        show=False
                    )

                # Save the waterfall plot
                plt.savefig(os.path.join(shap_plot_path, f'{self.stock_symbol}_{model_type}_waterfall.png'))
                plt.close()

                # Return results
                return {
                    'feature_importance': feature_importance,
                    'time_varying_importance': time_varying_importance,
                    'individual_attribution': individual_attribution,
                    'visualization_paths': {
                        'importance_plot': f'/static/images/shap/{self.stock_symbol}_{model_type}_shap.png',
                        'summary_plot': f'/static/images/shap/{self.stock_symbol}_{model_type}_shap_summary.png',
                        'time_varying_plot': f'/static/images/shap/{self.stock_symbol}_{model_type}_time_varying_importance.png',
                        'individual_attribution_plot': f'/static/images/shap/{self.stock_symbol}_{model_type}_individual_attribution.png',
                        'waterfall_plot': f'/static/images/shap/{self.stock_symbol}_{model_type}_waterfall.png'
                    }
                }

            except Exception as e:
                logger.error(f"Error in SHAP analysis: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in analyze_feature_importance_with_shap for {self.stock_symbol}: {str(e)}")
            return None

    def analyze_feature_importance(self, threshold=0.01, remove_least_important=True):
        """
        Analyze feature importance using permutation importance and optionally remove least important features

        Args:
            threshold (float): Features with importance below this threshold will be considered unimportant
            remove_least_important (bool): If True, remove least important features from the model

        Returns:
            dict: Dictionary containing feature importance scores and lists of important/unimportant features
        """
        try:
            X, y_price, y_signal = self.prepare_data()

            if X is None or y_price is None or y_signal is None:
                logger.error(f"No data available for feature importance analysis for {self.stock_symbol}")
                return None

            # Get feature names
            feature_columns = [col for col in pd.DataFrame(list(StockData.objects.filter(
                stock=Stock.objects.get(symbol=self.stock_symbol)
            ).values())).columns if col not in ['id', 'stock_id', 'date']]

            # Additional features from _calculate_features method
            additional_features = ['hl_ratio', 'co_ratio', 'candle_body', 'upper_shadow', 'lower_shadow', 
                                  'is_bullish', 'is_doji', 'daily_return', 'rsi', 'macd', 'macd_signal',
                                  'momentum_5', 'momentum_10', 'momentum_20', 'ma_5', 'ma_10', 'ma_20',
                                  'ma_50', 'ma_200', 'bb_width', 'bb_position', 'volatility_20']

            # Combine all possible features
            all_features = feature_columns + additional_features

            # Filter to only include features that exist in the data
            X_df = pd.DataFrame(X)
            if X_df.shape[1] <= len(all_features):
                feature_names = all_features[:X_df.shape[1]]
            else:
                # If we have more columns than expected, use more descriptive names
                # First, use all the known feature names we have
                feature_names = all_features.copy()

                # Then add generic names for the remaining features, but with better context
                remaining_count = X_df.shape[1] - len(feature_names)
                for i in range(remaining_count):
                    feature_names.append(f"calculated_feature_{i+1}")

                # Log this situation for debugging
                logger.info(f"More features than expected for {self.stock_symbol}: {X_df.shape[1]} vs {len(all_features)}")

            results = {}
            important_features = {'price': [], 'signal': []}
            unimportant_features = {'price': [], 'signal': []}
            feature_indices_to_keep = set(range(X.shape[1]))  # Start with all features

            # Analyze price model importance
            if self.price_model is not None:
                try:
                    # Use permutation importance for more reliable results
                    price_importance = permutation_importance(
                        self.price_model, X, y_price, 
                        n_repeats=10, 
                        random_state=42,
                        n_jobs=-1
                    )

                    # Get sorted indices by importance
                    sorted_idx = np.argsort(price_importance.importances_mean)[::-1]

                    # Create sorted importance dictionary
                    price_feature_importance = {}
                    for i in sorted_idx:
                        if i < len(feature_names):
                            feature_name = feature_names[i]
                            importance = float(price_importance.importances_mean[i])
                            price_feature_importance[feature_name] = importance

                            # Classify as important or unimportant
                            if importance >= threshold:
                                important_features['price'].append((feature_name, importance))
                            else:
                                unimportant_features['price'].append((feature_name, importance))
                                # Mark for removal if below threshold
                                if remove_least_important and i in feature_indices_to_keep:
                                    feature_indices_to_keep.remove(i)

                    results['price_importance'] = price_feature_importance
                    logger.info(f"Top 5 features for price prediction: {list(price_feature_importance.items())[:5]}")
                    logger.info(f"Found {len(unimportant_features['price'])} unimportant features for price prediction")
                except Exception as e:
                    logger.error(f"Error calculating price model importance: {str(e)}")

            # Analyze signal model importance
            if self.signal_model is not None:
                try:
                    # Use permutation importance for more reliable results
                    signal_importance = permutation_importance(
                        self.signal_model, X, y_signal, 
                        n_repeats=10, 
                        random_state=42,
                        n_jobs=-1
                    )

                    # Get sorted indices by importance
                    sorted_idx = np.argsort(signal_importance.importances_mean)[::-1]

                    # Create sorted importance dictionary
                    signal_feature_importance = {}
                    for i in sorted_idx:
                        if i < len(feature_names):
                            feature_name = feature_names[i]
                            importance = float(signal_importance.importances_mean[i])
                            signal_feature_importance[feature_name] = importance

                            # Classify as important or unimportant
                            if importance >= threshold:
                                important_features['signal'].append((feature_name, importance))
                            else:
                                unimportant_features['signal'].append((feature_name, importance))
                                # Mark for removal if below threshold
                                if remove_least_important and i in feature_indices_to_keep:
                                    feature_indices_to_keep.remove(i)

                    results['signal_importance'] = signal_feature_importance
                    logger.info(f"Top 5 features for signal prediction: {list(signal_feature_importance.items())[:5]}")
                    logger.info(f"Found {len(unimportant_features['signal'])} unimportant features for signal prediction")
                except Exception as e:
                    logger.error(f"Error calculating signal model importance: {str(e)}")

            # Add important and unimportant features to results
            results['important_features'] = important_features
            results['unimportant_features'] = unimportant_features

            # Convert set to sorted list for consistent results
            feature_indices_to_keep = sorted(list(feature_indices_to_keep))

            # Ensure we always keep at least some features, even if they're below the threshold
            if len(feature_indices_to_keep) == 0:
                logger.warning(f"No features deemed important enough for {self.stock_symbol}! Keeping top features instead.")

                # Keep at least the top 5 features from price model
                if self.price_model is not None:
                    try:
                        # Get the top 5 features by importance
                        sorted_indices = np.argsort([-importance for importance in price_feature_importance.values()])[:5]
                        top_features = [list(price_feature_importance.keys())[i] for i in sorted_indices]

                        # Convert feature names back to indices
                        top_price_indices = [feature_names.index(feature) for feature in top_features 
                                            if feature in feature_names]

                        feature_indices_to_keep.extend(top_price_indices)
                        logger.info(f"Keeping top price features: {top_features}")
                    except Exception as e:
                        logger.error(f"Error selecting top price features: {str(e)}")

                # Keep at least the top 5 features from signal model
                if self.signal_model is not None:
                    try:
                        # Get the top 5 features by importance
                        sorted_indices = np.argsort([-importance for importance in signal_feature_importance.values()])[:5]
                        top_features = [list(signal_feature_importance.keys())[i] for i in sorted_indices]

                        # Convert feature names back to indices
                        top_signal_indices = [feature_names.index(feature) for feature in top_features 
                                             if feature in feature_names]

                        feature_indices_to_keep.extend(top_signal_indices)
                        logger.info(f"Keeping top signal features: {top_features}")
                    except Exception as e:
                        logger.error(f"Error selecting top signal features: {str(e)}")

                # If still empty, just keep the first 10 features
                if len(feature_indices_to_keep) == 0:
                    feature_indices_to_keep = list(range(min(10, X.shape[1])))

                # Remove duplicates and sort
                feature_indices_to_keep = sorted(list(set(feature_indices_to_keep)))

                logger.info(f"Keeping {len(feature_indices_to_keep)} top features for {self.stock_symbol}")

            results['feature_indices_to_keep'] = feature_indices_to_keep

            # Calculate how many features were removed
            total_features = X.shape[1]
            kept_features = len(feature_indices_to_keep)
            removed_features = total_features - kept_features
            removal_percentage = (removed_features / total_features) * 100 if total_features > 0 else 0

            results['feature_reduction'] = {
                'total_features': total_features,
                'kept_features': kept_features,
                'removed_features': removed_features,
                'removal_percentage': removal_percentage
            }

            logger.info(f"Feature reduction: Removed {removed_features}/{total_features} features ({removal_percentage:.1f}%)")

            # Store feature importance in MLModelMetrics
            try:
                stock = Stock.objects.get(symbol=self.stock_symbol)

                # Combine both importance dictionaries for storage
                combined_importance = {
                    'price': dict(important_features['price'] + unimportant_features['price']),
                    'signal': dict(important_features['signal'] + unimportant_features['signal']),
                    'reduction_stats': results['feature_reduction']
                }

                # Get or create the metrics object first
                metrics, created = MLModelMetrics.objects.get_or_create(
                    stock=stock,
                    date=datetime.now().date(),
                    defaults={
                        'accuracy': 0,
                        'rmse': 0,
                        'feature_importance': combined_importance,
                        'directional_accuracy': 0
                    }
                )

                # If the object already exists, just update the feature_importance field
                if not created:
                    metrics.feature_importance = combined_importance
                    metrics.save(update_fields=['feature_importance'])

                logger.info(f"Stored feature importance in MLModelMetrics for {self.stock_symbol}")
            except Exception as e:
                logger.error(f"Error storing feature importance: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Error analyzing feature importance for {self.stock_symbol}: {str(e)}")
            return None

    def evaluate_model_performance(self):
        """
        Evaluate the performance of the models on historical data with comprehensive metrics
        and statistical significance testing
        """
        try:
            X, y_price, y_signal = self.prepare_data()

            if X is None or y_price is None or y_signal is None:
                return None

            # Get appropriate cross-validation strategy for time series
            price_cv = self.get_time_series_cv(y_price, is_classification=False)
            signal_cv = self.get_time_series_cv(y_signal, is_classification=True)

            # Initialize performance dictionary
            performance = {
                'price_metrics': {},
                'signal_metrics': {},
                'cross_validation': {},
                'statistical_tests': {}
            }

            # Evaluate price prediction model with cross-validation
            if self.price_model is not None and price_cv is not None:
                # If price_cv is a list of tuples, it's our custom CV
                if isinstance(price_cv, list):
                    cv_splits = price_cv
                else:
                    # Otherwise use the sklearn CV object
                    cv_splits = list(price_cv.split(X))

                # Cross-validation metrics
                cv_rmse = []
                cv_mae = []
                cv_r2 = []
                cv_mape = []
                cv_direction_accuracy = []

                # For confidence intervals
                all_residuals = []

                for train_idx, test_idx in cv_splits:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_price.iloc[train_idx], y_price.iloc[test_idx]

                    # Get predictions
                    preds = self.price_model.predict(X_test)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)

                    # Calculate MAPE with handling for zeros
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mape = mean_absolute_percentage_error(y_test, preds)

                    # Calculate directional accuracy
                    direction_actual = np.sign(y_test)
                    direction_pred = np.sign(preds)
                    direction_accuracy = np.mean(direction_actual == direction_pred)

                    # Store metrics
                    cv_rmse.append(rmse)
                    cv_mae.append(mae)
                    cv_r2.append(r2)
                    cv_mape.append(mape)
                    cv_direction_accuracy.append(direction_accuracy)

                    # Store residuals for confidence intervals
                    all_residuals.extend(y_test - preds)

                # Calculate mean and std of metrics across folds
                performance['cross_validation']['price'] = {
                    'rmse': {
                        'mean': float(np.mean(cv_rmse)),
                        'std': float(np.std(cv_rmse)),
                        'values': [float(x) for x in cv_rmse]
                    },
                    'mae': {
                        'mean': float(np.mean(cv_mae)),
                        'std': float(np.std(cv_mae)),
                        'values': [float(x) for x in cv_mae]
                    },
                    'r2': {
                        'mean': float(np.mean(cv_r2)),
                        'std': float(np.std(cv_r2)),
                        'values': [float(x) for x in cv_r2]
                    },
                    'mape': {
                        'mean': float(np.mean(cv_mape)),
                        'std': float(np.std(cv_mape)),
                        'values': [float(x) for x in cv_mape]
                    },
                    'direction_accuracy': {
                        'mean': float(np.mean(cv_direction_accuracy)),
                        'std': float(np.std(cv_direction_accuracy)),
                        'values': [float(x) for x in cv_direction_accuracy]
                    }
                }

                # Calculate confidence intervals for predictions
                # Using the distribution of residuals
                residuals_std = np.std(all_residuals)
                performance['price_metrics']['prediction_uncertainty'] = {
                    'residuals_std': float(residuals_std),
                    '95_percent_confidence_interval': float(1.96 * residuals_std)
                }

                # Final evaluation on test set (last 20% of data)
                test_size = int(0.2 * len(X))
                X_test = X[-test_size:]
                y_price_test = y_price.iloc[-test_size:].to_numpy()

                price_pred = self.price_model.predict(X_test)
                price_mse = mean_squared_error(y_price_test, price_pred)
                price_rmse = np.sqrt(price_mse)
                price_mae = mean_absolute_error(y_price_test, price_pred)
                price_r2 = r2_score(y_price_test, price_pred)

                # Calculate directional accuracy
                direction_actual = np.sign(y_price_test)
                direction_pred = np.sign(price_pred)
                direction_accuracy = np.mean(direction_actual == direction_pred)

                # Store final metrics
                performance['price_metrics'] = {
                    'mse': float(price_mse),
                    'rmse': float(price_rmse),
                    'mae': float(price_mae),
                    'r2': float(price_r2),
                    'direction_accuracy': float(direction_accuracy)
                }

                # Compare with a baseline model (e.g., linear regression)
                try:
                    baseline_model = LinearRegression()
                    baseline_model.fit(X[:-test_size], y_price.iloc[:-test_size])
                    baseline_pred = baseline_model.predict(X_test)

                    # Statistical significance test
                    significance_results = self.statistical_significance_test(
                        price_pred, baseline_pred, y_price_test, model_type='price'
                    )
                    performance['statistical_tests']['price'] = significance_results

                    logger.info(f"Price model statistical comparison: "
                               f"p-value={significance_results.get('t_test', {}).get('p_value', 'N/A')}, "
                               f"better model={significance_results.get('t_test', {}).get('better_model', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error in price model statistical comparison: {str(e)}")

            # Evaluate signal prediction model with cross-validation
            if self.signal_model is not None and signal_cv is not None:
                # If signal_cv is a list of tuples, it's our custom CV
                if isinstance(signal_cv, list):
                    cv_splits = signal_cv
                else:
                    # Otherwise use the sklearn CV object
                    cv_splits = list(signal_cv.split(X, y_signal))

                # Cross-validation metrics
                cv_accuracy = []
                cv_precision = []
                cv_recall = []
                cv_f1 = []
                cv_auc = []

                # For calibration assessment
                all_probs = []
                all_true = []

                for train_idx, test_idx in cv_splits:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_signal.iloc[train_idx], y_signal.iloc[test_idx]

                    # Get predictions
                    preds = self.signal_model.predict(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, preds)
                    precision = precision_score(y_test, preds, average='weighted')
                    recall = recall_score(y_test, preds, average='weighted')
                    f1 = f1_score(y_test, preds, average='weighted')

                    # Calculate AUC if possible
                    auc_value = 0.5  # Default value
                    if hasattr(self.signal_model, 'predict_proba'):
                        try:
                            proba_preds = self.signal_model.predict_proba(X_test)
                            # Store for calibration assessment
                            all_probs.extend(proba_preds)
                            all_true.extend(y_test)

                            # For binary classification
                            if proba_preds.shape[1] == 2:
                                auc_value = roc_auc_score(y_test, proba_preds[:, 1])
                            else:
                                # For multiclass, use one-vs-rest approach
                                auc_value = roc_auc_score(y_test, proba_preds, multi_class='ovr')
                        except Exception as e:
                            logger.warning(f"AUC calculation failed: {str(e)}")

                    # Store metrics
                    cv_accuracy.append(accuracy)
                    cv_precision.append(precision)
                    cv_recall.append(recall)
                    cv_f1.append(f1)
                    cv_auc.append(auc_value)

                # Calculate mean and std of metrics across folds
                performance['cross_validation']['signal'] = {
                    'accuracy': {
                        'mean': float(np.mean(cv_accuracy)),
                        'std': float(np.std(cv_accuracy)),
                        'values': [float(x) for x in cv_accuracy]
                    },
                    'precision': {
                        'mean': float(np.mean(cv_precision)),
                        'std': float(np.std(cv_precision)),
                        'values': [float(x) for x in cv_precision]
                    },
                    'recall': {
                        'mean': float(np.mean(cv_recall)),
                        'std': float(np.std(cv_recall)),
                        'values': [float(x) for x in cv_recall]
                    },
                    'f1': {
                        'mean': float(np.mean(cv_f1)),
                        'std': float(np.std(cv_f1)),
                        'values': [float(x) for x in cv_f1]
                    },
                    'auc': {
                        'mean': float(np.mean(cv_auc)),
                        'std': float(np.std(cv_auc)),
                        'values': [float(x) for x in cv_auc]
                    }
                }

                # Final evaluation on test set (last 20% of data)
                test_size = int(0.2 * len(X))
                X_test = X[-test_size:]
                y_signal_test = y_signal.iloc[-test_size:].to_numpy()

                signal_pred = self.signal_model.predict(X_test)
                signal_accuracy = accuracy_score(y_signal_test, signal_pred)
                signal_precision = precision_score(y_signal_test, signal_pred, average='weighted')
                signal_recall = recall_score(y_signal_test, signal_pred, average='weighted')
                signal_f1 = f1_score(y_signal_test, signal_pred, average='weighted')

                # Classification report
                report = classification_report(y_signal_test, signal_pred, output_dict=True)

                # Calculate AUC if possible
                signal_auc = 0.5  # Default value
                signal_calibration = {}

                if hasattr(self.signal_model, 'predict_proba'):
                    try:
                        proba_preds = self.signal_model.predict_proba(X_test)

                        # For binary classification
                        if proba_preds.shape[1] == 2:
                            signal_auc = roc_auc_score(y_signal_test, proba_preds[:, 1])

                            # Assess calibration (for binary classification)
                            # Calculate Brier score (lower is better)
                            from sklearn.metrics import brier_score_loss
                            brier_score = brier_score_loss(y_signal_test, proba_preds[:, 1])

                            # Calculate calibration curve
                            prob_true, prob_pred = calibration_curve(y_signal_test, proba_preds[:, 1], n_bins=5)

                            signal_calibration = {
                                'brier_score': float(brier_score),
                                'calibration_curve': {
                                    'prob_true': [float(x) for x in prob_true],
                                    'prob_pred': [float(x) for x in prob_pred]
                                }
                            }
                        else:
                            # For multiclass, use one-vs-rest approach
                            signal_auc = roc_auc_score(y_signal_test, proba_preds, multi_class='ovr')
                    except Exception as e:
                        logger.warning(f"AUC calculation failed: {str(e)}")

                # Store final metrics
                performance['signal_metrics'] = {
                    'accuracy': float(signal_accuracy),
                    'precision': float(signal_precision),
                    'recall': float(signal_recall),
                    'f1': float(signal_f1),
                    'auc': float(signal_auc),
                    'classification_report': report,
                    'calibration': signal_calibration
                }

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
                    performance['signal_metrics']['feature_importance'] = {k: v for k, v in sorted(
                        feature_importance.items(), key=lambda item: item[1], reverse=True
                    )}

                # Compare with a baseline model (e.g., dummy classifier)
                try:
                    from sklearn.dummy import DummyClassifier
                    baseline_model = DummyClassifier(strategy='stratified', random_state=42)
                    baseline_model.fit(X[:-test_size], y_signal.iloc[:-test_size])
                    baseline_pred = baseline_model.predict(X_test)

                    # Statistical significance test
                    significance_results = self.statistical_significance_test(
                        signal_pred, baseline_pred, y_signal_test, model_type='signal'
                    )
                    performance['statistical_tests']['signal'] = significance_results

                    logger.info(f"Signal model statistical comparison: "
                               f"p-value={significance_results.get('mcnemar_test', {}).get('p_value', 'N/A')}, "
                               f"better model={significance_results.get('mcnemar_test', {}).get('better_model', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error in signal model statistical comparison: {str(e)}")

            # Save metrics to database
            try:
                stock = Stock.objects.get(symbol=self.stock_symbol)

                # Extract metrics for database storage
                signal_metrics = performance.get('signal_metrics', {})
                price_metrics = performance.get('price_metrics', {})
                cross_val = performance.get('cross_validation', {})

                MLModelMetrics.objects.update_or_create(
                    stock=stock,
                    date=datetime.now().date(),
                    model_version='v2',  # Updated version for enhanced metrics
                    defaults={
                        'accuracy': signal_metrics.get('accuracy', 0),
                        'precision': signal_metrics.get('precision', 0),
                        'recall': signal_metrics.get('recall', 0),
                        'f1_score': signal_metrics.get('f1', 0),
                        'auc': signal_metrics.get('auc', 0),
                        'rmse': price_metrics.get('rmse', 0),
                        'mae': price_metrics.get('mae', 0),
                        'r2': price_metrics.get('r2', 0),
                        'feature_importance': signal_metrics.get('feature_importance', {}),
                        'confusion_matrix': signal_metrics.get('classification_report', {}),
                        'directional_accuracy': price_metrics.get('direction_accuracy', 0),
                        'cross_validation_metrics': {
                            'price': cross_val.get('price', {}),
                            'signal': cross_val.get('signal', {})
                        },
                        'statistical_tests': performance.get('statistical_tests', {}),
                        'calibration_metrics': signal_metrics.get('calibration', {})
                    }
                )
                logger.info(f"Enhanced ML metrics successfully saved for {self.stock_symbol}")
            except Exception as e:
                logger.error(f"Error saving enhanced ML metrics for {self.stock_symbol}: {str(e)}")

            return performance

        except Exception as e:
            logger.error(f"Error evaluating model performance for {self.stock_symbol}: {str(e)}")
            return None


class AdaptiveAnalyzer:
    """
    Analyzer that combines traditional technical analysis with ML predictions
    to create an adaptive scoring system
    """

    def __init__(self, stock_symbol, enable_ml=True):
        """Initialize the adaptive analyzer"""
        self.stock_symbol = stock_symbol
        self.ml_predictor = MLPredictor(stock_symbol) if enable_ml else None
        self.ta = None  # Wird später initialisiert
        self.df = None  # DataFrame für Live-Daten
        self.enable_ml = enable_ml

    def get_adaptive_score(self):
        """
        Calculate an adaptive technical score that combines traditional
        technical analysis with ML predictions
        """
        try:
            from .analysis import TechnicalAnalyzer

            # === Technische Analyse ausführen ===
            if self.ta is None:
                self.ta = TechnicalAnalyzer(self.stock_symbol)

                # Wenn ein DataFrame gesetzt wurde, dieses verwenden
                if self.df is not None:
                    print(f"[DEBUG] AdaptiveAnalyzer verwendet bereitgestelltes DataFrame mit {len(self.df)} Zeilen")
                    self.ta.df = self.df
                    self.ta.calculate_indicators()

            ta_result = self.ta.calculate_technical_score()
            ta_score = ta_result['score']
            ta_recommendation = ta_result['recommendation']

            print(f"[DEBUG] TA Score: {ta_score:.2f} | Empfehlung: {ta_recommendation}")

            # Check if ML analysis is enabled
            if not self.enable_ml or self.ml_predictor is None:
                print("[DEBUG] ML-Analyse deaktiviert – verwende nur TA")
                return ta_result

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

    def save_analysis_result(self, result=None):
        """Save the adaptive analysis result"""
        try:
            from .models import Stock

            # Use the provided result or get a new one if none was provided
            if result is None:
                result = self.get_adaptive_score()

            stock = Stock.objects.get(symbol=self.stock_symbol)

            # Immer das aktuelle Datum verwenden, um sicherzustellen, dass die "Letzte Analyse" aktualisiert wird
            from datetime import datetime
            date = datetime.now().date()
            print(f"[DEBUG] AdaptiveAnalyzer verwendet aktuelles Datum: {date}")

            # Werte aus dem DataFrame oder aus dem Ergebnis extrahieren
            if self.ta is not None and hasattr(self.ta, 'df') and self.ta.df is not None and not self.ta.df.empty and 'date' in self.ta.df.columns:
                # Sicherstellen, dass die Daten nach Datum sortiert sind
                self.ta.df = self.ta.df.sort_values(by='date')

                # Direkt die letzten Werte aus dem DataFrame extrahieren
                latest_row = self.ta.df.iloc[-1]

                # Debug-Ausgabe der letzten Werte
                print(f"Letzte Werte für {self.stock_symbol} (AdaptiveAnalyzer):")
                for col in ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200', 'bollinger_upper', 'bollinger_lower']:
                    if col in latest_row:
                        print(f"  {col}: {latest_row[col]}")

                # Verwende die letzten Werte aus dem DataFrame, falls verfügbar
                rsi_value = latest_row.get('rsi') if 'rsi' in latest_row else None
                macd_value = latest_row.get('macd') if 'macd' in latest_row else None
                macd_signal = latest_row.get('macd_signal') if 'macd_signal' in latest_row else None
                sma_20 = latest_row.get('sma_20') if 'sma_20' in latest_row else None
                sma_50 = latest_row.get('sma_50') if 'sma_50' in latest_row else None
                sma_200 = latest_row.get('sma_200') if 'sma_200' in latest_row else None
                bollinger_upper = latest_row.get('bollinger_upper') if 'bollinger_upper' in latest_row else None
                bollinger_lower = latest_row.get('bollinger_lower') if 'bollinger_lower' in latest_row else None
            else:
                # Fallback auf Details aus dem Ergebnis
                details = result.get('details', {})

                rsi_value = details.get('rsi')
                macd_value = details.get('macd')
                macd_signal = details.get('macd_signal')
                sma_20 = details.get('sma_20')
                sma_50 = details.get('sma_50')
                sma_200 = details.get('sma_200')
                bollinger_upper = details.get('bollinger_upper')
                bollinger_lower = details.get('bollinger_lower')

            # Save to AnalysisResult
            analysis_result, created = AnalysisResult.objects.update_or_create(
                stock=stock,
                date=date,
                defaults={
                    'technical_score': result['score'],
                    'recommendation': result['recommendation'],
                    'confluence_score': result.get('confluence_score'),
                    'rsi_value': rsi_value,
                    'macd_value': macd_value,
                    'macd_signal': macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'bollinger_upper': bollinger_upper,
                    'bollinger_lower': bollinger_lower
                }
            )

            print(f"Analyse-Ergebnis (AdaptiveAnalyzer) gespeichert mit RSI: {analysis_result.rsi_value}, MACD: {analysis_result.macd_value}")
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

    if symbols is None:
        stocks_with_data = StockData.objects.values('stock') \
            .annotate(data_count=Count('id')) \
            .filter(data_count__gte=200)
        stock_ids = [item['stock'] for item in stocks_with_data]
        symbols = Stock.objects.filter(id__in=stock_ids).exclude(symbol='SPY').values_list('symbol', flat=True)

    results = {}

    for symbol in symbols:
        if symbol == 'SPY':
            continue
        try:
            predictor = MLPredictor(symbol)

            if force_retrain:
                predictor._train_model('price')
                predictor._train_model('signal')

                # After retraining, we need to recalculate features to match the new models
                X, _, _ = predictor.prepare_data()
                if X is None or X.shape[0] == 0:
                    logger.error(f"No data available for prediction after retraining for {symbol}")
                    results[symbol] = {
                        'status': 'error',
                        'message': 'No data available after retraining'
                    }
                    continue

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
