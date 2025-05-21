# stock_analyzer/ml_models.py
import numpy as np
import pandas as pd
from django.db.models import Count
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.inspection import permutation_importance
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
                df[col] = df[col].astype(float)

            # SPY-Daten laden und mit df mergen
            spy_stock = Stock.objects.get(symbol='SPY')
            spy_data = StockData.objects.filter(
                stock=spy_stock,
                date__range=[start_date, end_date]
            ).values('date', 'close_price')

            spy_df = pd.DataFrame(list(spy_data)).rename(columns={'close_price': 'spy_close'})

            if not spy_df.empty:
                df = df.merge(spy_df, on='date', how='left')
                df['spy_close'] = df['spy_close'].astype(float)

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

        # 1. BASIS-FEATURES - immer berechnen
        # ------------------------------
        # Preis- und Kerzen-Features
        df_features['hl_ratio'] = df_features['high_price'] / df_features['low_price']
        df_features['co_ratio'] = df_features['close_price'] / df_features['open_price']
        df_features['candle_body'] = df_features['close_price'] - df_features['open_price']
        df_features['upper_shadow'] = df_features['high_price'] - np.maximum(df_features['close_price'],
                                                                             df_features['open_price'])
        df_features['lower_shadow'] = np.minimum(df_features['close_price'], df_features['open_price']) - df_features[
            'low_price']
        df_features['is_bullish'] = (df_features['close_price'] > df_features['open_price']).astype(int)
        df_features['is_doji'] = (abs(df_features['candle_body']) < 0.1 * (
                df_features['high_price'] - df_features['low_price'])).astype(int)

        # Heikin-Ashi Kerzen - reduzieren Rauschen und zeigen Trends deutlicher
        try:
            # Heikin-Ashi Berechnung
            ha_close = (df_features['open_price'] + df_features['high_price'] + 
                        df_features['low_price'] + df_features['close_price']) / 4

            # Ersten Wert für HA Open setzen
            ha_open = df_features['open_price'].copy()

            # Iterativ berechnen (erfordert Schleife)
            for i in range(1, len(df_features)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

            # HA High und Low
            ha_high = df_features[['high_price', 'open_price', 'close_price']].max(axis=1)
            ha_low = df_features[['low_price', 'open_price', 'close_price']].min(axis=1)

            # Als Features speichern
            df_features['ha_open'] = ha_open
            df_features['ha_close'] = ha_close
            df_features['ha_high'] = ha_high
            df_features['ha_low'] = ha_low

            # Heikin-Ashi Kerzen-Features
            df_features['ha_body'] = ha_close - ha_open
            df_features['ha_is_bullish'] = (ha_close > ha_open).astype(int)

            # Trendstärke basierend auf Heikin-Ashi
            # Mehrere aufeinanderfolgende gleichfarbige Kerzen deuten auf starken Trend hin
            df_features['ha_trend_strength'] = df_features['ha_is_bullish'].rolling(window=3).sum()
            df_features.loc[df_features['ha_trend_strength'] == 0, 'ha_trend_strength'] = -3  # Alle 3 bearish

            # Trendwechsel-Signale
            df_features['ha_trend_change'] = df_features['ha_is_bullish'].diff().fillna(0)
            df_features['ha_bullish_reversal'] = (df_features['ha_trend_change'] > 0).astype(int)
            df_features['ha_bearish_reversal'] = (df_features['ha_trend_change'] < 0).astype(int)

        except Exception as e:
            print(f"Fehler bei Heikin-Ashi-Berechnung: {str(e)}")

        # Kurzfristige Renditen - maximal 1-Tages-Lag
        df_features['daily_return'] = df_features['close_price'].pct_change(fill_method=None).fillna(0)

        # 2. KURZFRISTIGE FEATURES - mindestens 10 Datenpunkte
        # ------------------------------
        if can_calculate_short:
            df_features[f'ma_{SHORT_WINDOW}'] = df_features['close_price'].rolling(window=SHORT_WINDOW,
                                                                                   min_periods=3).mean()
            if 'ma_5' in df_features.columns:  # Sicherheitsprüfung
                df_features[f'ma_{SHORT_WINDOW}_dist'] = (df_features['close_price'] - df_features[
                    f'ma_{SHORT_WINDOW}']) / df_features[f'ma_{SHORT_WINDOW}'].replace(0, np.nan)

            df_features['volatility_5'] = df_features['daily_return'].rolling(window=SHORT_WINDOW, min_periods=3).std()
            df_features['volume_ma_5'] = df_features['volume'].rolling(window=SHORT_WINDOW, min_periods=3).mean()

            # EMA für MACD (kurze Komponente)
            df_features['ema_12'] = df_features['close_price'].ewm(span=12, min_periods=5, adjust=False).mean()

            # Kurzfristige Momentum-Features
            df_features[f'momentum_{SHORT_WINDOW}'] = df_features['close_price'] / df_features['close_price'].shift(
                SHORT_WINDOW).replace(0, np.nan) - 1

            # RSI mit kleinerem Fenster
            delta = df_features['close_price'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ema_up = up.ewm(com=SHORT_WINDOW, min_periods=3, adjust=False).mean()
            ema_down = down.ewm(com=SHORT_WINDOW, min_periods=3, adjust=False).mean()
            rs = ema_up / (ema_down.replace(0, np.nan))  # Vermeidung von Division durch Null
            df_features['rsi'] = 100 - (100 / (1 + rs))

            # Close-Lags für kurzfristige Vergleiche
            df_features['close_lag_1'] = df_features['close_price'].shift(1)

            # Preis-Geschwindigkeit über kurze Perioden
            df_features['price_velocity_3'] = df_features['close_price'].diff(min(3, data_length - 1)) / min(3,
                                                                                                             data_length - 1)

        # 3. MITTELFRISTIGE FEATURES - mindestens 15 Datenpunkte
        # ------------------------------
        if can_calculate_medium:
            df_features[f'ma_{MEDIUM_WINDOW}'] = df_features['close_price'].rolling(window=MEDIUM_WINDOW,
                                                                                    min_periods=5).mean()
            if 'ma_10' in df_features.columns:  # Sicherheitsprüfung
                df_features[f'ma_{MEDIUM_WINDOW}_dist'] = (df_features['close_price'] - df_features[
                    f'ma_{MEDIUM_WINDOW}']) / df_features[f'ma_{MEDIUM_WINDOW}'].replace(0, np.nan)

            # Mittelfristige Renditen
            df_features['weekly_return'] = df_features['close_price'].pct_change(min(5, data_length - 1), fill_method=None).fillna(0)

            # Mittelfristige Momentum-Features
            df_features[f'momentum_{MEDIUM_WINDOW}'] = df_features['close_price'] / df_features['close_price'].shift(
                MEDIUM_WINDOW).replace(0, np.nan) - 1

            # MACD-Komponenten (wenn möglich)
            if 'ema_12' in df_features.columns:
                df_features['ema_26'] = df_features['close_price'].ewm(span=26, min_periods=10, adjust=False).mean()
                df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
                df_features['macd_signal'] = df_features['macd'].ewm(span=9, min_periods=4, adjust=False).mean()
                df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

            # SMA-Kreuzungen (falls möglich)
            if 'ma_5' in df_features.columns and 'ma_10' in df_features.columns:
                df_features['ma_5_10_cross'] = df_features['ma_5'] - df_features['ma_10']

            # Weitere Close-Lags
            df_features['close_lag_2'] = df_features['close_price'].shift(2)
            if 'rsi' in df_features.columns:
                df_features['rsi_lag_1'] = df_features['rsi'].shift(1)

            # Mittelfristige Trendstärke
            df_features['trend_strength_10'] = df_features['close_price'].diff(min(MEDIUM_WINDOW, data_length - 1))

            # Beschleunigung der Rendite
            if 'daily_return' in df_features.columns:
                df_features['return_acceleration'] = df_features['daily_return'].diff()

        # 4. STANDARD-FEATURES - mindestens 25 Datenpunkte
        # ------------------------------
        if can_calculate_standard:
            df_features[f'ma_{STANDARD_WINDOW}'] = df_features['close_price'].rolling(window=STANDARD_WINDOW,
                                                                                      min_periods=10).mean()
            if 'ma_20' in df_features.columns:  # Sicherheitsprüfung
                df_features[f'ma_{STANDARD_WINDOW}_dist'] = (df_features['close_price'] - df_features[
                    f'ma_{STANDARD_WINDOW}']) / df_features[f'ma_{STANDARD_WINDOW}'].replace(0, np.nan)

            # Längerfristige Renditen
            df_features['monthly_return'] = df_features['close_price'].pct_change(min(STANDARD_WINDOW, data_length - 1), fill_method=None).fillna(0)

            # Volatilitätsfeatures
            df_features['volatility_20'] = df_features['daily_return'].rolling(window=STANDARD_WINDOW,
                                                                               min_periods=10).std()

            # Volume-Features
            df_features['volume_ma_20'] = df_features['volume'].rolling(window=STANDARD_WINDOW, min_periods=10).mean()
            if 'volume_ma_20' in df_features.columns and df_features['volume_ma_20'].max() > 0:
                df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_20'].replace(0, np.nan)

            # Bollinger Bands
            if 'ma_20' in df_features.columns:
                std = df_features['close_price'].rolling(window=STANDARD_WINDOW, min_periods=10).std()
                df_features['bb_middle'] = df_features['ma_20']
                df_features['bb_upper'] = df_features['bb_middle'] + 2 * std
                df_features['bb_lower'] = df_features['bb_middle'] - 2 * std

                # Vermeidung von Division durch Null
                bb_width_divisor = df_features['bb_middle'].replace(0, np.nan)
                df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / bb_width_divisor

                # BB Position nur berechnen, wenn nicht durch Null geteilt wird
                bb_denominator = (df_features['bb_upper'] - df_features['bb_lower'])
                bb_position_valid = bb_denominator > 0
                df_features['bb_position'] = np.nan  # Default-Wert
                if bb_position_valid.any():
                    df_features.loc[bb_position_valid, 'bb_position'] = (
                            (df_features.loc[bb_position_valid, 'close_price'] -
                             df_features.loc[bb_position_valid, 'bb_lower']) /
                            bb_denominator[bb_position_valid]
                    )
                # Fallback für ungültige Positionen
                df_features['bb_position'] = df_features['bb_position'].fillna(0.5)

            # Z-Score
            if 'ma_20' in df_features.columns:
                std20 = df_features['close_price'].rolling(window=STANDARD_WINDOW, min_periods=10).std()
                std20_valid = std20 > 0
                df_features['zscore_20'] = np.nan
                if std20_valid.any():
                    df_features.loc[std20_valid, 'zscore_20'] = (
                            (df_features.loc[std20_valid, 'close_price'] -
                             df_features.loc[std20_valid, 'ma_20']) /
                            std20[std20_valid]
                    )

            # Standardfeatures für längerfristige Momentum-Features
            df_features[f'momentum_{STANDARD_WINDOW}'] = df_features['close_price'] / df_features['close_price'].shift(
                STANDARD_WINDOW).replace(0, np.nan) - 1

            # Bullish Signal Count basierend auf verfügbaren Indikatoren
            bullish_indicators = []
            if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
                bullish_indicators.append((df_features['macd'] > df_features['macd_signal']).astype(int))
            if 'rsi' in df_features.columns:
                bullish_indicators.append((df_features['rsi'] < 30).astype(int))
            if 'ma_20' in df_features.columns:
                bullish_indicators.append((df_features['close_price'] > df_features['ma_20']).astype(int))

            if bullish_indicators:
                df_features['bullish_signals'] = sum(bullish_indicators)

            # Close Lag 3 für längere Trends
            df_features['close_lag_3'] = df_features['close_price'].shift(3)

            # MACD Lags für Trendveränderungen
            if 'macd' in df_features.columns:
                df_features['macd_lag_1'] = df_features['macd'].shift(1)

            # SMA-Kreuzungen und Veränderungen
            if 'ma_20' in df_features.columns and can_calculate_medium and 'ma_10' in df_features.columns:
                df_features['ma_10_20_cross'] = df_features['ma_10'] - df_features['ma_20']

            # Bullish Engulfing Muster
            if 'is_bullish' in df_features.columns and 'candle_body' in df_features.columns:
                df_features['is_bullish_engulfing'] = ((df_features['is_bullish'] == 1) &
                                                       (df_features['candle_body'] > df_features[
                                                           'candle_body'].shift())).astype(int)

            # ADX (Average Directional Index) - Trendstärke-Indikator
            try:
                # Berechnung der Richtungsbewegungen
                plus_dm = df_features['high_price'].diff()
                minus_dm = df_features['low_price'].diff(-1).abs()

                # Positive/Negative Richtungsbewegung
                plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
                minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

                # True Range
                high_low = df_features['high_price'] - df_features['low_price']
                high_close = (df_features['high_price'] - df_features['close_price'].shift()).abs()
                low_close = (df_features['low_price'] - df_features['close_price'].shift()).abs()
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
                    df_features['adx'] = dx.rolling(window=window, min_periods=5).mean()
                    df_features['+di'] = plus_di
                    df_features['-di'] = minus_di

                    # ADX Interpretation als Features
                    df_features['strong_trend'] = (df_features['adx'] > 25).astype(int)
                    df_features['very_strong_trend'] = (df_features['adx'] > 50).astype(int)
                    df_features['bullish_trend'] = ((df_features['+di'] > df_features['-di']) & 
                                                   (df_features['adx'] > 20)).astype(int)
                    df_features['bearish_trend'] = ((df_features['+di'] < df_features['-di']) & 
                                                   (df_features['adx'] > 20)).astype(int)
            except Exception as e:
                print(f"Fehler bei ADX-Berechnung: {str(e)}")

        # 5. SPY-KORRELATIONS-FEATURES - nur wenn SPY Daten vorhanden sind
        # ------------------------------
        if 'spy_close' in df_features.columns:
            # Grundlegende SPY-Renditen berechnen
            df_features['spy_daily_return'] = df_features['spy_close'].pct_change(fill_method=None).fillna(0)

            # Kurzfristige relative Stärke
            if 'daily_return' in df_features.columns:
                df_features['rel_strength_daily'] = df_features['daily_return'] - df_features['spy_daily_return']

            # Medium-Term SPY Features
            if can_calculate_medium:
                df_features['spy_return_10d'] = df_features['spy_close'].pct_change(MEDIUM_WINDOW, fill_method=None).fillna(0)
                if 'weekly_return' in df_features.columns:
                    df_features['rel_strength_10d'] = df_features['weekly_return'] - df_features['spy_return_10d']

            # Standard SPY Features
            if can_calculate_standard:
                # 20-Tage Korrelation - kritisch, dass genügend Datenpunkte vorhanden sind
                min_corr_periods = min(15, data_length - 5)
                if min_corr_periods >= 10:  # Mindestens 10 Punkte für sinnvolle Korrelation
                    df_features['corr_with_spy_20d'] = df_features['close_price'].rolling(STANDARD_WINDOW,
                                                                                          min_periods=min_corr_periods).corr(
                        df_features['spy_close'])

                # SPY und Stock Rolling Return
                min_periods_spy = min(15, data_length - 5)
                if can_calculate_standard and min_periods_spy >= 10:
                    spy_ret_window = min(30, data_length - 5)
                    df_features['spy_return_30d'] = df_features['spy_close'].pct_change(fill_method=None).fillna(0).rolling(window=spy_ret_window,
                                                                                                  min_periods=min_periods_spy).sum()
                    df_features['stock_return_30d'] = df_features['close_price'].pct_change(fill_method=None).fillna(0).rolling(
                        window=spy_ret_window, min_periods=min_periods_spy).sum()

                    # Alpha und Beta nur berechnen, wenn genügend nicht-NaN Werte vorhanden sind
                    df_features['rolling_alpha'] = np.nan
                    df_features['rolling_beta'] = np.nan

                    valid_indices = ~(df_features['spy_return_30d'].isna() | df_features['stock_return_30d'].isna())

                    # Nur Alpha/Beta für einzelne Datenpunkte berechnen, wenn genug Vergangenheitsdaten verfügbar sind
                    if valid_indices.sum() >= spy_ret_window:
                        window = min(30, valid_indices.sum() - 5)

                        for i in range(window, len(df_features)):
                            # Prüfe, ob genügend Daten für eine Regression vorhanden sind
                            valid_window = ~(df_features['spy_return_30d'].iloc[i - window:i].isna() |
                                             df_features['stock_return_30d'].iloc[i - window:i].isna())

                            if valid_window.sum() >= 10:  # Mindestens 10 Punkte für Regression
                                x = df_features['spy_return_30d'].iloc[i - window:i][valid_window].values.reshape(-1, 1)
                                y = df_features['stock_return_30d'].iloc[i - window:i][valid_window].values

                                try:
                                    from sklearn.linear_model import LinearRegression
                                    model = LinearRegression()
                                    model.fit(x, y)
                                    df_features.loc[df_features.index[i], 'rolling_alpha'] = model.intercept_
                                    df_features.loc[df_features.index[i], 'rolling_beta'] = model.coef_[0]
                                except:
                                    # Fallback bei Fehlern in der Regression
                                    pass

        # 6. ERWEITERTE FEATURES - nur wenn genügend Daten vorhanden sind (50+ Datenpunkte)
        # ------------------------------
        if can_calculate_extended:
            df_features[f'ma_{EXTENDED_WINDOW}'] = df_features['close_price'].rolling(window=EXTENDED_WINDOW,
                                                                                      min_periods=30).mean()

            # SMA Kreuzungen mit Extended Windows
            if 'ma_20' in df_features.columns and 'ma_50' in df_features.columns:
                df_features['ma_20_50_cross'] = np.sign(df_features['ma_20'] - df_features['ma_50'])
                df_features['sma_cross_change'] = df_features['ma_20_50_cross'].diff().fillna(0)
                df_features['sma_bullish_cross'] = (df_features['sma_cross_change'] > 0).astype(int)
                df_features['sma_bearish_cross'] = (df_features['sma_cross_change'] < 0).astype(int)

            # EMA Kreuzungen
            if 'ema_12' in df_features.columns and 'ema_26' in df_features.columns:
                df_features['ema_12_26_cross'] = np.sign(df_features['ema_12'] - df_features['ema_26'])
                df_features['ema_cross_change'] = df_features['ema_12_26_cross'].diff().fillna(0)
                df_features['ema_bullish_cross'] = (df_features['ema_cross_change'] > 0).astype(int)
                df_features['ema_bearish_cross'] = (df_features['ema_cross_change'] < 0).astype(int)

            # Ichimoku Cloud - umfassendes Indikator-System
            try:
                # Parameter für Ichimoku (angepasst an verfügbare Datenmenge)
                tenkan_period = min(9, data_length // 5)  # Conversion Line
                kijun_period = min(26, data_length // 3)  # Base Line
                senkou_b_period = min(52, data_length // 2)  # Leading Span B

                if tenkan_period >= 3 and kijun_period >= 5 and senkou_b_period >= 10:
                    # Tenkan-sen (Conversion Line): (n-period high + n-period low) / 2
                    period_high = df_features['high_price'].rolling(window=tenkan_period).max()
                    period_low = df_features['low_price'].rolling(window=tenkan_period).min()
                    df_features['tenkan_sen'] = (period_high + period_low) / 2

                    # Kijun-sen (Base Line): (n-period high + n-period low) / 2
                    period_high = df_features['high_price'].rolling(window=kijun_period).max()
                    period_low = df_features['low_price'].rolling(window=kijun_period).min()
                    df_features['kijun_sen'] = (period_high + period_low) / 2

                    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
                    df_features['senkou_span_a'] = ((df_features['tenkan_sen'] + df_features['kijun_sen']) / 2)

                    # Senkou Span B (Leading Span B): (n-period high + n-period low) / 2
                    period_high = df_features['high_price'].rolling(window=senkou_b_period).max()
                    period_low = df_features['low_price'].rolling(window=senkou_b_period).min()
                    df_features['senkou_span_b'] = (period_high + period_low) / 2

                    # Chikou Span (Lagging Span): Aktuelle Schlusskurse
                    df_features['chikou_span'] = df_features['close_price']

                    # Ichimoku-Signale als Features
                    # Preis über der Cloud (bullish)
                    df_features['price_above_cloud'] = (df_features['close_price'] > df_features['senkou_span_a']) & \
                                                      (df_features['close_price'] > df_features['senkou_span_b'])
                    df_features['price_above_cloud'] = df_features['price_above_cloud'].astype(int)

                    # Preis unter der Cloud (bearish)
                    df_features['price_below_cloud'] = (df_features['close_price'] < df_features['senkou_span_a']) & \
                                                      (df_features['close_price'] < df_features['senkou_span_b'])
                    df_features['price_below_cloud'] = df_features['price_below_cloud'].astype(int)

                    # Collect all new columns in a dictionary
                    new_columns = {
                        'price_in_cloud': (~(df_features['price_above_cloud'] | df_features['price_below_cloud'])).astype(int),

                        # TK Cross (Tenkan kreuzt Kijun) - starkes Signal
                        'tk_cross': np.sign(df_features['tenkan_sen'] - df_features['kijun_sen']),
                    }

                    # Add tk_cross_change after tk_cross is calculated
                    new_columns['tk_cross_change'] = new_columns['tk_cross'].diff().fillna(0)
                    new_columns['bullish_tk_cross'] = (new_columns['tk_cross_change'] > 0).astype(int)
                    new_columns['bearish_tk_cross'] = (new_columns['tk_cross_change'] < 0).astype(int)

                    # Kijun Cross (Preis kreuzt Kijun) - wichtiges Signal
                    new_columns['kijun_cross'] = np.sign(df_features['close_price'] - df_features['kijun_sen'])
                    new_columns['kijun_cross_change'] = new_columns['kijun_cross'].diff().fillna(0)
                    new_columns['bullish_kijun_cross'] = (new_columns['kijun_cross_change'] > 0).astype(int)
                    new_columns['bearish_kijun_cross'] = (new_columns['kijun_cross_change'] < 0).astype(int)

                    # Cloud-Twist (Senkou Span A kreuzt Senkou Span B) - langfristiges Signal
                    new_columns['cloud_twist'] = np.sign(df_features['senkou_span_a'] - df_features['senkou_span_b'])
                    new_columns['cloud_twist_change'] = new_columns['cloud_twist'].diff().fillna(0)
                    new_columns['bullish_cloud_twist'] = (new_columns['cloud_twist_change'] > 0).astype(int)
                    new_columns['bearish_cloud_twist'] = (new_columns['cloud_twist_change'] < 0).astype(int)

                    # Add all new columns at once
                    df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)
            except Exception as e:
                print(f"Fehler bei Ichimoku-Berechnung: {str(e)}")

        # 7. LANGFRISTIGE FEATURES - nur wenn genügend Daten vorhanden sind (200+ Datenpunkte)
        # ------------------------------
        if can_calculate_long:
            # Collect long-term features in a dictionary
            long_term_columns = {
                f'ma_{LONG_WINDOW}': df_features['close_price'].rolling(window=LONG_WINDOW, min_periods=120).mean()
            }

            # SMA Kreuzungen mit Long Windows
            if 'ma_50' in df_features.columns and 'ma_200' in df_features.columns:
                long_term_columns['ma_50_200_cross'] = df_features['ma_50'] - df_features['ma_200']

            # Add all long-term columns at once
            df_features = pd.concat([df_features, pd.DataFrame(long_term_columns, index=df_features.index)], axis=1)

        # Volatilitäts-Kategorisierung (nur wenn vorhanden)
        new_columns = {}

        if 'volatility_20' in df_features.columns and df_features['volatility_20'].count() >= 10:
            try:
                # Quantile-Berechnung könnte fehlschlagen bei zu wenigen Werten
                new_columns['volatility_category'] = pd.qcut(
                    df_features['volatility_20'].rank(method='first'),  # Ranking für gleiche Werte
                    q=3,
                    labels=[0, 1, 2]
                ).astype(int)
            except:
                # Fallback: Manuelle Kategorisierung
                if df_features['volatility_20'].max() > 0:
                    thresholds = [
                        df_features['volatility_20'].quantile(0.33, interpolation='nearest'),
                        df_features['volatility_20'].quantile(0.67, interpolation='nearest')
                    ]

                    # Create a Series with default value 1
                    volatility_category = pd.Series(1, index=df_features.index)
                    # Update values based on thresholds
                    volatility_category[df_features['volatility_20'] <= thresholds[0]] = 0
                    volatility_category[df_features['volatility_20'] > thresholds[1]] = 2
                    new_columns['volatility_category'] = volatility_category

        # Bei High Volatility Feature, ein einfacheres Fallback verwenden
        if 'volatility_20' in df_features.columns and df_features['volatility_20'].count() >= 5:
            # Verwende Median statt Quantil bei wenigen Datenpunkten
            vol_thresh = df_features['volatility_20'].median() * 1.5
            new_columns['high_volatility_flag'] = (df_features['volatility_20'] > vol_thresh).astype(int)

        # Add all new columns at once if there are any
        if new_columns:
            df_features = pd.concat([df_features, pd.DataFrame(new_columns, index=df_features.index)], axis=1)

        # Bereinigen der Daten
        df_features = df_features.replace([np.inf, -np.inf], np.nan)

        # Bestimme wichtige Feature-Spalten die komplett sein müssen (keine NaN erlaubt)
        critical_features = ['close_price', 'open_price', 'high_price', 'low_price']

        # Wichtige Features, die berechnet werden sollten, aber NaN haben dürfen
        useful_features = []

        if can_calculate_short:
            useful_features.extend(['daily_return', 'ma_5', 'ema_12', 'rsi'])

        if can_calculate_medium:
            useful_features.extend(['weekly_return', 'ma_10', 'macd', 'macd_signal'])

        if can_calculate_standard:
            useful_features.extend(['monthly_return', 'ma_20', 'bb_middle', 'bb_upper', 'bb_lower'])

        # Prüfe, ob kritische Features NaN-Werte enthalten
        missing_critical = [col for col in critical_features if
                            col in df_features.columns and df_features[col].isna().any()]
        if missing_critical:
            print(f"DEBUG: Kritische Features mit NaN: {missing_critical}")

        # NaN-Werte in nicht-kritischen Features mit Durchschnittswerten füllen
        for col in df_features.columns:
            if col not in critical_features and df_features[col].isna().any():
                # Für Features mit signifikanten NaN-Werten, aber ausreichend Nicht-NaN-Werten
                non_nan_count = df_features[col].count()
                if non_nan_count >= 5 and non_nan_count >= len(df_features) * 0.3:
                    df_features[col] = df_features[col].fillna(df_features[col].mean())

        # Fallback: Wenn nach all diesen Berechnungen immer noch alle Zeilen NaN-Werte enthalten
        original_len = len(df_features)
        df_no_nan = df_features.dropna()

        if len(df_no_nan) == 0:
            print(f"DEBUG: Nach allen Berechnungen immer noch keine vollständigen Zeilen! Führe Minimal-Fallback aus.")

            # Nur minimale kritische Features behalten
            min_features = critical_features.copy()
            if 'daily_return' in df_features.columns:
                min_features.append('daily_return')

            # Auch nützliche Features hinzufügen, die berechnet wurden
            available_useful = []
            for f in useful_features:
                if f in df_features.columns:
                    non_nan_ratio = df_features[f].count() / len(df_features)
                    if non_nan_ratio > 0.8:  # If at least 80% of values are not NaN
                        available_useful.append(f)

            min_features.extend(available_useful)

            # Reduziertes DataFrame erstellen
            available_min_features = [col for col in min_features if col in df_features.columns]
            if not available_min_features:
                # If no features are available, at least keep the price columns
                available_min_features = [col for col in ['open_price', 'high_price', 'low_price', 'close_price']
                                          if col in df_features.columns]

            df_minimal = df_features[available_min_features].copy()

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
                # XGBoost statt GradientBoostingRegressor für bessere Performance
                base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
                param_grid = {
                    'n_estimators': [100, 150, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 3, 5]
                }
                scoring = 'neg_mean_squared_error'
            else:
                model_path = self.signal_model_path
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
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
                    predicted_return = float(self.price_model.predict(latest_features)[0])
                    print(f"DEBUG: Price prediction successful: {predicted_return}")
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error in price prediction: {str(e)}")
                    print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
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
                    probas = self.signal_model.predict_proba(latest_features_signal)
                    confidence = float(max(probas[0]))
                    print(f"DEBUG: Signal prediction successful, confidence: {confidence}")
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error in signal prediction: {str(e)}")
                    print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
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

            result = {
                'stock_symbol': self.stock_symbol,
                'current_price': current_price,
                'predicted_return': round(predicted_return * 100, 2),  # in Prozent
                'predicted_price': round(predicted_price, 2),
                'recommendation': recommendation,
                'confidence': round(confidence, 2),
                'prediction_days': self.prediction_days,
                'adaptive_thresholds': volatility_info,
                'feature_importance': feature_importance_summary
            }

            print(
                f"[MLPredictor] {self.stock_symbol}: pred_return={predicted_return:.4f}, confidence={confidence:.2f}, recommendation={recommendation}")

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
        self.ta = None  # Wird später initialisiert
        self.df = None  # DataFrame für Live-Daten

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

            # Datum aus dem DataFrame oder aktuelles Datum
            from datetime import datetime

            # Wenn wir einen TechnicalAnalyzer und ein DataFrame haben, verwenden wir das letzte Datum daraus
            if self.ta is not None and hasattr(self.ta, 'df') and self.ta.df is not None and not self.ta.df.empty and 'date' in self.ta.df.columns:
                # Sicherstellen, dass die Daten nach Datum sortiert sind
                self.ta.df = self.ta.df.sort_values(by='date')
                latest_date = self.ta.df['date'].iloc[-1]
                date = latest_date.date() if hasattr(latest_date, 'date') else latest_date
                print(f"[DEBUG] AdaptiveAnalyzer verwendet Datum aus DataFrame: {date}")

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
                # Fallback auf aktuelles Datum und Details aus dem Ergebnis
                date = datetime.now().date()
                print(f"[DEBUG] AdaptiveAnalyzer verwendet aktuelles Datum: {date}")

                # Get details from the result
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
