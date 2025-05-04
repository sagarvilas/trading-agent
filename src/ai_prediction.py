import numpy as np
import tulipy as ti
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import logging
import joblib
import os


class PredictionModel:
    """
    Responsible for training AI models to predict stock price movements
    and generate trading signals using tulipy for technical indicators
    """

    def __init__(self, model_dir="models", logger=None):
        """
        Initialize the prediction model

        Args:
            model_dir (str): Directory to save/load models
            logger: Optional logger instance
        """
        self.model_dir = model_dir
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.classification_model = None  # For predicting direction (up/down)
        self.regression_model = None  # For predicting actual price changes

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _calculate_indicators(self, data):
        """
        Calculate technical indicators using tulipy

        Args:
            data (DataFrame): Raw stock data with OHLCV columns

        Returns:
            DataFrame: Data with added technical indicators
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()

        # Convert to numpy arrays for tulipy (which requires float64 numpy arrays)
        close = df['Close'].to_numpy().astype(np.float64)
        high = df['High'].to_numpy().astype(np.float64)
        low = df['Low'].to_numpy().astype(np.float64)
        volume = df['Volume'].to_numpy().astype(np.float64)

        # Initialize indicator columns with NaN
        indicator_columns = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal',
                             'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ADX',
                             'CCI', 'STOCH_K', 'STOCH_D', 'WILLR', 'OBV']

        for col in indicator_columns:
            df[col] = np.nan

        try:
            # Calculate indicators
            sma20 = ti.sma(close, 20)
            sma50 = ti.sma(close, 50)
            sma200 = ti.sma(close, 200)
            rsi = ti.rsi(close, 14)
            macd, macd_signal, macd_hist = ti.macd(close, 12, 26, 9)
            bb_upper, bb_middle, bb_lower = ti.bbands(close, 20, 2)
            adx = ti.adx(high, low, close, 14)
            cci = ti.cci(high, low, close, 14)
            stoch_k, stoch_d = ti.stoch(high, low, close, 14, 3, 3)
            willr = ti.willr(high, low, close, 14)
            obv = ti.obv(close, volume)

            # Use integer position indexing with iloc instead of loc
            # SMA 20
            if len(sma20) > 0:
                start_idx = 19
                end_idx = min(start_idx + len(sma20), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('SMA_20')] = sma20[:end_idx - start_idx]

            # SMA 50
            if len(sma50) > 0:
                start_idx = 49
                end_idx = min(start_idx + len(sma50), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('SMA_50')] = sma50[:end_idx - start_idx]

            # SMA 200
            if len(sma200) > 0:
                start_idx = 199
                end_idx = min(start_idx + len(sma200), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('SMA_200')] = sma200[:end_idx - start_idx]

            # RSI
            if len(rsi) > 0:
                start_idx = 13
                end_idx = min(start_idx + len(rsi), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('RSI')] = rsi[:end_idx - start_idx]

            # MACD
            if len(macd) > 0:
                start_idx = 26 + 9 - 2  # MACD start index
                end_idx = min(start_idx + len(macd), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('MACD')] = macd[:end_idx - start_idx]
                    df.iloc[start_idx:end_idx, df.columns.get_loc('MACD_Signal')] = macd_signal[:end_idx - start_idx]
                    df.iloc[start_idx:end_idx, df.columns.get_loc('MACD_Hist')] = macd_hist[:end_idx - start_idx]

            # Bollinger Bands
            if len(bb_upper) > 0:
                start_idx = 19
                end_idx = min(start_idx + len(bb_upper), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('BB_Upper')] = bb_upper[:end_idx - start_idx]
                    df.iloc[start_idx:end_idx, df.columns.get_loc('BB_Middle')] = bb_middle[:end_idx - start_idx]
                    df.iloc[start_idx:end_idx, df.columns.get_loc('BB_Lower')] = bb_lower[:end_idx - start_idx]

            # ADX
            if len(adx) > 0:
                start_idx = 27
                end_idx = min(start_idx + len(adx), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('ADX')] = adx[:end_idx - start_idx]

            # CCI
            if len(cci) > 0:
                start_idx = 13
                end_idx = min(start_idx + len(cci), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('CCI')] = cci[:end_idx - start_idx]

            # Stochastic
            if len(stoch_k) > 0:
                start_idx = 14 + 3 - 2
                end_idx = min(start_idx + len(stoch_k), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('STOCH_K')] = stoch_k[:end_idx - start_idx]
                    df.iloc[start_idx:end_idx, df.columns.get_loc('STOCH_D')] = stoch_d[:end_idx - start_idx]

            # Williams %R
            if len(willr) > 0:
                start_idx = 13
                end_idx = min(start_idx + len(willr), len(df))
                if end_idx > start_idx:
                    df.iloc[start_idx:end_idx, df.columns.get_loc('WILLR')] = willr[:end_idx - start_idx]

            # OBV
            if len(obv) > 0:
                end_idx = min(len(obv), len(df))
                df.iloc[:end_idx, df.columns.get_loc('OBV')] = obv[:end_idx]

            # Calculate volume indicators directly on the dataframe
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

            # Calculate Returns
            df['Returns'] = df['Close'].pct_change() * 100

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

        return df

    def _prepare_features(self, data):
        """
        Extract and prepare features from the raw data with indicators

        Args:
            data (DataFrame): Raw stock data with OHLCV columns

        Returns:
            DataFrame: Feature matrix ready for model training/prediction
        """
        # Calculate technical indicators
        df = self._calculate_indicators(data)

        # Define the features to use
        features = [
            'SMA_20', 'SMA_50', 'SMA_200',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_Change', 'Volume_MA',
            'Returns', 'ADX', 'CCI', 'STOCH_K', 'STOCH_D',
            'WILLR', 'OBV'
        ]

        # Additional derived features
        df['SMA_20_50_Cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        df['Price_SMA_20_Ratio'] = df['Close'] / df['SMA_20']
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['MACD_Signal_Cross'] = ((df['MACD'] > df['MACD_Signal']) &
                                   (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['STOCH_Oversold'] = (df['STOCH_K'] < 20).astype(int)
        df['STOCH_Overbought'] = (df['STOCH_K'] > 80).astype(int)

        # Add these derived features
        features.extend([
            'SMA_20_50_Cross', 'Price_SMA_20_Ratio',
            'RSI_Oversold', 'RSI_Overbought',
            'MACD_Signal_Cross', 'BB_Position',
            'STOCH_Oversold', 'STOCH_Overbought'
        ])

        # Create lag features (prior days' data)
        for feature in ['Returns', 'RSI', 'MACD', 'Volume_Change', 'CCI', 'WILLR']:
            for lag in [1, 2, 3, 5]:
                df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
                features.append(f'{feature}_Lag_{lag}')

        # Drop rows with NaN values
        df_clean = df.dropna()

        # Extract features
        X = df_clean[features].copy()

        return X, df_clean

    def train_models(self, data, target_days=1, symbol=None, test_size=0.2):
        """
        Train classification and regression models for price prediction

        Args:
            data (DataFrame): Historical stock data with OHLCV columns
            target_days (int): Number of days ahead to predict
            symbol (str): Stock symbol (used for model naming)
            test_size (float): Proportion of data to use for testing

        Returns:
            dict: Training results and metrics
        """
        self.logger.info(f"Training models for {symbol or 'stock'} with {target_days}-day target")

        # Prepare features
        X, data_clean = self._prepare_features(data)

        # Create target variables
        # Classification target: 1 if price goes up, 0 if down
        data_clean[f'Target_Direction_{target_days}d'] = (
                data_clean['Close'].shift(-target_days) > data_clean['Close']
        ).astype(int)

        # Regression target: Percentage price change
        data_clean[f'Target_Change_{target_days}d'] = (
                                                              data_clean['Close'].shift(-target_days) /
                                                              data_clean['Close'] - 1
                                                      ) * 100

        # Drop the last 'target_days' rows which don't have targets
        data_clean = data_clean.iloc[:-target_days]

        # Extract targets
        y_class = data_clean[f'Target_Direction_{target_days}d']
        y_reg = data_clean[f'Target_Change_{target_days}d']

        # IMPORTANT: Make sure X matches the cleaned dataframe rows
        # This is likely where your inconsistency error is happening
        X = X.loc[data_clean.index]

        # Check for NaN values and drop them to ensure consistent array lengths
        mask = ~(X.isna().any(axis=1) | y_class.isna() | y_reg.isna())
        X = X[mask]
        y_class = y_class[mask]
        y_reg = y_reg[mask]

        # Verify all arrays have the same length before splitting
        if not (len(X) == len(y_class) == len(y_reg)):
            self.logger.error(f"Inconsistent array lengths: X={len(X)}, y_class={len(y_class)}, y_reg={len(y_reg)}")
            raise ValueError("Input arrays have different lengths")

        # Split data into training and testing sets
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=test_size, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classification model (predict direction)
        self.logger.info("Training classification model...")
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classification_model.fit(X_train_scaled, y_class_train)

        # Train regression model (predict price change magnitude)
        self.logger.info("Training regression model...")
        self.regression_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.regression_model.fit(X_train_scaled, y_reg_train)

        # Evaluate models
        y_class_pred = self.classification_model.predict(X_test_scaled)
        y_reg_pred = self.regression_model.predict(X_test_scaled)

        # Calculate metrics
        class_accuracy = accuracy_score(y_class_test, y_class_pred)
        class_precision = precision_score(y_class_test, y_class_pred)
        class_recall = recall_score(y_class_test, y_class_pred)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        reg_rmse = np.sqrt(reg_mse)

        self.logger.info(f"Classification Accuracy: {class_accuracy:.4f}")
        self.logger.info(f"Classification Precision: {class_precision:.4f}")
        self.logger.info(f"Classification Recall: {class_recall:.4f}")
        self.logger.info(f"Regression RMSE: {reg_rmse:.4f}%")

        # Save models if a symbol is provided
        if symbol:
            model_prefix = f"{symbol}_{target_days}day"
            scaler_path = os.path.join(self.model_dir, f"{model_prefix}_scaler.pkl")
            class_model_path = os.path.join(self.model_dir, f"{model_prefix}_classification.pkl")
            reg_model_path = os.path.join(self.model_dir, f"{model_prefix}_regression.pkl")

            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.classification_model, class_model_path)
            joblib.dump(self.regression_model, reg_model_path)
            self.logger.info(f"Models saved to {self.model_dir}")

        # Return metrics
        return {
            "classification": {
                "accuracy": class_accuracy,
                "precision": class_precision,
                "recall": class_recall
            },
            "regression": {
                "mse": reg_mse,
                "rmse": reg_rmse
            },
            "feature_importance": {
                "classification": dict(zip(X.columns, self.classification_model.feature_importances_)),
                "regression": dict(zip(X.columns, self.regression_model.feature_importances_))
            }
        }

    def load_models(self, symbol, target_days=1):
        """
        Load pre-trained models for a specific stock

        Args:
            symbol (str): Stock symbol
            target_days (int): Prediction time horizon

        Returns:
            bool: True if models loaded successfully
        """
        model_prefix = f"{symbol}_{target_days}day"
        scaler_path = os.path.join(self.model_dir, f"{model_prefix}_scaler.pkl")
        class_model_path = os.path.join(self.model_dir, f"{model_prefix}_classification.pkl")
        reg_model_path = os.path.join(self.model_dir, f"{model_prefix}_regression.pkl")

        try:
            self.scaler = joblib.load(scaler_path)
            self.classification_model = joblib.load(class_model_path)
            self.regression_model = joblib.load(reg_model_path)
            self.logger.info(f"Models for {symbol} loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models for {symbol}: {str(e)}")
            return False

    def predict(self, data):
        """
        Generate predictions for the given data

        Args:
            data (DataFrame): Stock data with OHLCV columns

        Returns:
            dict: Prediction results including direction probability and price change
        """
        if self.classification_model is None or self.regression_model is None:
            self.logger.error("Models not trained or loaded. Call train_models() or load_models() first.")
            return None

        # Prepare features
        X, data_clean = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)

        # Make predictions
        direction_prob = self.classification_model.predict_proba(X_scaled)[:, 1]
        price_change = self.regression_model.predict(X_scaled)

        # Add predictions to the data
        data_clean['Direction_Probability'] = direction_prob
        data_clean['Predicted_Change'] = price_change

        # Generate trading signals
        # Bullish signal: high probability of upward movement and positive predicted change
        data_clean['Signal'] = 0  # Neutral
        # Strong buy signal
        data_clean.loc[(direction_prob > 0.65) & (price_change > 1.0), 'Signal'] = 2
        # Buy signal
        data_clean.loc[(direction_prob > 0.55) & (price_change > 0.5), 'Signal'] = 1
        # Strong sell signal
        data_clean.loc[(direction_prob < 0.35) & (price_change < -1.0), 'Signal'] = -2
        # Sell signal
        data_clean.loc[(direction_prob < 0.45) & (price_change < -0.5), 'Signal'] = -1

        return data_clean[['Direction_Probability', 'Predicted_Change', 'Signal']]


# Usage example:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create dummy data for testing
    import yfinance as yf

    ticker = yf.Ticker('AAPL')
    data = ticker.history(period="1y")

    # Initialize and train model
    model = PredictionModel()
    result = model.train_models(data, symbol='AAPL')

    # Make predictions
    predictions = model.predict(data)
    print("\nPrediction Sample:")
    print(predictions.tail())