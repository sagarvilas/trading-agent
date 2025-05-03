import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import logging
import joblib
import os


class PredictionModel:
    """
    Responsible for training AI models to predict stock price movements
    and generate trading signals
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

    def _prepare_features(self, data):
        """
        Extract and prepare features from the raw data

        Args:
            data (DataFrame): Raw stock data with indicators

        Returns:
            DataFrame: Feature matrix ready for model training/prediction
        """
        # Define the features to use
        features = [
            'SMA_20', 'SMA_50', 'SMA_200',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Volume_Change', 'Volume_MA',
            'Returns'
        ]

        # Additional derived features
        data['SMA_20_50_Cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
        data['Price_SMA_20_Ratio'] = data['Close'] / data['SMA_20']
        data['RSI_Oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_Overbought'] = (data['RSI'] > 70).astype(int)
        data['MACD_Signal_Cross'] = ((data['MACD'] > data['MACD_Signal']) &
                                     (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))).astype(int)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

        # Add these derived features
        features.extend([
            'SMA_20_50_Cross', 'Price_SMA_20_Ratio',
            'RSI_Oversold', 'RSI_Overbought',
            'MACD_Signal_Cross', 'BB_Position'
        ])

        # Create lag features (prior days' data)
        for feature in ['Returns', 'RSI', 'MACD', 'Volume_Change']:
            for lag in [1, 2, 3, 5]:
                data[f'{feature}_Lag_{lag}'] = data[feature].shift(lag)
                features.append(f'{feature}_Lag_{lag}')

        # Drop rows with NaN values
        data_clean = data.dropna()

        # Extract features
        X = data_clean[features].copy()

        return X, data_clean

    def train_models(self, data, target_days=1, symbol=None, test_size=0.2):
        """
        Train classification and regression models for price prediction

        Args:
            data (DataFrame): Historical stock data with indicators
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
                                                              data_clean['Close'].shift(-target_days) / data_clean[
                                                          'Close'] - 1
                                                      ) * 100

        # Drop the last 'target_days' rows which don't have targets
        data_clean = data_clean.iloc[:-target_days]

        # Extract targets
        y_class = data_clean[f'Target_Direction_{target_days}d']
        y_reg = data_clean[f'Target_Change_{target_days}d']

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
            data (DataFrame): Stock data with technical indicators

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

    # Add some basic indicators for testing
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Initialize and train model
    model = PredictionModel()
    result = model.train_models(data, symbol='AAPL')

    # Make predictions
    predictions = model.predict(data)
    print("\nPrediction Sample:")
    print(predictions.tail())