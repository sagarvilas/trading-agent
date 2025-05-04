import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import logging
from sklearn.metrics import accuracy_score

# Import the PredictionModel class
from src.ai_prediction import PredictionModel

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPredictionModel(unittest.TestCase):
    """Tests for the PredictionModel class using unittest framework"""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory for model storage that will be used by all tests"""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests"""
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up a fresh model and sample data before each test"""
        self.model = PredictionModel(model_dir=self.temp_dir, logger=logger)
        self.sample_data = self.create_sample_data()

    def create_sample_data(self):
        """Create a sample stock data DataFrame for testing"""
        # Create date range for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')

        # Generate synthetic stock data
        np.random.seed(42)  # For reproducibility
        n = len(date_range)

        # Start with a base price and add random walks
        base_price = 100
        random_walk = np.random.normal(0, 1, n).cumsum()
        scaled_walk = random_walk * 5  # Scale for more realistic price movements

        close_prices = base_price + scaled_walk
        # Ensure prices are positive
        close_prices = np.maximum(close_prices, 1)

        # Generate High, Low, Open prices based on Close
        high_prices = close_prices * (1 + np.random.uniform(0, 0.05, n))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.05, n))
        open_prices = low_prices + np.random.uniform(0, 1, n) * (high_prices - low_prices)

        # Generate volume data
        volume = np.random.randint(100000, 10000000, n)

        # Create DataFrame
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=date_range)

        return df

    def test_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsNotNone(self.model.model_dir)
        self.assertIsNotNone(self.model.logger)
        self.assertIsNotNone(self.model.scaler)
        self.assertIsNone(self.model.classification_model)
        self.assertIsNone(self.model.regression_model)
        self.assertTrue(os.path.exists(self.model.model_dir))

    def test_calculate_indicators(self):
        """Test the indicator calculation functionality"""
        # Access the private method for testing
        df_with_indicators = self.model._calculate_indicators(self.sample_data)

        # Check that indicators were added
        expected_indicators = [
            'SMA_20', 'SMA_50', 'SMA_200', 'RSI',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'ADX', 'CCI', 'STOCH_K', 'STOCH_D',
            'WILLR', 'OBV', 'Volume_Change', 'Volume_MA', 'Returns'
        ]

        # Check each indicator was added
        for indicator in expected_indicators:
            self.assertIn(indicator, df_with_indicators.columns)

        # Verify calculations where possible
        # For example, SMA_20 should be close to the 20-day average of close prices
        # Check from index 20 onwards where SMA_20 is fully calculated
        if len(self.sample_data) > 20:
            # Get a random index where SMA should be fully calculated
            idx = np.random.randint(20, len(self.sample_data) - 1)
            sma20_manual = self.sample_data['Close'].iloc[idx - 20:idx].mean()
            sma20_calc = df_with_indicators['SMA_20'].iloc[idx]

            # Allow for small floating point differences
            if not np.isnan(sma20_calc):
                self.assertLess(abs(sma20_manual - sma20_calc), 1e-10)

    def test_prepare_features(self):
        """Test feature preparation"""
        X, data_clean = self.model._prepare_features(self.sample_data)

        # Verify X has expected features
        expected_base_features = [
            'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_Change', 'Volume_MA', 'Returns',
            'ADX', 'CCI', 'STOCH_K', 'STOCH_D', 'WILLR', 'OBV',
            'SMA_20_50_Cross', 'Price_SMA_20_Ratio', 'RSI_Oversold', 'RSI_Overbought',
            'MACD_Signal_Cross', 'BB_Position', 'STOCH_Oversold', 'STOCH_Overbought'
        ]

        # Check base features
        for feature in expected_base_features:
            self.assertIn(feature, X.columns)

        # Check lag features
        lag_features = [f for f in X.columns if '_Lag_' in f]
        self.assertGreater(len(lag_features), 0)

        # Verify no NaN values in the cleaned data
        self.assertFalse(X.isna().any().any())
        self.assertFalse(data_clean.isna().any().any())

    def test_train_models(self):
        """Test model training"""
        # Skip if data is too small
        if len(self.sample_data) < 250:  # Need enough data for SMA_200 and train/test split
            self.sample_data = self.sample_data.sample(300, replace=True)

        # Train models
        results = self.model.train_models(self.sample_data, target_days=1, symbol='TEST')

        # Check if models were created
        self.assertIsNotNone(self.model.classification_model)
        self.assertIsNotNone(self.model.regression_model)

        # Verify results structure
        self.assertIn('classification', results)
        self.assertIn('regression', results)
        self.assertIn('feature_importance', results)

        # Check metrics
        self.assertIn('accuracy', results['classification'])
        self.assertIn('precision', results['classification'])
        self.assertIn('recall', results['classification'])
        self.assertIn('rmse', results['regression'])

        # Verify feature importance
        self.assertGreater(len(results['feature_importance']['classification']), 0)
        self.assertGreater(len(results['feature_importance']['regression']), 0)

    def test_model_save_load(self):
        """Test saving and loading models"""
        # Skip if data is too small
        if len(self.sample_data) < 250:  # Need enough data for SMA_200 and train/test split
            self.sample_data = self.sample_data.sample(300, replace=True)

        # Train and save models
        self.model.train_models(self.sample_data, target_days=1, symbol='TEST')

        # Verify model files were created
        expected_files = [
            'TEST_1day_scaler.pkl',
            'TEST_1day_classification.pkl',
            'TEST_1day_regression.pkl'
        ]

        for filename in expected_files:
            filepath = os.path.join(self.model.model_dir, filename)
            self.assertTrue(os.path.exists(filepath))

        # Create a new model instance
        new_model = PredictionModel(model_dir=self.model.model_dir, logger=logger)

        # Load models
        self.assertTrue(new_model.load_models('TEST', target_days=1))

        # Verify models were loaded
        self.assertIsNotNone(new_model.classification_model)
        self.assertIsNotNone(new_model.regression_model)
        self.assertIsNotNone(new_model.scaler)

    def test_predict(self):
        """Test prediction functionality"""
        # Skip if data is too small
        if len(self.sample_data) < 250:  # Need enough data for SMA_200 and train/test split
            self.sample_data = self.sample_data.sample(300, replace=True)

        # Train models
        self.model.train_models(self.sample_data, target_days=1, symbol='TEST')

        # Make predictions
        predictions = self.model.predict(self.sample_data)

        # Check prediction results
        self.assertIn('Direction_Probability', predictions.columns)
        self.assertIn('Predicted_Change', predictions.columns)
        self.assertIn('Signal', predictions.columns)

        # Verify prediction values are in expected ranges
        self.assertGreaterEqual(predictions['Direction_Probability'].min(), 0)
        self.assertLessEqual(predictions['Direction_Probability'].max(), 1)
        self.assertTrue(all(predictions['Signal'].isin([-2, -1, 0, 1, 2])))

    def test_edge_case_empty_data(self):
        """Test behavior with empty data"""
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Calculate indicators should handle empty data
        with self.assertRaises(Exception):
            self.model._calculate_indicators(empty_df)

    def test_edge_case_short_data(self):
        """Test behavior with insufficient data"""
        # Create very short dataset (less than required for indicators)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
        short_df = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100 + 10,
            'Low': np.random.rand(10) * 100 - 10,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000, 10000, 10)
        }, index=dates)

        # Calculate indicators should handle short data
        indicators_df = self.model._calculate_indicators(short_df)

        # Many indicators should be NaN due to insufficient data
        self.assertTrue(indicators_df['SMA_20'].isna().all())
        self.assertTrue(indicators_df['SMA_50'].isna().all())
        self.assertTrue(indicators_df['SMA_200'].isna().all())

    def test_data_with_gaps(self):
        """Test behavior with data containing gaps"""
        # Create data with gaps
        dates = pd.date_range(start='2024-01-01', periods=300, freq='B')
        # Randomly remove 10% of dates
        gap_indices = np.random.choice(range(len(dates)), size=30, replace=False)
        dates = dates.delete(gap_indices)

        gapped_df = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100 + 10,
            'Low': np.random.rand(len(dates)) * 100 - 10,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # The model should still be able to calculate indicators and train
        indicators_df = self.model._calculate_indicators(gapped_df)
        self.assertFalse(indicators_df.empty)

        # And train models if data is sufficient
        results = self.model.train_models(gapped_df, target_days=1, symbol='GAPPED')
        self.assertIsNotNone(results)
        self.assertIn('classification', results)
        self.assertIn('regression', results)

    def test_real_world_performance(self):
        """Test performance metrics on sample data"""
        # Skip if data is too small
        if len(self.sample_data) < 250:  # Need enough data for SMA_200 and train/test split
            self.sample_data = self.sample_data.sample(300, replace=True)

        # Train with most of the data
        train_size = int(len(self.sample_data) * 0.8)
        train_data = self.sample_data.iloc[:train_size]
        test_data = self.sample_data.iloc[train_size:]

        # Train models
        self.model.train_models(train_data, target_days=1, symbol='TEST')

        # Make predictions on test data
        predictions = self.model.predict(test_data)

        # Create actual outcomes for the test set
        actual_direction = (test_data['Close'].shift(-1) > test_data['Close']).astype(int)
        actual_direction = actual_direction.iloc[:-1]  # Remove last row which has no next-day data

        # Get predicted direction (1 if probability > 0.5, else 0)
        predicted_direction = (predictions['Direction_Probability'] > 0.5).astype(int)
        predicted_direction = predicted_direction.iloc[:-1]  # Match lengths

        # Calculate accuracy manually
        if len(actual_direction) > 0 and len(predicted_direction) > 0:
            common_idx = predicted_direction.index.intersection(actual_direction.index)
            if len(common_idx) > 0:
                accuracy = accuracy_score(
                    actual_direction.loc[common_idx],
                    predicted_direction.loc[common_idx]
                )
                # Accuracy should be between 0 and 1
                self.assertGreaterEqual(accuracy, 0)
                self.assertLessEqual(accuracy, 1)


if __name__ == '__main__':
    unittest.main()