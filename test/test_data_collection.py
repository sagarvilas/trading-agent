# tests/test_data_collection.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_collection import DataCollector


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        self.data_collector = DataCollector()

        # Create sample test data
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range(start='2023-01-01', periods=5))

    @patch('src.data_collection.yf.Ticker')
    def test_fetch_historical_data(self, mock_ticker):
        """Test fetching historical data"""
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_instance

        # Call the method
        result = self.data_collector.fetch_historical_data(['AAPL'], period='5d')

        # Assertions
        self.assertIn('AAPL', result)
        self.assertEqual(len(result['AAPL']), 5)
        mock_ticker.assert_called_once_with('AAPL')
        mock_instance.history.assert_called_once_with(period='5d', interval='1d')

    def test_preprocess_data(self):
        """Test data preprocessing functionality"""
        # Call the method
        processed_data = self.data_collector.preprocess_data(self.sample_data)

        # Assertions
        self.assertIn('Returns', processed_data.columns)
        self.assertIn('SMA_20', processed_data.columns)
        self.assertIn('RSI', processed_data.columns)
        self.assertIn('MACD', processed_data.columns)
        self.assertIn('BB_Upper', processed_data.columns)

        # Check calculations
        pd.testing.assert_series_equal(
            processed_data['Returns'],
            self.sample_data['Close'].pct_change(),
            check_names=False
        )

    @patch('src.data_collection.yf.Ticker')
    def test_fetch_real_time_data(self, mock_ticker):
        """Test fetching real-time data"""
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.history.return_value = self.sample_data.iloc[-1:].reset_index()
        mock_ticker.return_value = mock_instance

        # Call the method
        result = self.data_collector.fetch_real_time_data(['AAPL'])

        # Assertions
        self.assertIn('AAPL', result)
        mock_ticker.assert_called_once_with('AAPL')


if __name__ == '__main__':
    unittest.main()