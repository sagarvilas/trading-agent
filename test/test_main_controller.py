# tests/test_main_controller.py
import unittest
import os
from unittest.mock import patch, MagicMock
from src.main_controller import TradingSystem


class TestMainController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create a test config file
        self.test_config = 'test_config.ini'
        with open(self.test_config, 'w') as f:
            f.write("[system]\n")
            f.write("trading_enabled = false\n")
            f.write("broker = alpaca\n")
            f.write("paper_trading = true\n")
            f.write("check_interval_min = 1\n")

            f.write("\n[risk_management]\n")
            f.write("max_position_pct = 0.1\n")
            f.write("stop_loss_pct = 0.05\n")
            f.write("take_profit_pct = 0.15\n")

        # Create test watchlist
        self.test_watchlist = 'test_watchlist.json'
        with open(self.test_watchlist, 'w') as f:
            f.write('{"symbols": ["AAPL", "MSFT"]}')

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_config):
            os.remove(self.test_config)
        if os.path.exists(self.test_watchlist):
            os.remove(self.test_watchlist)

    @patch('src.main_controller.DataCollector')
    @patch('src.main_controller.PredictionModel')
    @patch('src.main_controller.TradingStrategy')
    @patch('src.main_controller.TradingAPIFactory')
    def test_initialization(self, mock_api_factory, mock_strategy, mock_prediction, mock_data):
        """Test system initialization"""
        # Configure mocks
        mock_api_factory.create_client.return_value = MagicMock()

        # Create trading system
        system = TradingSystem(config_file=self.test_config)

        # Assertions
        self.assertIsNotNone(system.data_collector)
        self.assertIsNotNone(system.prediction_model)
        self.assertIsNotNone(system.trading_strategy)
        self.assertIsNotNone(system.api_client)
        self.assertFalse(system.is_trading_active)

    @patch('src.main_controller.schedule')
    @patch('src.main_controller.DataCollector')
    @patch('src.main_controller.PredictionModel')
    @patch('src.main_controller.TradingStrategy')
    @patch('src.main_controller.TradingAPIFactory')
    def test_start(self, mock_api_factory, mock_strategy, mock_prediction, mock_data, mock_schedule):
        """Test system start"""
        # Configure mocks
        mock_api_factory.create_client.return_value = MagicMock()

        # Create trading system
        system = TradingSystem(config_file=self.test_config)

        # Patch the main loop to avoid infinite loop
        with patch.object(system, 'pre_market_preparation') as mock_prep:
            with patch.object(system, 'is_trading_active', True, create=True):
                # Mock schedule.run_pending to raise an exception after one call to break the loop
                mock_schedule.run_pending.side_effect = [None, Exception("End test")]

                # Start system
                with self.assertRaises(Exception):
                    system.start()

                # Assertions
                mock_prep.assert_called_once()
                self.assertEqual(mock_schedule.every().call_count, 4)  # 4 scheduled tasks

    @patch('src.main_controller.DataCollector')
    @patch('src.main_controller.PredictionModel')
    @patch('src.main_controller.TradingStrategy')
    @patch('src.main_controller.TradingAPIFactory')
    def test_check_market(self, mock_api_factory, mock_strategy, mock_prediction, mock_data):
        """Test market check functionality"""
        # Configure mocks
        mock_api = MagicMock()
        mock_api_factory.create_client.return_value = mock_api

        strategy_instance = MagicMock()
        strategy_instance.positions = {}
        strategy_instance.get_portfolio_summary.return_value = {
            'positions': []
        }
        mock_strategy.return_value = strategy_instance

        # Create trading system
        system = TradingSystem(config_file=self.test_config)

        # Patch _is_trading_time to return True
        with patch.object(system, '_is_trading_time', return_value=True):
            # Patch _check_entry_signal
            with patch.object(system, '_check_entry_signal', return_value=False) as mock_entry:
                # Run market check
                system.check_market()

                # Assertions
                self.assertEqual(mock_entry.call_count, 2)  # Called for AAPL and MSFT


if __name__ == '__main__':
    unittest.main()