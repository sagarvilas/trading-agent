# tests/test_utilities.py
import unittest
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from src.utilities import TradingSystemUtils


class TestUtilities(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create test data directories
        if not os.path.exists('test_reports'):
            os.makedirs('test_reports')

        # Create a sample summary file
        self.summary_data = {
            'date': '20230101',
            'portfolio': {
                'cash': 80000,
                'positions_value': 20000,
                'total_value': 100000,
                'positions': [
                    {'symbol': 'AAPL', 'shares': 10, 'value': 15000},
                    {'symbol': 'MSFT', 'shares': 5, 'value': 5000}
                ]
            },
            'trade_history': {
                'total_trades': 5,
                'profitable_trades': 3,
                'win_rate': 0.6,
                'avg_profit': 500,
                'avg_loss': -200
            }
        }

        with open('test_reports/summary_20230101.json', 'w') as f:
            json.dump(self.summary_data, f)

        # Create a second summary file
        self.summary_data2 = {
            'date': '20230102',
            'portfolio': {
                'cash': 75000,
                'positions_value': 28000,
                'total_value': 103000,
                'positions': [
                    {'symbol': 'AAPL', 'shares': 10, 'value': 16000},
                    {'symbol': 'MSFT', 'shares': 5, 'value': 6000},
                    {'symbol': 'GOOGL', 'shares': 2, 'value': 6000}
                ]
            },
            'trade_history': {
                'total_trades': 7,
                'profitable_trades': 4,
                'win_rate': 0.57,
                'avg_profit': 550,
                'avg_loss': -220
            }
        }

        with open('test_reports/summary_20230102.json', 'w') as f:
            json.dump(self.summary_data2, f)

    def tearDown(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists('test_reports'):
            shutil.rmtree('test_reports')

    def test_generate_performance_report(self):
        """Test generating performance reports"""
        # Generate report
        report = TradingSystemUtils.generate_performance_report(
            report_dir='test_reports'
        )

        # Assertions
        self.assertIsNotNone(report)
        self.assertIn("Trading System Performance Report", report)
        self.assertIn("Overall Performance", report)
        self.assertIn("3.00%", report)  # Total return (103000/100000 - 1) * 100

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_plot_performance(self, mock_figure, mock_savefig):
        """Test performance plotting"""
        # Create sample performance data
        performance_data = [
            {
                'timestamp': '2023-01-01T00:00:00',
                'total_value': 100000,
                'daily_return': 0
            },
            {
                'timestamp': '2023-01-02T00:00:00',
                'total_value': 102000,
                'daily_return': 0.02
            },
            {
                'timestamp': '2023-01-03T00:00:00',
                'total_value': 103000,
                'daily_return': 0.01
            }
        ]

        # Write to file
        performance_file = 'test_performance.json'
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f)

        try:
            # Plot performance
            fig, axes = TradingSystemUtils.plot_performance(
                performance_file, output_file='test_plot.png'
            )

            # Assertions
            mock_savefig.assert_called_once_with('test_plot.png')
        finally:
            if os.path.exists(performance_file):
                os.remove(performance_file)


if __name__ == '__main__':
    unittest.main()