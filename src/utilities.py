import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime, timedelta


class TradingSystemUtils:
    """
    Utility functions for the trading system
    """

    @staticmethod
    def generate_performance_report(report_dir='reports', output_file=None):
        """
        Generate a comprehensive performance report

        Args:
            report_dir (str): Directory containing report files
            output_file (str): Output file path

        Returns:
            str: Path to generated report
        """
        if not os.path.exists(report_dir):
            print(f"Report directory {report_dir} not found")
            return None

        # Find all summary files
        summary_files = [f for f in os.listdir(report_dir) if f.startswith('summary_') and f.endswith('.json')]

        if not summary_files:
            print("No summary files found")
            return None

        # Load all summaries
        summaries = []
        for file in sorted(summary_files):
            try:
                with open(os.path.join(report_dir, file), 'r') as f:
                    summary = json.load(f)
                    summaries.append(summary)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        if not summaries:
            print("No valid summary data found")
            return None

        # Generate report
        report = []
        report.append("# Trading System Performance Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overall performance
        first_date = summaries[0]['date']
        last_date = summaries[-1]['date']
        first_value = summaries[0]['portfolio']['total_value']
        last_value = summaries[-1]['portfolio']['total_value']

        total_return = (last_value / first_value - 1) * 100

        report.append("## Overall Performance")
        report.append(f"* Period: {first_date} to {last_date}")
        report.append(f"* Starting Value: ${first_value:.2f}")
        report.append(f"* Ending Value: ${last_value:.2f}")
        report.append(f"* Total Return: {total_return:.2f}%")

        # Combine trade history
        all_trades = []
        for summary in summaries:
            # Extract completed trades
            if 'trade_history' in summary:
                trade_history = summary['trade_history']
                if 'total_trades' in trade_history and trade_history['total_trades'] > 0:
                    trades = trade_history.get('trades', [])
                    all_trades.extend(trades)

        # Trade statistics
        if all_trades:
            # Count total trades
            total_trades = len([t for t in all_trades if t['action'] == 'SELL'])

            # Calculate win rate
            profitable_trades = len([t for t in all_trades if t.get('action') == 'SELL' and t.get('profit', 0) > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            report.append("\n## Trading Statistics")
            report.append(f"* Total Trades: {total_trades}")
            report.append(f"* Profitable Trades: {profitable_trades}")
            report.append(f"* Win Rate: {win_rate * 100:.2f}%")

            # Top performers
            if profitable_trades > 0:
                # Sort by profit
                profitable = [t for t in all_trades if t.get('action') == 'SELL' and t.get('profit', 0) > 0]
                profitable.sort(key=lambda x: x.get('profit_pct', 0), reverse=True)

                report.append("\n### Top Performers")
                for i, trade in enumerate(profitable[:5]):
                    report.append(
                        f"{i + 1}. {trade.get('symbol')}: {trade.get('profit_pct', 0):.2f}% (${trade.get('profit', 0):.2f})")

            # Worst performers
            if total_trades - profitable_trades > 0:
                # Sort by loss
                losing = [t for t in all_trades if t.get('action') == 'SELL' and t.get('profit', 0) <= 0]
                losing.sort(key=lambda x: x.get('profit_pct', 0))

                report.append("\n### Worst Performers")
                for i, trade in enumerate(losing[:5]):
                    report.append(
                        f"{i + 1}. {trade.get('symbol')}: {trade.get('profit_pct', 0):.2f}% (${trade.get('profit', 0):.2f})")

        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report))

            print(f"Report saved to {output_file}")
            return output_file
        else:
            return '\n'.join(report)

    @staticmethod
    def plot_performance(performance_file, output_file=None):
        """
        Generate performance visualization

        Args:
            performance_file (str): Path to performance data file
            output_file (str): Output file path for plot

        Returns:
            tuple: Figure and axes objects
        """
        if not os.path.exists(performance_file):
            print(f"Performance file {performance_file} not found")
            return None

        try:
            with open(performance_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading performance data: {str(e)}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set up the plotting environment
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot portfolio value
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['total_value'], label='Total Value', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('USD')
        ax1.legend()

        # Plot daily returns
        ax2 = axes[1]
        ax2.bar(df['timestamp'], df['daily_return'] * 100, label='Daily Return (%)')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('Daily Returns')
        ax2.set_ylabel('Percent (%)')
        ax2.set_xlabel('Date')
        ax2.legend()

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Performance plot saved to {output_file}")

        return fig, axes

    @staticmethod
    def analyze_trades(trade_history_file):
        """
        Analyze trade history to identify patterns and insights

        Args:
            trade_history_file (str): Path to trade history file

        Returns:
            dict: Analysis results
        """
        if not os.path.exists(trade_history_file):
            print(f"Trade history file {trade_history_file} not found")
            return None

        try:
            with open(trade_history_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading trade history: {str(e)}")
            return None

        trades = data.get('trades', [])
        if not trades:
            print("No trades found in history")
            return None

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)

        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()

        # Separate buy and sell trades
        buys = df[df['action'] == 'BUY']
        sells = df[df['action'] == 'SELL']

        # Calculate holding periods
        if not buys.empty and not sells.empty:
            # This is a simplified approach - in a real system you would match buys and sells
            symbol_hold_times = {}

            for symbol in df['symbol'].unique():
                symbol_buys = buys[buys['symbol'] == symbol]
                symbol_sells = sells[sells['symbol'] == symbol]

                if len(symbol_buys) > 0 and len(symbol_sells) > 0:
                    avg_hold_days = (symbol_sells['timestamp'].mean() - symbol_buys['timestamp'].mean()).days
                    symbol_hold_times[symbol] = avg_hold_days

        # Analyze profitability by symbol
        symbol_performance = {}

        if not sells.empty and 'symbol' in sells.columns and 'profit' in sells.columns:
            for symbol in sells['symbol'].unique():
                symbol_sells = sells[sells['symbol'] == symbol]
                total_profit = symbol_sells['profit'].sum()
                avg_profit = symbol_sells['profit'].mean()
                win_rate = (symbol_sells['profit'] > 0).mean()

                symbol_performance[symbol] = {
                    'trades': len(symbol_sells),
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'win_rate': win_rate
                }

        # Analyze trading by time of day
        hourly_performance = {}

        if not sells.empty and 'hour' in sells.columns and 'profit' in sells.columns:
            for hour in sells['hour'].unique():
                hour_sells = sells[sells['hour'] == hour]
                win_rate = (hour_sells['profit'] > 0).mean()
                avg_profit = hour_sells['profit'].mean()

                hourly_performance[int(hour)] = {
                    'trades': len(hour_sells),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit
                }

        # Analyze trading by day of week
        dow_performance = {}

        if not sells.empty and 'day_of_week' in sells.columns and 'profit' in sells.columns:
            for day in sells['day_of_week'].unique():
                day_sells = sells[sells['day_of_week'] == day]
                win_rate = (day_sells['profit'] > 0).mean()
                avg_profit = day_sells['profit'].mean()

                dow_performance[day] = {
                    'trades': len(day_sells),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit
                }

        # Compile results
        results = {
            'total_trades': len(trades),
            'symbols_traded': df['symbol'].nunique(),
            'avg_hold_time': np.mean(list(symbol_hold_times.values())) if symbol_hold_times else None,
            'symbol_performance': symbol_performance,
            'hourly_performance': hourly_performance,
            'day_of_week_performance': dow_performance
        }

        return results

    @staticmethod
    def backtest_strategy(symbols, strategy_params, start_date, end_date, initial_capital=100000):
        """
        Backtest a trading strategy on historical data

        Args:
            symbols (list): List of stock symbols to trade
            strategy_params (dict): Strategy parameters
            start_date (str): Start date for backtest (YYYY-MM-DD)
            end_date (str): End date for backtest (YYYY-MM-DD)
            initial_capital (float): Initial capital for backtest

        Returns:
            dict: Backtest results
        """
        try:
            # Import necessary classes
            from data_collection import DataCollector
            from ai_prediction import PredictionModel
            from trading_strategy import TradingStrategy, TradeAction

            # Initialize components
            data_collector = DataCollector()
            prediction_model = PredictionModel()
            trading_strategy = TradingStrategy(initial_capital=initial_capital)

            # Apply strategy parameters
            if 'max_position_pct' in strategy_params:
                trading_strategy.max_position_pct = strategy_params['max_position_pct']
            if 'stop_loss_pct' in strategy_params:
                trading_strategy.stop_loss_pct = strategy_params['stop_loss_pct']
            if 'take_profit_pct' in strategy_params:
                trading_strategy.take_profit_pct = strategy_params['take_profit_pct']

            # Fetch historical data
            historical_data = {}
            for symbol in symbols:
                print(f"Fetching data for {symbol}...")
                data = data_collector.fetch_historical_data([symbol], period="2y")
                if symbol in data and not data[symbol].empty:
                    # Filter by date range
                    mask = (data[symbol].index >= start_date) & (data[symbol].index <= end_date)
                    historical_data[symbol] = data[symbol].loc[mask]

            # Train prediction models
            trained_symbols = []
            for symbol, data in historical_data.items():
                if len(data) < 100:  # Need enough data to train
                    print(f"Not enough data for {symbol}, skipping")
                    continue

                # Use 70% of data for training
                train_size = int(len(data) * 0.7)
                train_data = data.iloc[:train_size]

                print(f"Training model for {symbol}...")
                prediction_model.train_models(train_data, symbol=symbol, target_days=1)
                trained_symbols.append(symbol)

            # Run backtest
            backtest_results = {}

            for symbol in trained_symbols:
                print(f"Running backtest for {symbol}...")
                symbol_data = historical_data[symbol]

                # Use remaining 30% of data for testing
                train_size = int(len(symbol_data) * 0.7)
                test_data = symbol_data.iloc[train_size:]

                # Load model
                if not prediction_model.load_models(symbol):
                    print(f"No model available for {symbol}, skipping")
                    continue

                # Generate predictions
                predictions = prediction_model.predict(test_data)

                # Initialize metrics
                trades = []
                portfolio_values = []
                current_position = None

                # Simulate trading
                for i in range(len(test_data) - 1):  # -1 to avoid using the last day (no next day price)
                    date = test_data.index[i]
                    current_row = test_data.iloc[i]
                    current_price = current_row['Close']

                    # Get prediction for this row
                    if i < len(predictions):
                        prediction = {
                            'Direction_Probability': predictions.iloc[i]['Direction_Probability'],
                            'Predicted_Change': predictions.iloc[i]['Predicted_Change'],
                            'Signal': predictions.iloc[i]['Signal']
                        }
                    else:
                        # Default neutral prediction if we don't have one
                        prediction = {
                            'Direction_Probability': 0.5,
                            'Predicted_Change': 0,
                            'Signal': 0
                        }

                    # Check if we have a position
                    if current_position is None:
                        # No position - check for entry
                        # Generate trade signal
                        trade_action, details = trading_strategy.generate_trade_signal(
                            symbol, test_data.iloc[:i + 1], prediction, current_price
                        )

                        if trade_action == TradeAction.BUY:
                            # Execute trade
                            result = trading_strategy.execute_trade(
                                symbol, trade_action, details, current_price, date
                            )

                            if result['success']:
                                trades.append(result['trade'])
                                current_position = symbol
                    else:
                        # Have position - check for exit
                        trade_action, details = trading_strategy.generate_trade_signal(
                            symbol, test_data.iloc[:i + 1], prediction, current_price
                        )

                        if trade_action == TradeAction.SELL:
                            # Execute trade
                            result = trading_strategy.execute_trade(
                                symbol, trade_action, details, current_price, date
                            )

                            if result['success']:
                                trades.append(result['trade'])
                                current_position = None

                    # Record portfolio value
                    portfolio = trading_strategy.get_portfolio_summary()
                    portfolio_values.append({
                        'date': date,
                        'total_value': portfolio['total_value']
                    })

                # Calculate final metrics
                trade_history = trading_strategy.get_trade_history_summary()
                final_portfolio = trading_strategy.get_portfolio_summary()

                # Calculate returns
                starting_value = initial_capital
                ending_value = final_portfolio['total_value']
                total_return = (ending_value / starting_value - 1) * 100

                # Calculate drawdown
                portfolio_df = pd.DataFrame(portfolio_values)
                if not portfolio_df.empty:
                    portfolio_df['drawdown'] = portfolio_df['total_value'].cummax() - portfolio_df['total_value']
                    portfolio_df['drawdown_pct'] = portfolio_df['drawdown'] / portfolio_df['total_value'].cummax() * 100
                    max_drawdown = portfolio_df['drawdown_pct'].max()
                else:
                    max_drawdown = 0

                # Store results
                backtest_results[symbol] = {
                    'trades': len(trades),
                    'win_rate': trade_history['win_rate'] if 'win_rate' in trade_history else 0,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'avg_profit_pct': trade_history['avg_profit_pct'] if 'avg_profit_pct' in trade_history else 0,
                    'avg_loss_pct': trade_history['avg_loss_pct'] if 'avg_loss_pct' in trade_history else 0
                }

            # Calculate aggregate results
            if backtest_results:
                total_trades = sum(result['trades'] for result in backtest_results.values())
                avg_win_rate = np.mean([result['win_rate'] for result in backtest_results.values()])
                avg_return = np.mean([result['total_return'] for result in backtest_results.values()])
                avg_drawdown = np.mean([result['max_drawdown'] for result in backtest_results.values()])

                aggregate = {
                    'symbols_tested': len(backtest_results),
                    'total_trades': total_trades,
                    'avg_win_rate': avg_win_rate,
                    'avg_return': avg_return,
                    'avg_drawdown': avg_drawdown
                }
            else:
                aggregate = {
                    'symbols_tested': 0,
                    'total_trades': 0,
                    'avg_win_rate': 0,
                    'avg_return': 0,
                    'avg_drawdown': 0
                }

            return {
                'symbol_results': backtest_results,
                'aggregate': aggregate,
                'params': strategy_params
            }

        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Example usage of the trading system
if __name__ == "__main__":
    print("Trading System Example Usage\n")

    # Example 1: Running the trading system
    print("Example 1: Running the trading system")
    print("from main_controller import TradingSystem")
    print("system = TradingSystem('config.ini')")
    print("system.start()")
    print("\n")

    # Example 2: Backtesting a strategy
    print("Example 2: Backtesting a strategy")
    print("from utilities import TradingSystemUtils")
    print("strategy_params = {")
    print("    'max_position_pct': 0.05,")
    print("    'stop_loss_pct': 0.03,")
    print("    'take_profit_pct': 0.10")
    print("}")
    print("symbols = ['AAPL', 'MSFT', 'GOOGL']")
    print("results = TradingSystemUtils.backtest_strategy(")
    print("    symbols,")
    print("    strategy_params,")
    print("    start_date='2022-01-01',")
    print("    end_date='2022-12-31',")
    print("    initial_capital=100000")
    print(")")
    print("print(results['aggregate'])")
    print("\n")

    # Example 3: Generating a performance report
    print("Example 3: Generating a performance report")
    print("from utilities import TradingSystemUtils")
    print("report = TradingSystemUtils.generate_performance_report(")
    print("    report_dir='reports',")
    print("    output_file='performance_report.md'")
    print(")")
    print("\n")

    # Example 4: Analyzing trades
    print("Example 4: Analyzing trades")
    print("from utilities import TradingSystemUtils")
    print("analysis = TradingSystemUtils.analyze_trades('state/trade_history.json')")
    print("print(analysis['symbol_performance'])")