import logging
import time
import schedule
import json
import os
from datetime import datetime, timedelta
import configparser
import traceback

# Import our modules
# In a real project, you would use proper imports based on your package structure
# For simplicity we're assuming all modules are in the same directory
from data_collection import DataCollector
from ai_prediction import PredictionModel
from trading_strategy import TradingStrategy, TradeAction, PositionSize
from api_integration import TradingAPIFactory


class TradingSystem:
    """
    Main controller class that orchestrates the entire trading system
    """

    def __init__(self, config_file='config.ini'):
        """
        Initialize the trading system

        Args:
            config_file (str): Path to configuration file
        """
        # Setup logging
        self._setup_logging()

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize components
        self._initialize_components()

        # Trading state
        self.is_trading_active = False
        self.watchlist = self._load_watchlist()
        self.trading_hours = {
            'start': datetime.strptime(self.config.get('trading_hours', {}).get('start', '09:30'), '%H:%M').time(),
            'end': datetime.strptime(self.config.get('trading_hours', {}).get('end', '16:00'), '%H:%M').time()
        }

        # Performance tracking
        self.performance_history = []
        self.last_performance_check = datetime.now()

    def _setup_logging(self):
        """Configure logging for the trading system"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing trading system")

    def _load_config(self, config_file):
        """
        Load configuration from file

        Args:
            config_file (str): Path to configuration file

        Returns:
            dict: Configuration parameters
        """
        if not os.path.exists(config_file):
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return self._get_default_config()

        try:
            config = configparser.ConfigParser()
            config.read(config_file)

            # Convert to dictionary
            config_dict = {section: dict(config[section]) for section in config.sections()}

            # Convert certain values from strings
            if 'system' in config_dict:
                if 'trading_enabled' in config_dict['system']:
                    config_dict['system']['trading_enabled'] = config.getboolean('system', 'trading_enabled')

                if 'max_positions' in config_dict['system']:
                    config_dict['system']['max_positions'] = config.getint('system', 'max_positions')

            if 'risk_management' in config_dict:
                for key in ['max_position_pct', 'stop_loss_pct', 'take_profit_pct']:
                    if key in config_dict['risk_management']:
                        config_dict['risk_management'][key] = config.getfloat('risk_management', key)

            self.logger.info("Configuration loaded successfully")
            return config_dict

        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self):
        """
        Get default configuration

        Returns:
            dict: Default configuration
        """
        return {
            'system': {
                'trading_enabled': False,
                'broker': 'alpaca',
                'paper_trading': True,
                'max_positions': 5,
                'check_interval_min': 5
            },
            'risk_management': {
                'max_position_pct': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.15
            },
            'trading_hours': {
                'start': '09:30',
                'end': '16:00'
            }
        }

    def _initialize_components(self):
        """Initialize all trading system components"""
        try:
            # Data collector
            self.logger.info("Initializing data collector")
            self.data_collector = DataCollector(logger=logging.getLogger('data_collector'))

            # AI prediction model
            self.logger.info("Initializing prediction model")
            model_dir = self.config.get('system', {}).get('model_dir', 'models')
            self.prediction_model = PredictionModel(model_dir=model_dir,
                                                    logger=logging.getLogger('prediction_model'))

            # Trading strategy
            self.logger.info("Initializing trading strategy")
            risk_config = self.config.get('risk_management', {})

            self.trading_strategy = TradingStrategy(
                initial_capital=float(risk_config.get('initial_capital', 100000)),
                max_position_pct=float(risk_config.get('max_position_pct', 0.1)),
                stop_loss_pct=float(risk_config.get('stop_loss_pct', 0.05)),
                take_profit_pct=float(risk_config.get('take_profit_pct', 0.15)),
                logger=logging.getLogger('trading_strategy')
            )

            # Trading API client
            self.logger.info("Initializing trading API client")
            system_config = self.config.get('system', {})
            broker = system_config.get('broker', 'alpaca')
            paper_trading = system_config.get('paper_trading', True)

            self.api_client = TradingAPIFactory.create_client(
                broker,
                config_file=self.config.get('api', {}).get('config_file', 'api_credentials.ini'),
                paper_trading=paper_trading,
                logger=logging.getLogger('api_client')
            )

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _load_watchlist(self):
        """
        Load the watchlist of symbols to monitor and trade

        Returns:
            list: List of stock symbols
        """
        watchlist_file = self.config.get('system', {}).get('watchlist_file', 'watchlist.json')

        if os.path.exists(watchlist_file):
            try:
                with open(watchlist_file, 'r') as f:
                    data = json.load(f)
                    symbols = data.get('symbols', [])
                    self.logger.info(f"Loaded {len(symbols)} symbols from watchlist")
                    return symbols
            except Exception as e:
                self.logger.error(f"Error loading watchlist: {str(e)}")
                return self._get_default_watchlist()
        else:
            self.logger.warning(f"Watchlist file {watchlist_file} not found, using defaults")
            return self._get_default_watchlist()

    def _get_default_watchlist(self):
        """
        Get default watchlist of popular stocks

        Returns:
            list: Default list of stock symbols
        """
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']

    def _is_trading_time(self):
        """
        Check if current time is within trading hours

        Returns:
            bool: True if within trading hours
        """
        now = datetime.now().time()
        return self.trading_hours['start'] <= now <= self.trading_hours['end']

    def start(self):
        """Start the trading system"""
        self.logger.info("Starting trading system")

        try:
            # Check if trading is enabled in config
            trading_enabled = self.config.get('system', {}).get('trading_enabled', False)
            if not trading_enabled:
                self.logger.warning("Trading is disabled in configuration")
                self.logger.info("Running in analysis-only mode")

            # Set up recurring tasks
            check_interval = int(self.config.get('system', {}).get('check_interval_min', 5))
            schedule.every(check_interval).minutes.do(self.check_market)
            schedule.every().day.at("16:05").do(self.end_of_day_summary)
            schedule.every().day.at("08:45").do(self.pre_market_preparation)
            schedule.every(1).hours.do(self.update_performance_metrics)

            # Initial market check and model training
            self.pre_market_preparation()

            # Main loop
            self.is_trading_active = True
            self.logger.info("Trading system active, entering main loop")

            while self.is_trading_active:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            self.logger.error(f"Error in trading system: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.stop()

    def stop(self):
        """Stop the trading system"""
        self.logger.info("Stopping trading system")
        self.is_trading_active = False

        # Save current state
        self._save_state()

        # Final performance report
        self.end_of_day_summary()

        self.logger.info("Trading system stopped")

    def _save_state(self):
        """Save the current state of the trading system"""
        try:
            state_dir = 'state'
            if not os.path.exists(state_dir):
                os.makedirs(state_dir)

            # Save portfolio state
            portfolio = self.trading_strategy.get_portfolio_summary()
            with open(os.path.join(state_dir, 'portfolio.json'), 'w') as f:
                json.dump(portfolio, f, indent=2)

            # Save trade history
            trade_history = {
                'trades': self.trading_strategy.trade_history,
                'summary': self.trading_strategy.get_trade_history_summary()
            }
            with open(os.path.join(state_dir, 'trade_history.json'), 'w') as f:
                json.dump(trade_history, f, indent=2, default=str)

            # Save performance metrics
            with open(os.path.join(state_dir, 'performance.json'), 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)

            self.logger.info("System state saved")

        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}")

    def pre_market_preparation(self):
        """Run pre-market preparation tasks"""
        self.logger.info("Running pre-market preparation")

        try:
            # Fetch latest historical data for all symbols
            historical_data = {}
            for symbol in self.watchlist:
                try:
                    data = self.data_collector.fetch_historical_data([symbol], period="6mo")
                    if symbol in data and not data[symbol].empty:
                        historical_data[symbol] = data[symbol]
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")

            # Train or update prediction models
            for symbol, data in historical_data.items():
                try:
                    self.logger.info(f"Training model for {symbol}")
                    self.prediction_model.train_models(data, symbol=symbol, target_days=1)
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {str(e)}")

            # Get account information
            account_info = self.api_client.get_account_info()
            self.logger.info(f"Account value: ${account_info.get('equity', 'N/A')}")

            # Get current positions from the broker
            positions = self.api_client.get_positions()
            self.logger.info(f"Current positions: {len(positions) if positions else 0}")

            self.logger.info("Pre-market preparation complete")

        except Exception as e:
            self.logger.error(f"Error in pre-market preparation: {str(e)}")
            self.logger.error(traceback.format_exc())

    def check_market(self):
        """
        Main market check function that runs on schedule
        Analyzes current market data and makes trading decisions
        """
        self.logger.info("Running market check")

        # Check if trading is enabled
        trading_enabled = self.config.get('system', {}).get('trading_enabled', False)
        if not trading_enabled:
            self.logger.info("Trading is disabled, running in analysis-only mode")

        # Check if within trading hours
        is_trading_time = self._is_trading_time()
        if not is_trading_time:
            self.logger.info("Outside of trading hours, skipping trade execution")

        try:
            # Get current portfolio state
            portfolio = self.trading_strategy.get_portfolio_summary()
            current_positions = list(self.trading_strategy.positions.keys())

            # Max positions check
            max_positions = int(self.config.get('system', {}).get('max_positions', 5))
            positions_available = max_positions - len(current_positions)

            # Process each symbol in the watchlist
            for symbol in self.watchlist:
                try:
                    # Skip if we already have a position and not checking for exit
                    if symbol in current_positions:
                        self._check_exit_signal(symbol)
                        continue

                    # Skip if we're at max positions
                    if positions_available <= 0:
                        self.logger.info(f"Maximum positions reached ({max_positions}), skipping {symbol}")
                        continue

                    # Check for entry signals
                    entry_signal = self._check_entry_signal(symbol)

                    # Execute trade if conditions are met
                    if entry_signal and trading_enabled and is_trading_time:
                        # Reduce available positions count
                        positions_available -= 1

                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")

            self.logger.info("Market check complete")

        except Exception as e:
            self.logger.error(f"Error in market check: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _check_entry_signal(self, symbol):
        """
        Check for entry signals for a symbol

        Args:
            symbol (str): Stock symbol

        Returns:
            bool: True if entry signal generated
        """
        try:
            # Fetch recent data
            data = self.data_collector.fetch_historical_data([symbol], period="1mo")
            if symbol not in data or data[symbol].empty:
                self.logger.warning(f"No data available for {symbol}")
                return False

            # Load prediction model
            if not self.prediction_model.load_models(symbol):
                self.logger.warning(f"No prediction model available for {symbol}, skipping")
                return False

            # Generate prediction
            prediction_df = self.prediction_model.predict(data[symbol])
            if prediction_df is None or prediction_df.empty:
                self.logger.warning(f"No prediction generated for {symbol}")
                return False

            # Get latest prediction
            latest_prediction = prediction_df.iloc[-1]

            # Get current market data
            market_data = self.api_client.get_market_data(symbol)
            current_price = float(market_data.get('price', 0))

            if current_price <= 0:
                self.logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return False

            # Generate trade signal
            trade_action, details = self.trading_strategy.generate_trade_signal(
                symbol, data[symbol], latest_prediction, current_price
            )

            # If buy signal
            if trade_action == TradeAction.BUY:
                self.logger.info(f"BUY signal for {symbol} at ${current_price:.2f}")
                self.logger.info(f"Signal details: {details}")

                # Execute trade
                trading_enabled = self.config.get('system', {}).get('trading_enabled', False)
                if trading_enabled and self._is_trading_time():
                    result = self.trading_strategy.execute_trade(
                        symbol, trade_action, details, current_price
                    )

                    if result['success']:
                        # Place actual order with broker
                        try:
                            order = self.api_client.place_market_order(
                                symbol, details['shares'], 'buy'
                            )
                            self.logger.info(f"Order placed: {order}")
                        except Exception as e:
                            self.logger.error(f"Error placing order: {str(e)}")

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking entry signal for {symbol}: {str(e)}")
            return False

    def _check_exit_signal(self, symbol):
        """
        Check for exit signals for a position

        Args:
            symbol (str): Stock symbol

        Returns:
            bool: True if exit signal generated
        """
        if symbol not in self.trading_strategy.positions:
            return False

        try:
            # Fetch recent data
            data = self.data_collector.fetch_historical_data([symbol], period="1mo")
            if symbol not in data or data[symbol].empty:
                self.logger.warning(f"No data available for {symbol}")
                return False

            # Load prediction model
            if not self.prediction_model.load_models(symbol):
                self.logger.warning(f"No prediction model available for {symbol}, using default exit logic")
                # Continue with basic exit logic
            else:
                # Generate prediction
                prediction_df = self.prediction_model.predict(data[symbol])
                if prediction_df is not None and not prediction_df.empty:
                    # Extract latest prediction
                    latest_prediction = prediction_df.iloc[-1]
                else:
                    latest_prediction = None

            # Get current market data
            market_data = self.api_client.get_market_data(symbol)
            current_price = float(market_data.get('price', 0))

            if current_price <= 0:
                self.logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return False

            # Generate trade signal
            trade_action, details = self.trading_strategy.generate_trade_signal(
                symbol, data[symbol],
                latest_prediction if 'latest_prediction' in locals() else None,
                current_price
            )

            # If sell signal
            if trade_action == TradeAction.SELL:
                self.logger.info(f"SELL signal for {symbol} at ${current_price:.2f}")
                self.logger.info(f"Signal details: {details}")

                # Execute trade
                trading_enabled = self.config.get('system', {}).get('trading_enabled', False)
                if trading_enabled and self._is_trading_time():
                    result = self.trading_strategy.execute_trade(
                        symbol, trade_action, details, current_price
                    )

                    if result['success']:
                        # Place actual order with broker
                        try:
                            position = self.trading_strategy.positions.get(symbol, {})
                            shares = position.get('qty', 0)

                            order = self.api_client.place_market_order(
                                symbol, shares, 'sell'
                            )
                            self.logger.info(f"Order placed: {order}")
                        except Exception as e:
                            self.logger.error(f"Error placing order: {str(e)}")

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking exit signal for {symbol}: {str(e)}")
            return False

    def end_of_day_summary(self):
        """Generate end of day summary and statistics"""
        self.logger.info("Generating end of day summary")

        try:
            # Get portfolio summary
            portfolio = self.trading_strategy.get_portfolio_summary()

            # Get trade history summary
            trade_history = self.trading_strategy.get_trade_history_summary()

            # Log summary
            self.logger.info(f"=== Portfolio Summary ===")
            self.logger.info(f"Cash: ${portfolio['cash']:.2f}")
            self.logger.info(f"Positions value: ${portfolio['positions_value']:.2f}")
            self.logger.info(f"Total value: ${portfolio['total_value']:.2f}")
            self.logger.info(f"Number of positions: {len(portfolio['positions'])}")

            self.logger.info(f"=== Trading Summary ===")
            self.logger.info(f"Total trades: {trade_history['total_trades']}")
            self.logger.info(f"Win rate: {trade_history['win_rate'] * 100:.2f}%")
            self.logger.info(
                f"Average profit: ${trade_history['avg_profit']:.2f} ({trade_history['avg_profit_pct']:.2f}%)")
            self.logger.info(f"Average loss: ${trade_history['avg_loss']:.2f} ({trade_history['avg_loss_pct']:.2f}%)")

            # Save summary to file
            summary_dir = 'reports'
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)

            date_str = datetime.now().strftime("%Y%m%d")
            with open(os.path.join(summary_dir, f'summary_{date_str}.json'), 'w') as f:
                summary = {
                    'date': date_str,
                    'portfolio': portfolio,
                    'trade_history': trade_history
                }
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(f"End of day summary saved to reports/summary_{date_str}.json")

        except Exception as e:
            self.logger.error(f"Error generating end of day summary: {str(e)}")

    def update_performance_metrics(self):
        """Update performance metrics"""
        self.logger.info("Updating performance metrics")

        try:
            # Get current portfolio value
            portfolio = self.trading_strategy.get_portfolio_summary()
            total_value = portfolio['total_value']

            # Calculate daily return if we have previous metrics
            daily_return = 0
            if self.performance_history:
                last_value = self.performance_history[-1]['total_value']
                daily_return = (total_value - last_value) / last_value

            # Record metrics
            metrics = {
                'timestamp': datetime.now(),
                'total_value': total_value,
                'cash': portfolio['cash'],
                'positions_value': portfolio['positions_value'],
                'num_positions': len(portfolio['positions']),
                'daily_return': daily_return
            }

            self.performance_history.append(metrics)
            self.last_performance_check = datetime.now()

            # Log metrics
            self.logger.info(
                f"Performance update: Total value: ${total_value:.2f}, Daily return: {daily_return * 100:.2f}%")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    def add_to_watchlist(self, symbols):
        """
        Add symbols to the watchlist

        Args:
            symbols (list): List of symbols to add
        """
        if not isinstance(symbols, list):
            symbols = [symbols]

        for symbol in symbols:
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                self.logger.info(f"Added {symbol} to watchlist")

        # Save updated watchlist
        self._save_watchlist()

    def remove_from_watchlist(self, symbols):
        """
        Remove symbols from the watchlist

        Args:
            symbols (list): List of symbols to remove
        """
        if not isinstance(symbols, list):
            symbols = [symbols]

        for symbol in symbols:
            if symbol in self.watchlist:
                self.watchlist.remove(symbol)
                self.logger.info(f"Removed {symbol} from watchlist")

        # Save updated watchlist
        self._save_watchlist()

    def _save_watchlist(self):
        """Save the current watchlist to file"""
        watchlist_file = self.config.get('system', {}).get('watchlist_file', 'watchlist.json')

        try:
            with open(watchlist_file, 'w') as f:
                json.dump({'symbols': self.watchlist}, f, indent=2)

            self.logger.info(f"Watchlist saved with {len(self.watchlist)} symbols")

        except Exception as e:
            self.logger.error(f"Error saving watchlist: {str(e)}")


# Main entry point
if __name__ == "__main__":
    try:
        # Create trading system
        trading_system = TradingSystem()

        # Start trading
        trading_system.start()

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()