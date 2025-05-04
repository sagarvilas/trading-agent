import yfinance as yf
import logging
import pandas as pd

class DataCollector:
    """
    Responsible for gathering and preprocessing stock market data
    """

    def __init__(self, logger=None):
        """Initialize the data collector with optional custom logger"""
        self.logger = logger or logging.getLogger(__name__)

    def fetch_historical_data(self, ticker_symbols, period="1y", interval="1d"):
        """
        Fetch historical price data for the given symbols

        Args:
            ticker_symbols (list): List of ticker symbols
            period (str): Data period (e.g., '1d', '1mo', '1y')
            interval (str): Data interval (e.g., '1m', '1h', '1d')

        Returns:
            dict: Dictionary mapping symbols to their respective dataframes
        """
        self.logger.info(f"Fetching historical data for {len(ticker_symbols)} symbols")
        previous_data = {}

        for ticker_symbol in ticker_symbols:
            try:
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    self.logger.warning(f"No data found for {ticker_symbol}")
                    continue

                # Basic preprocessing
                df = self.preprocess_data(df)
                previous_data[ticker_symbol] = df
                self.logger.debug(f"Successfully fetched data for {ticker_symbol}")

            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker_symbol}: {str(e)}")

        return previous_data

    def preprocess_data(self, df):
        """
        Perform basic preprocessing on the data

        Args:
            df (DataFrame): Raw price data

        Returns:
            DataFrame: Preprocessed data
        """
        # Handle missing values
        df = df.ffill()

        # Calculate returns
        df['Returns'] = df['Close'].pct_change()

        # Calculate common technical indicators
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Relative Strength Index (simplified)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)

        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        return df

    def fetch_real_time_data(self, ticker_symbols):
        """
        Fetch the latest market data for the given symbols

        Args:
            ticker_symbols (list): List of ticker symbols

        Returns:
            dict: Dictionary mapping symbols to their respective latest data
        """
        self.logger.info(f"Fetching real-time data for {len(ticker_symbols)} symbols")
        previous_data = {}

        for ticker_symbol in ticker_symbols:
            try:
                ticker = yf.Ticker(ticker_symbol)
                latest = ticker.history(period="1d")

                if latest.empty:
                    self.logger.warning(f"No real-time data found for {ticker_symbol}")
                    continue

                previous_data[ticker_symbol] = latest.iloc[-1].to_dict()
                self.logger.debug(f"Successfully fetched real-time data for {ticker_symbol}")

            except Exception as e:
                self.logger.error(f"Error fetching real-time data for {ticker_symbol}: {str(e)}")

        return previous_data


# Usage example:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create data collector
    collector = DataCollector()

    # Fetch data for some popular stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    historical_data = collector.fetch_historical_data(symbols, period="2y")

    # Print a sample of the data
    for symbol, data in historical_data.items():
        print(f"\n=== {symbol} Data Sample ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Auto-detect terminal width
        pd.set_option('display.expand_frame_repr', False)
        print(data.tail(3))