import os
import json
import logging
import time
import hashlib
import hmac
import base64
import requests
from datetime import datetime
import configparser
from abc import ABC, abstractmethod


class TradingAPIBase(ABC):
    """
    Abstract base class for trading API integrations
    """

    def __init__(self, config_file=None, logger=None):
        """
        Initialize the API client

        Args:
            config_file (str): Path to config file with API credentials
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        """
        Load configuration from file

        Args:
            config_file (str): Path to config file

        Returns:
            dict: Configuration parameters
        """
        if config_file and os.path.exists(config_file):
            try:
                config = configparser.ConfigParser()
                config.read(config_file)

                # Get API section for this class
                api_name = self.__class__.__name__
                if api_name in config:
                    return dict(config[api_name])
                else:
                    self.logger.warning(f"No configuration found for {api_name}")
                    return {}
            except Exception as e:
                self.logger.error(f"Error loading config file: {str(e)}")
                return {}
        else:
            self.logger.warning("No config file provided or file not found")
            return {}

    @abstractmethod
    def get_account_info(self):
        """Get account information and balances"""
        pass

    @abstractmethod
    def get_positions(self):
        """Get current positions"""
        pass

    @abstractmethod
    def get_market_data(self, symbol):
        """Get market data for a symbol"""
        pass

    @abstractmethod
    def place_market_order(self, symbol, qty, side):
        """Place a market order"""
        pass

    @abstractmethod
    def place_limit_order(self, symbol, qty, price, side):
        """Place a limit order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """Cancel a pending order"""
        pass


class AlpacaAPI(TradingAPIBase):
    """
    Integration with Alpaca Trading API

    Alpaca provides commission-free stock and ETF trading API,
    making it a popular choice for algorithmic trading
    """

    def __init__(self, config_file=None, paper_trading=True, logger=None):
        super().__init__(config_file, logger)
        self.paper_trading = paper_trading

        # Set API endpoints based on paper/live
        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        self.data_url = "https://data.alpaca.markets"

        # Get API keys from config
        self.api_key = self.config.get('api_key', os.environ.get('ALPACA_API_KEY', ''))
        self.api_secret = self.config.get('api_secret', os.environ.get('ALPACA_API_SECRET', ''))

        if not self.api_key or not self.api_secret:
            self.logger.error("Alpaca API key and secret must be provided")

    def _get_headers(self):
        """
        Get HTTP headers for API authentication

        Returns:
            dict: HTTP headers
        """
        return {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }

    def _make_request(self, method, endpoint, params=None, data=None):
        """
        Make an HTTP request to the API

        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            params (dict): Query parameters
            data (dict): Request body data

        Returns:
            dict: API response
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            )

            response.raise_for_status()
            return response.json() if response.content else {}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response: {e.response.text}")
            raise

    def get_account_info(self):
        """
        Get account information and balances

        Returns:
            dict: Account information
        """
        return self._make_request('GET', '/v2/account')

    def get_positions(self):
        """
        Get current positions

        Returns:
            list: Current positions
        """
        return self._make_request('GET', '/v2/positions')

    def get_market_data(self, symbol):
        """
        Get latest market data for a symbol

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Latest quote information
        """
        params = {'symbols': symbol}
        return self._make_request('GET', '/v2/stocks/quotes/latest', params=params)

    def get_bars(self, symbols, timeframe='1D', start=None, end=None, limit=100):
        """
        Get historical price bars for symbols

        Args:
            symbols (list): List of stock symbols
            timeframe (str): Bar timeframe (1Min, 5Min, 15Min, 1H, 1D)
            start (str): Start date (YYYY-MM-DD)
            end (str): End date (YYYY-MM-DD)
            limit (int): Maximum number of bars

        Returns:
            dict: Historical bar data
        """
        params = {
            'symbols': ','.join(symbols) if isinstance(symbols, list) else symbols,
            'timeframe': timeframe,
            'limit': limit
        }

        if start:
            params['start'] = start
        if end:
            params['end'] = end

        url = f"{self.data_url}/v2/stocks/bars"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response: {e.response.text}")
            raise

    def place_market_order(self, symbol, qty, side):
        """
        Place a market order

        Args:
            symbol (str): Stock symbol
            qty (int): Quantity of shares
            side (str): Order side ('buy' or 'sell')

        Returns:
            dict: Order information
        """
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': side.lower(),
            'type': 'market',
            'time_in_force': 'gtc'
        }

        return self._make_request('POST', '/v2/orders', data=data)

    def place_limit_order(self, symbol, qty, price, side):
        """
        Place a limit order

        Args:
            symbol (str): Stock symbol
            qty (int): Quantity of shares
            price (float): Limit price
            side (str): Order side ('buy' or 'sell')

        Returns:
            dict: Order information
        """
        data = {
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'side': side.lower(),
            'type': 'limit',
            'time_in_force': 'gtc'
        }

        return self._make_request('POST', '/v2/orders', data=data)

    def place_stop_loss_order(self, symbol, qty, stop_price, side):
        """
        Place a stop loss order

        Args:
            symbol (str): Stock symbol
            qty (int): Quantity of shares
            stop_price (float): Stop price
            side (str): Order side ('buy' or 'sell')

        Returns:
            dict: Order information
        """
        data = {
            'symbol': symbol,
            'qty': qty,
            'stop_price': stop_price,
            'side': side.lower(),
            'type': 'stop',
            'time_in_force': 'gtc'
        }

        return self._make_request('POST', '/v2/orders', data=data)

    def cancel_order(self, order_id):
        """
        Cancel a pending order

        Args:
            order_id (str): Order ID

        Returns:
            dict: Empty dict on success
        """
        return self._make_request('DELETE', f'/v2/orders/{order_id}')

    def get_orders(self, status=None, limit=100):
        """
        Get list of orders

        Args:
            status (str): Order status ('open', 'closed', 'all')
            limit (int): Maximum number of orders to return

        Returns:
            list: Orders
        """
        params = {'limit': limit}
        if status:
            params['status'] = status

        return self._make_request('GET', '/v2/orders', params=params)


class IBKRAdapter(TradingAPIBase):
    """
    Integration with Interactive Brokers API via IB's REST gateway

    Requires running the IBKR REST gateway locally:
    https://interactivebrokers.github.io/cpwebapi/
    """

    def __init__(self, config_file=None, gateway_url=None, logger=None):
        super().__init__(config_file, logger)
        self.gateway_url = gateway_url or self.config.get('gateway_url', 'http://localhost:5000')
        self.session_id = None

    def _authenticate(self):
        """
        Authenticate with the IBKR gateway

        Returns:
            bool: True if authentication successful
        """
        username = self.config.get('username', '')
        password = self.config.get('password', '')

        if not username or not password:
            self.logger.error("IBKR username and password must be provided")
            return False

        try:
            response = requests.post(
                f"{self.gateway_url}/v1/api/authenticate",
                json={'username': username, 'password': password}
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('sessionID')
                return True
            else:
                self.logger.error(f"Authentication failed: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False

    def _make_request(self, method, endpoint, params=None, data=None):
        """
        Make a request to the IBKR gateway

        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict): Query parameters
            data (dict): Request body

        Returns:
            dict: Response data
        """
        if not self.session_id and endpoint != '/v1/api/authenticate':
            self._authenticate()

        url = f"{self.gateway_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}

        if self.session_id:
            headers['X-IB-Session'] = self.session_id

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data
            )

            response.raise_for_status()
            return response.json() if response.content else {}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response: {e.response.text}")
            raise

    def get_account_info(self):
        """Get account information"""
        accounts = self._make_request('GET', '/v1/api/accounts')

        if accounts and len(accounts) > 0:
            account_id = accounts[0]
            return self._make_request('GET', f'/v1/api/portfolio/{account_id}/summary')
        else:
            self.logger.error("No accounts found")
            return {}

    def get_positions(self):
        """Get current positions"""
        accounts = self._make_request('GET', '/v1/api/accounts')

        if accounts and len(accounts) > 0:
            account_id = accounts[0]
            return self._make_request('GET', f'/v1/api/portfolio/{account_id}/positions')
        else:
            self.logger.error("No accounts found")
            return []

    def get_market_data(self, symbol):
        """Get market data for a symbol"""
        # Convert symbol to IBKR contract format
        contract = {
            'symbol': symbol,
            'secType': 'STK',
            'exchange': 'SMART',
            'currency': 'USD'
        }

        return self._make_request('POST', '/v1/api/market/data', data={'contracts': [contract]})

    def place_market_order(self, symbol, qty, side):
        """Place a market order"""
        accounts = self._make_request('GET', '/v1/api/accounts')

        if not accounts or len(accounts) == 0:
            self.logger.error("No accounts found")
            return None

        account_id = accounts[0]

        # Create order
        order = {
            'account': account_id,
            'conid': self._get_contract_id(symbol),  # Would need to implement this method
            'secType': 'STK',
            'orderType': 'MKT',
            'side': side.upper(),
            'quantity': qty,
            'tif': 'DAY'
        }

        return self._make_request('POST', '/v1/api/order', data=order)

    def place_limit_order(self, symbol, qty, price, side):
        """Place a limit order"""
        accounts = self._make_request('GET', '/v1/api/accounts')

        if not accounts or len(accounts) == 0:
            self.logger.error("No accounts found")
            return None

        account_id = accounts[0]

        # Create order
        order = {
            'account': account_id,
            'conid': self._get_contract_id(symbol),  # Would need to implement this method
            'secType': 'STK',
            'orderType': 'LMT',
            'side': side.upper(),
            'quantity': qty,
            'price': price,
            'tif': 'DAY'
        }

        return self._make_request('POST', '/v1/api/order', data=order)

    def cancel_order(self, order_id):
        """Cancel a pending order"""
        return self._make_request('DELETE', f'/v1/api/order/{order_id}')

    def _get_contract_id(self, symbol):
        """
        Get the contract ID for a symbol

        This is a simplified version - in a real system you would
        need to implement a proper contract ID lookup

        Args:
            symbol (str): Stock symbol

        Returns:
            str: Contract ID
        """
        # This would need to be implemented correctly for IBKR
        # Here we're just returning a placeholder
        return "12345"  # Placeholder


# Example of implementing other broker APIs
class TDAmeritrade(TradingAPIBase):
    """TD Ameritrade API implementation (placeholder)"""

    def get_account_info(self):
        """Get account info - placeholder"""
        # Implement TD Ameritrade specific code
        pass

    def get_positions(self):
        """Get positions - placeholder"""
        # Implement TD Ameritrade specific code
        pass

    def get_market_data(self, symbol):
        """Get market data - placeholder"""
        # Implement TD Ameritrade specific code
        pass

    def place_market_order(self, symbol, qty, side):
        """Place market order - placeholder"""
        # Implement TD Ameritrade specific code
        pass

    def place_limit_order(self, symbol, qty, price, side):
        """Place limit order - placeholder"""
        # Implement TD Ameritrade specific code
        pass

    def cancel_order(self, order_id):
        """Cancel order - placeholder"""
        # Implement TD Ameritrade specific code
        pass


# Factory class to select the appropriate API client
class TradingAPIFactory:
    """Factory class to create the appropriate API client"""

    @staticmethod
    def create_client(broker_name, config_file=None, **kwargs):
        """
        Create an API client for the specified broker

        Args:
            broker_name (str): Name of the broker
            config_file (str): Path to config file
            **kwargs: Additional arguments for the API client

        Returns:
            TradingAPIBase: API client instance
        """
        broker_name = broker_name.lower()

        if broker_name == 'alpaca':
            return AlpacaAPI(config_file, **kwargs)
        elif broker_name == 'ibkr' or broker_name == 'interactivebrokers':
            return IBKRAdapter(config_file, **kwargs)
        elif broker_name == 'tdameritrade' or broker_name == 'tda':
            return TDAmeritrade(config_file, **kwargs)
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")


# Usage example:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create an API client using the factory
    # For testing with Alpaca paper trading
    try:
        client = TradingAPIFactory.create_client('alpaca', paper_trading=True)

        # Get account info
        account = client.get_account_info()
        print(f"Account: {account}")

        # Place a test order (if we have valid API keys)
        if client.api_key and client.api_secret:
            order = client.place_market_order('AAPL', 1, 'buy')
            print(f"Order placed: {order}")
    except Exception as e:
        print(f"Error: {str(e)}")