import logging
from datetime import datetime
from enum import Enum


class TradeAction(Enum):
    """Enum for possible trade actions"""
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionSize(Enum):
    """Enum for position sizing"""
    NONE = 0
    SMALL = 1  # 25% of max allocation
    MEDIUM = 2  # 50% of max allocation
    LARGE = 3  # 75% of max allocation
    FULL = 4  # 100% of max allocation


class TradingStrategy:
    """
    Implements various trading strategies based on AI predictions and risk management
    """

    def __init__(self,
                 initial_capital=100000,
                 max_position_pct=0.1,
                 stop_loss_pct=0.05,
                 take_profit_pct=0.15,
                 logger=None):
        """
        Initialize the trading strategy

        Args:
            initial_capital (float): Starting capital
            max_position_pct (float): Maximum percentage of capital per position
            stop_loss_pct (float): Default stop loss percentage
            take_profit_pct (float): Default take profit percentage
            logger: Optional logger instance
        """
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.logger = logger or logging.getLogger(__name__)

        # Portfolio state
        self.positions = {}  # Symbol -> {qty, entry_price, stop_loss, take_profit}
        self.trade_history = []

    def calculate_position_size(self, symbol, price, signal_strength, confidence):
        """
        Calculate the appropriate position size based on signal strength and model confidence

        Args:
            symbol (str): Stock symbol
            price (float): Current price
            signal_strength (int): Signal strength (-2 to 2)
            confidence (float): Model confidence (0 to 1)

        Returns:
            tuple: (PositionSize, number of shares, dollar amount)
        """
        # Already have a position in this stock?
        if symbol in self.positions:
            return PositionSize.NONE, 0, 0

        # Determine position size category based on signal and confidence
        size_category = PositionSize.NONE

        if signal_strength == 2 and confidence > 0.7:
            size_category = PositionSize.FULL
        elif signal_strength == 2 and confidence > 0.6:
            size_category = PositionSize.LARGE
        elif signal_strength == 1 and confidence > 0.65:
            size_category = PositionSize.MEDIUM
        elif signal_strength == 1 and confidence > 0.55:
            size_category = PositionSize.SMALL
        else:
            return PositionSize.NONE, 0, 0

        # Calculate dollar amount based on position size category
        position_pct = self.max_position_pct
        if size_category == PositionSize.SMALL:
            position_pct *= 0.25
        elif size_category == PositionSize.MEDIUM:
            position_pct *= 0.5
        elif size_category == PositionSize.LARGE:
            position_pct *= 0.75

        # Calculate dollar amount and shares
        dollar_amount = self.capital * position_pct
        shares = int(dollar_amount / price)

        # Adjust dollar amount based on actual shares
        actual_amount = shares * price

        return size_category, shares, actual_amount

    def generate_trade_signal(self, symbol, data, prediction, current_price):
        """
        Generate a trading signal based on the model prediction and current market data

        Args:
            symbol (str): Stock symbol
            data (DataFrame): Recent market data
            prediction (dict): Model prediction output
            current_price (float): Current price of the stock

        Returns:
            tuple: (TradeAction, details)
        """
        # Extract prediction values
        direction_prob = prediction['Direction_Probability']
        predicted_change = prediction['Predicted_Change']
        signal = prediction['Signal']

        # No position currently - check for potential entry
        if symbol not in self.positions:
            # Strong buy signals
            if signal >= 1:
                # Calculate position size
                size_category, shares, amount = self.calculate_position_size(
                    symbol, current_price, signal, direction_prob
                )

                if size_category != PositionSize.NONE:
                    # Calculate stop loss and take profit
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)

                    # Adjust stop loss based on recent volatility
                    if len(data) >= 20:
                        volatility = data['Close'].pct_change().std() * 16  # ~2 std deviations
                        adjusted_stop = current_price * (1 - volatility)
                        stop_loss = max(adjusted_stop, stop_loss)  # Use the higher of the two

                    return TradeAction.BUY, {
                        'shares': shares,
                        'amount': amount,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': direction_prob,
                        'predicted_change': predicted_change,
                        'size_category': size_category
                    }

            return TradeAction.HOLD, {'reason': 'No buy signal'}

        else:
            # We have a position - check exit conditions
            position = self.positions[symbol]
            entry_price = position['entry_price']

            # Check if stop loss or take profit hit
            if current_price <= position['stop_loss']:
                return TradeAction.SELL, {'reason': 'Stop loss triggered',
                                          'profit_pct': (current_price / entry_price - 1) * 100}

            if current_price >= position['take_profit']:
                return TradeAction.SELL, {'reason': 'Take profit triggered',
                                          'profit_pct': (current_price / entry_price - 1) * 100}

            # Check for strong sell signals
            if signal <= -1 and direction_prob < 0.4:
                return TradeAction.SELL, {'reason': 'Sell signal',
                                          'profit_pct': (current_price / entry_price - 1) * 100}

            # Consider time-based exit (position age)
            days_held = (datetime.now() - position['entry_date']).days
            if days_held > 10 and current_price > entry_price:
                return TradeAction.SELL, {'reason': 'Time-based exit (profitable)',
                                          'profit_pct': (current_price / entry_price - 1) * 100}

            return TradeAction.HOLD, {'reason': 'Maintaining position'}

    def execute_trade(self, symbol, action, details, current_price, timestamp=None):
        """
        Execute a trade based on the signal

        Args:
            symbol (str): Stock symbol
            action (TradeAction): Action to take
            details (dict): Additional details about the trade
            current_price (float): Current price
            timestamp (datetime): Timestamp of the trade

        Returns:
            dict: Trade result
        """
        timestamp = timestamp or datetime.now()

        if action == TradeAction.BUY:
            shares = details['shares']
            amount = shares * current_price

            # Check if we have enough capital
            if amount > self.capital:
                self.logger.warning(f"Insufficient capital for {symbol} trade. Need ${amount}, have ${self.capital}")
                return {'success': False, 'reason': 'Insufficient capital'}

            # Record the position
            self.positions[symbol] = {
                'qty': shares,
                'entry_price': current_price,
                'stop_loss': details['stop_loss'],
                'take_profit': details['take_profit'],
                'entry_date': timestamp,
                'size_category': details['size_category']
            }

            # Update capital
            self.capital -= amount

            # Record the trade
            trade_record = {
                'symbol': symbol,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'amount': amount,
                'timestamp': timestamp,
                'stop_loss': details['stop_loss'],
                'take_profit': details['take_profit'],
                'confidence': details.get('confidence'),
                'predicted_change': details.get('predicted_change')
            }

            self.trade_history.append(trade_record)
            self.logger.info(f"BUY {shares} shares of {symbol} at ${current_price:.2f}")

            return {'success': True, 'trade': trade_record}

        elif action == TradeAction.SELL:
            # Check if we have the position
            if symbol not in self.positions:
                self.logger.warning(f"Cannot sell {symbol} - no position")
                return {'success': False, 'reason': 'No position'}

            position = self.positions[symbol]
            shares = position['qty']
            entry_price = position['entry_price']
            amount = shares * current_price
            profit = amount - (shares * entry_price)
            profit_pct = (current_price / entry_price - 1) * 100

            # Update capital
            self.capital += amount

            # Record the trade
            trade_record = {
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'amount': amount,
                'timestamp': timestamp,
                'entry_price': entry_price,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': details.get('reason', 'Not specified')
            }

            self.trade_history.append(trade_record)
            self.logger.info(f"SELL {shares} shares of {symbol} at ${current_price:.2f} " +
                             f"(P&L: ${profit:.2f}, {profit_pct:.2f}%)")

            # Remove the position
            del self.positions[symbol]

            return {'success': True, 'trade': trade_record}

        else:  # HOLD
            return {'success': True, 'action': 'HOLD', 'reason': details.get('reason', 'Not specified')}

    def get_portfolio_summary(self):
        """
        Get a summary of the current portfolio

        Returns:
            dict: Portfolio summary
        """
        positions_value = 0
        positions_summary = []

        for symbol, position in self.positions.items():
            # For a real system, you'd get the current price from the market
            # Here we'll simulate with the entry price
            current_price = position['entry_price']  # This would come from real-time data
            value = position['qty'] * current_price
            positions_value += value

            positions_summary.append({
                'symbol': symbol,
                'shares': position['qty'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'value': value,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'profit_pct': (current_price / position['entry_price'] - 1) * 100
            })

        return {
            'cash': self.capital,
            'positions_value': positions_value,
            'total_value': self.capital + positions_value,
            'positions': positions_summary
        }

    def get_trade_history_summary(self):
        """
        Get a summary of trade history statistics

        Returns:
            dict: Trade statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }

        # Filter completed trades (pairs of buy and sell)
        completed_trades = [t for t in self.trade_history if t['action'] == 'SELL']

        if not completed_trades:
            return {
                'total_trades': len([t for t in self.trade_history if t['action'] == 'BUY']),
                'open_positions': len(self.positions),
                'profitable_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }

        # Calculate statistics
        profitable_trades = [t for t in completed_trades if t['profit'] > 0]
        losing_trades = [t for t in completed_trades if t['profit'] <= 0]

        win_rate = len(profitable_trades) / len(completed_trades) if completed_trades else 0
        avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
        avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_profit_pct = sum(t['profit_pct'] for t in profitable_trades) / len(
            profitable_trades) if profitable_trades else 0
        avg_loss_pct = sum(t['profit_pct'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            'total_trades': len(completed_trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_profit_pct': avg_profit_pct,
            'avg_loss_pct': avg_loss_pct,
            'open_positions': len(self.positions)
        }


# Usage example:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create trading strategy
    strategy = TradingStrategy(initial_capital=100000)

    # Simulate some trades
    symbol = 'AAPL'
    price = 175.0

    # Buy signal
    buy_details = {
        'shares': 100,
        'amount': 17500,
        'stop_loss': price * 0.95,
        'take_profit': price * 1.1,
        'confidence': 0.75,
        'predicted_change': 2.5,
        'size_category': PositionSize.MEDIUM
    }

    result = strategy.execute_trade(symbol, TradeAction.BUY, buy_details, price)
    print(f"Buy trade result: {result}")

    # Portfolio after buy
    portfolio = strategy.get_portfolio_summary()
    print(f"Portfolio after buy: {portfolio}")

    # Simulate price increase and sell
    new_price = 190.0
    sell_details = {'reason': 'Take profit triggered', 'profit_pct': 8.57}

    result = strategy.execute_trade(symbol, TradeAction.SELL, sell_details, new_price)
    print(f"Sell trade result: {result}")

    # Trade history
    history = strategy.get_trade_history_summary()
    print(f"Trade history: {history}")