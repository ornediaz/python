from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from base import Strategy
from schedule import Schedule

AssetData = Dict[str, float]

@dataclass(frozen=True)
class EqualWeightStrategyState:
    """
    Represents the state of an equal weight strategy at a specific point in time.
    
    Attributes:
        returns: Dictionary mapping asset names to their returns for the period
        portfolio_return: The overall portfolio return for the period
        index_level: The current level/value of the index
        weights: Dictionary mapping asset names to their current portfolio weights
    """
    returns: AssetData
    portfolio_return: float
    index_level: float
    weights: AssetData

@dataclass(frozen=True)
class EqualWeightStrategy(Strategy[EqualWeightStrategyState]):
    """
    An equal weight index strategy that rebalances monthly.
    
    This strategy maintains equal weights across all assets in the basket,
    rebalancing at the end of each month to restore equal weighting.
    
    Attributes:
        basket: List of asset identifiers (tickers) in the strategy
        seed_date: The starting date for index calculation
        calendar: Schedule of valid trading dates
        initial_index_level: Starting value of the index (e.g., 100.0)
    """
    basket: List[str]
    seed_date: date
    calendar: Schedule
    initial_index_level: float

    def resolve_dates(self, from_date: Optional[date], to_date: date) -> Schedule:
        """
        Get a schedule of dates within the specified range.
        
        Args:
            from_date: Start date (defaults to seed_date if None)
            to_date: End date (inclusive)
            
        Returns:
            Schedule: Sub-schedule containing dates in the specified range
        """
        if from_date is None:
            from_date = self.seed_date
        
        return self.calendar.sub_schedule(from_date, to_date)

    def compute_state(self, date: date) -> EqualWeightStrategyState:
        """
        Compute the index state for a given date.
        
        This method incrementally calculates the index state by:
        1. Starting from the seed date with initial conditions
        2. Computing daily returns for each asset
        3. Calculating portfolio return using previous day's weights
        4. Updating index level based on portfolio return
        5. Rebalancing weights to equal weight at month-end
        
        Args:
            date: The date for which to compute the index state
            
        Returns:
            EqualWeightStrategyState: The complete state of the strategy on the given date
        """
        if date == self.seed_date:
            # Base case: return initial state at seed date
            return EqualWeightStrategyState(
                returns={asset: 0.0 for asset in self.basket},
                portfolio_return=0.0,
                index_level=self.initial_index_level,
                weights={asset: 1/len(self.basket) for asset in self.basket},
            )

        # Incremental case: compute based on previous day
        prev_date = self.calendar.prev(date)
        prev_state = self.compute_state(prev_date)

        # Calculate daily returns for each asset: (today_price / yesterday_price) - 1
        returns = {
            asset: self.md.get(date, asset) / self.md.get(prev_date, asset) - 1
            for asset in self.basket
        }

        # Calculate portfolio return as weighted sum of asset returns
        portfolio_return = sum([returns[asset] * weight for asset, weight in prev_state.weights.items()])
        index_level = prev_state.index_level * (1 + portfolio_return)

        # Rebalance weights at end of month, otherwise let them drift
        if self.calendar.is_last_day_of_month(date):
            # Rebalance to equal weights (1/n for each asset)
            weights = {asset: 1/len(self.basket) for asset in self.basket}
        else:
            # Recalculate weights based on price movements
            # Each weight is adjusted by the return of that asset, normalized to sum to 1
            weights = {
                asset: prev_state.weights[asset] * (1 + returns[asset]) / (1 + portfolio_return)
                for asset in self.basket
            }

        # Return the calculated state
        return EqualWeightStrategyState(
            returns=returns,
            portfolio_return=portfolio_return,
            index_level=index_level,
            weights=weights,
        )
