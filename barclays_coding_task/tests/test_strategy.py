from datetime import date
from typing import List
from marketdata import MarketData
from rule import EqualWeightStrategy
from runner import get_states

def compute_and_check(strategy: EqualWeightStrategy, final_date: str, expected: float):
    final_level = strategy.compute_state(date.fromisoformat(final_date)).index_level
    assert round(final_level, 6) == expected, f"Index level to 6dp on {final_date} should be {expected}, got {final_level}"

def get_states_and_check(strategy: EqualWeightStrategy, from_date: str, to_date: str, expected: List[float]):
    states = get_states(strategy, date.fromisoformat(from_date), date.fromisoformat(to_date)).values()
    levels = [round(state.index_level, 6) for state in states]
    assert levels == expected

def initialise() -> EqualWeightStrategy:
    md = MarketData('sample_prices.csv')
    strategy = EqualWeightStrategy(
        md=md,
        basket=["SPX", "SX5E", "HSI"],
        seed_date=date.fromisoformat("2023-01-02"),
        calendar=md.get_calendar(),
        initial_index_level=100,
    )
    return strategy


def test_strategy_calculation():
    strategy = initialise()
    compute_and_check(strategy, "2023-01-03", 100.066461)
    compute_and_check(strategy, "2023-01-31", 93.227305)
    compute_and_check(strategy, "2023-02-01", 92.277544)
    compute_and_check(strategy, "2023-05-19", 92.441678)

def test_calculate_range():
    strategy = initialise()
    get_states_and_check(strategy, "2023-02-05", "2023-02-08", [94.098372, 93.541086, 92.601076])
