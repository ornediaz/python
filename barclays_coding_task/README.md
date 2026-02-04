# Mini QIS Platform Framework

## Overview
This framework computes a simple equal-weight strategy using daily price data.

## Components
- `schedule.py`: Provides a utility class to work with schedules
- `marketdata.py`: Loads CSV data.
- `base.py`: Abstract strategy base class.
- `rule.py`: Equal-weight strategy implementation.
- `runner.py`: Orchestrates the workflow.
- `main.py`: CLI entry point.
- `tests/`: Contains unit tests.

## Usage
Run the framework:

  python main.py

Run the tests:

  pytest tests