# Implied Volatility Surface Reconstruction for NIFTY Options

![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)

This repository contains a strictly causal, high-performance solution for the reconstruction of missing Implied Volatility (IV) values in NIFTY options market data. 

The primary objective of this project is to accurately interpolate and extrapolate missing IVs across the options surface (strike and time) **without introducing look-ahead bias**. This guarantees that the models constructed here can be safely deployed in live, sequential algorithmic trading strategies.

## The Challenge: Look-Ahead Bias

A common pitfall when handling time-series options data is utilizing global statistical imputers (like Ridge Regression or Iterative Imputers) which calculate the mean, median, or relationships across an entire dataset. While this yields a low Mean Squared Error (MSE) in cross-validation, it introduces severe **look-ahead bias**, allowing future market data to leak into past predictions.

This project tackles this by restricting the data available at row $i$ strictly to information that was known at or before timestamp $t_i$.

## The Solution Methodology

The code implements a highly robust **Multi-Pass Causal Imputation Strategy**:

### Pass 1: Strike-Axis Log-Moneyness Interpolation (Primary)
- Operates independently on every row.
- Converts strikes ($K$) and underlying spot prices ($S$) into **log-moneyness** space: $x = \log(K/S)$. This ensures that the IV smile is modeled over a uniform, normalized distance metric.
- Uses **Linear Interpolation** for the inner strikes (center cells/belly of the smile).
- Uses **Slope-Based Edge Extrapolation** for out-of-the-money (OTM) and deep in-the-money (ITM) wing cells. Instead of defaulting to flat extrapolations, it measures the slope of the outermost known segments to naturally extend the volatility smile.
- **Look-ahead audit**: `CAUSAL` (Uses only same-row data).

### Pass 2: Strictly Causal Forward Fill (Safety Net)
- In the rare event that an entire expiry row is missing and Pass 1 cannot anchor to any data, Pass 2 sweeps chronologically column-by-column.
- It pulls the last known valid IV for that specific contract from a prior timestamp.
- **Look-ahead audit**: `CAUSAL` (Only reads rows $0$ to $i-1$).

### Pass 3: Static Causal Prior (Fallback)
- If a contract is entirely missing from the very first row of the dataset (meaning there is nothing to forward-fill), the pipeline falls back to a realistic global static prior (15% IV). 
- **Look-ahead audit**: `CAUSAL` (Constant prior).

## Performance & Benchmarking

The codebase includes an integrated **$k$-seed Cross-Validation** engine that masks 20% of known data to simulate true missing segments.

- Automatically splits performance evaluation into **Center Cells** (interpolation) and **Edge Cells** (extrapolation).
- Ensures that adjustments don't artificially overfit to the wings at the expense of the center.
- Prints exact MSE/RMSE vs Ground Truth upon execution.

## How to Run

1. Ensure you have `dataset.csv` in the root directory. This wide-format CSV should contain `datetime`, `underlying_price`, and all option contract columns (e.g., `NIFTY27JAN2615000CE`).
2. Run the main script:
   ```bash
   python solution_final.py
   ```
3. The script will output two files:
   - `filled_dataset.csv`: The entire wide-format matrix with all NaNs populated.
   - `submission.csv`: A flattened, ID-value mapped CSV ready for automated grader or competition ingestion.

## Requirements
- `numpy`
- `pandas`

## Future Work
While this implementation focuses on a blazing-fast, linear approach, further non-linear enhancements such as **Stochastic Volatility Inspired (SVI)** parametric fitting or **Akima 1D Splines** with dampened wing slopes can be integrated into `Pass 1` for even tighter MSE floors, provided they are restricted to operate causally.
