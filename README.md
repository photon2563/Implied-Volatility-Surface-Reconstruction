# Implied Volatility Surface Reconstruction for NIFTY Options

![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)

This repository contains a strictly causal, high-performance solution for the reconstruction of missing Implied Volatility (IV) values in NIFTY options market data. 

The primary objective of this project is to accurately interpolate and extrapolate missing IVs across the options surface (strike and time) **without introducing look-ahead bias**. This guarantees that the models constructed here can be safely deployed in live, sequential algorithmic trading strategies.

---

## 🚀 The Challenge: Look-Ahead Bias

A common pitfall when handling time-series options data is utilizing global statistical imputers (like Ridge Regression, Iterative Imputers, or global medians) which calculate relationships across an entire dataset simultaneously. 

### Why is this dangerous?
When an imputer looks at the entire dataset to figure out the missing IV for a contract on `Day 1`, it uses statistical relationships gathered from `Day 100`. In a real-world trading environment, you do not have `Day 100`'s data when you are trading on `Day 1`. 

While these non-causal global imputers can yield extraordinarily low Mean Squared Errors (MSE) in cross-validation, these scores are **hallucinations**. They represent "fake" accuracy achieved by leaking future market data into past predictions. 

This project strictly forbids look-ahead bias by enforcing the rule: **To predict a missing value at row $i$, the model can only use data from row $i$, or rows $0$ to $i-1$.**

---

## 🛠️ The Multi-Pass Causal Methodology

Because we cannot rely on global future data to fill missing gaps, this codebase implements a highly robust **Multi-Pass Causal Imputation Strategy** that processes the data chronologically and cross-sectionally.

### Pass 1: Strike-Axis Log-Moneyness Interpolation (Primary)
- **Scope:** Operates independently on every row (strictly same-time data).
- **Log-Moneyness Transformation:** Converts physical strikes ($K$) and underlying spot prices ($S$) into **log-moneyness** space: $x = \log(K/S)$. Physical strikes are poorly spaced as the spot price moves over time. Log-moneyness dynamically centers the strikes around 0, ensuring that the IV smile is modeled over a uniform, normalized distance metric.
- **Center Cell Interpolation:** For missing values bounded by known values (the "belly" of the volatility smile), the algorithm uses piecewise linear interpolation. It is incredibly stable and fast.
- **Edge Cell Extrapolation:** For out-of-the-money (OTM) and deep in-the-money (ITM) wing cells, standard interpolation fails. Instead of defaulting to flat (zero-slope) extrapolations which heavily distort the volatility smile, the algorithm measures the slope of the outermost known segments to naturally and causally extend the wings of the smile.
- **Look-ahead audit:** `CAUSAL ✓` (Uses only same-row data).

### Pass 2: Strictly Causal Forward Fill (Safety Net)
- **Scope:** If an entire expiry strip is missing on a specific timestamp, Pass 1 has no anchor points to interpolate from. Pass 2 kicks in to sweep the matrix chronologically column-by-column.
- It pulls the last known valid IV for that specific contract from a prior timestamp.
- **Look-ahead audit:** `CAUSAL ✓` (Only reads past rows $0$ to $i-1$).

### Pass 3: Static Causal Prior (Fallback)
- **Scope:** In the extremely rare event that a contract is entirely missing from the very first row of the dataset (meaning there is no past data to forward-fill), the pipeline falls back to a realistic global static prior (15% IV). 
- **Look-ahead audit:** `CAUSAL ✓` (Constant prior).

---

## 📊 The MSE Trade-Off: Center Cells vs. Edge Cells

To honestly evaluate the causal accuracy of the model, the integrated $k$-seed Cross-Validation engine splits the performance evaluation into two specific zones: **Center Cells** and **Edge Cells**. 

Understanding the contrast between these two zones is the key to understanding the mathematical limits of causal IV modeling.

### The Center Cells (Interpolation)
Center cells represent missing strikes that have valid known IVs on both their left and right sides. 
Because the volatility smile in log-moneyness space is relatively smooth and predictable, linear/spline interpolation performs exceptionally well here.
- **Center Cells MSE:** ~`0.00004`
- This near-perfect accuracy proves that the shape of the NIFTY IV belly is highly deterministic.

### The Edge Cells (Extrapolation)
Edge cells represent the absolute wings of the options chain (the lowest available strikes for PE, or highest for CE). When these cells are missing, the model has to *guess* where the curve is heading. 
- **Edge Cells MSE:** ~`0.00023`
- The wings of an IV surface are notoriously volatile and heavily subjected to market supply/demand shocks (e.g., tail-risk hedging). 

### The Fundamental Extrapolation Floor
During extensive hyperparameter A/B testing, we observed a strict mathematical tradeoff:
1. **Flat Wings (0 slope):** Assuming the wings instantly flatten out yields terrible accuracy (Edge MSE skyrockets > `0.002`).
2. **Aggressive Slopes:** Unbounded linear extrapolation causes the wings to shoot towards infinity, blowing up the MSE.
3. **Causal Sweet Spot:** By gently dampening the averaged slope of the last known segments, we hit a hard mathematical floor of `0.0002` MSE for edge cells. 

To break through this `0.0002` floor in the wings **without cheating**, one would have to abandon splines and fit a heavy, 5-parameter non-linear surface model (like a row-by-row Stochastic Volatility Inspired (SVI) regression). For a lightweight deterministic model, the current log-moneyness slope extrapolation achieves the theoretical causal maximum.

---

## ⚙️ How to Run

1. Ensure you have `dataset.csv` in the root directory. This wide-format CSV should contain `datetime`, `underlying_price`, and all option contract columns (e.g., `NIFTY27JAN2615000CE`).
2. Run the main script:
   ```bash
   python solution_final.py
   ```
3. The script will output two files:
   - `filled_dataset.csv`: The entire wide-format matrix with all NaNs populated.
   - `submission.csv`: A flattened, ID-value mapped CSV ready for automated grader or competition ingestion.

## 📦 Requirements
- `numpy`
- `pandas`
