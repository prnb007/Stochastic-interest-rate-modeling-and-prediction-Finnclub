# Stochastic Interest Rate Modelling and Prediction
## Cox-Ingersoll-Ross (CIR) Model — Finance Club, IIT Roorkee · Open Projects 2026

---

## Overview

This project implements, calibrates, and extends the **Cox-Ingersoll-Ross (CIR)** short-rate model to reconstruct the full US Treasury yield curve (6M to 30Y maturities) using **only the 3-Month rate as input**. The core challenge is a strict information constraint: at prediction time, the model is given nothing except today's 3M yield and must infer all other maturities from the model's closed-form bond pricing formulas and its calibrated parameters.

The target metric is **out-of-sample R² ≥ 0.85** across all predicted maturities.

---

## Project Structure

```
finclub/
├── CIR_Model_FinClub_IIT_Roorkee.ipynb     # Main notebook (run this)
├── CIR_Model_FinClub_IIT_Roorkee_executed.ipynb  # Pre-run copy (for reference only)
├── train_data.csv                           # Training data: all 9 tenors, 2016–2024
├── test_data.csv                            # Test actuals: 5 tenors (3M–2Y), 2024–2026
├── test_data_3M.csv                         # Prediction input: 3M only, 2024–2026
└── README.md
```

---

## Data

| File | Date Range | Rows | Columns |
|---|---|---|---|
| `train_data.csv` | 2016-05-19 → 2024-04-26 | ~1,976 trading days | 3M, 6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y |
| `test_data.csv` | 2024-04-29 → 2026-04-29 | ~495 trading days | 3M, 6M, 9M, 1Y, 2Y |
| `test_data_3M.csv` | 2024-04-29 → 2026-04-29 | ~495 trading days | 3M only |

- All yields are in **decimal form** (e.g., 0.045 = 4.5%).
- Column names in the raw CSVs use the `ZCxxxYR` format (e.g., `ZC025YR` = 3M), which the loader normalises automatically to `3M`, `6M`, etc.
- The test actuals file only contains maturities up to 2Y; the model predicts out to 30Y but is only evaluated against what's available.

---

## Mathematical Framework

### The CIR SDE

The short rate $r_t$ evolves as:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

| Parameter | Symbol | Interpretation |
|---|---|---|
| Mean-reversion speed | $\kappa > 0$ | How fast rates snap back to equilibrium |
| Long-run mean | $\theta > 0$ | The equilibrium interest rate |
| Volatility | $\sigma > 0$ | Magnitude of random shocks |

The **square-root diffusion** $\sigma\sqrt{r_t}$ is the model's key feature: noise shrinks as rates fall toward zero, making negative rates impossible (unlike the Vasicek model).

### Feller Condition

$$2\kappa\theta \geq \sigma^2$$

When satisfied, zero is an inaccessible boundary and rates remain strictly positive almost surely. When violated, rates can touch zero. The notebook monitors this condition globally and across rolling windows.

### Closed-Form Bond Pricing

Zero-coupon bond prices have an affine closed form:

$$P(t,T) = A(\tau)\,e^{-B(\tau)\,r_t}, \qquad \tau = T - t$$

$$B(\tau) = \frac{2(e^{\gamma\tau}-1)}{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}, \qquad \gamma = \sqrt{\kappa^2 + 2\sigma^2}$$

$$A(\tau) = \left[\frac{2\gamma\,e^{(\kappa+\gamma)\tau/2}}{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}\right]^{2\kappa\theta/\sigma^2}$$

The continuously compounded yield is then:

$$y(\tau) = \frac{B(\tau)\,r_t - \ln A(\tau)}{\tau}$$

This is the **core prediction equation**: given only $r_t$ (the 3M rate) and the calibrated parameters $(\kappa, \theta, \sigma)$, the full yield curve is computed analytically.

### CIR Transition Density

Given $r_t$, the next-step rate $r_{t+\Delta t}$ follows a **scaled non-central chi-squared** distribution with degrees of freedom $\nu = 4\kappa\theta/\sigma^2$ and non-centrality $\lambda = 2u$, where $c = 2\kappa / [\sigma^2(1 - e^{-\kappa\Delta t})]$ and $u = c\,r_t\,e^{-\kappa\Delta t}$. This exact density is used for maximum likelihood estimation, making MLE strictly superior to OLS (which wrongly assumes Gaussian increments, biasing $\kappa$ downward).

---

## Notebook Walkthrough (8 Sections)

### Section 0 — Setup
Imports (NumPy, Pandas, SciPy, sklearn, Matplotlib, Seaborn), global tenor constants, file paths, and plot style configuration.

### Section 1 — Mathematical Framework
Full derivation of the CIR SDE, Feller condition, zero-coupon bond pricing formulas, the non-central chi-squared transition density used in MLE, and the CIR++ extension theory.

### Section 2 — Data Loading & EDA
`YieldDataLoader` handles flexible column name normalisation — it maps dozens of naming conventions (`ZC025YR`, `dtb3`, `3month`, `0.25`, etc.) to standard labels (`3M`, `6M`, …). EDA includes:
- Time-series plots for all 9 tenors
- Sample yield curve overlays (normal, inverted, humped shapes)
- Cross-tenor correlation heatmap confirming a dominant level factor

### Section 3 — Preprocessing
`YieldPreprocessor` applies a four-step pipeline:
1. Type coercion to float
2. Time-indexed interpolation (respects unequal business-day gaps)
3. Forward/backward fill for terminal NaNs
4. Rolling z-score outlier detection (window=20, threshold=4σ) with median replacement
5. Validation: zero NaNs, all yields > 0

Also performs **date alignment** between `test_data.csv` and `test_data_3M.csv` using index intersection — this fixes the `ValueError: inconsistent numbers of samples` that arises when the two test files cover different date ranges.

### Section 4 — CIR Model Implementation
`CIRModel` class with fully vectorised methods:

| Method | Description |
|---|---|
| `yield_curve(r0, tenors)` | Yield curve for a single short rate → shape `(n_tenors,)` |
| `yield_matrix(r0_array, tenors)` | Yield matrix for n dates → shape `(n_dates, n_tenors)`, one NumPy operation |
| `simulate(r0, T, n_steps, n_paths)` | Euler-Maruyama simulation with `max(r, 0)` reflection |
| `feller_value` | $2\kappa\theta - \sigma^2$ (positive = condition satisfied) |
| `half_life_days` | $\ln(2)/\kappa \times 252$ trading days |

Three unit tests verify: (1) $\kappa \to \infty$ collapses all yields to $\theta$; (2) $B(\tau)$ is positive and monotone; (3) `yield_matrix` output shape is correct.

### Section 5 — Two-Stage MLE Calibration
`CIRCalibrator` runs two sequential optimisation stages, both using L-BFGS-B with 10 random restarts:

**Stage 1 — Time-series MLE on 3M rate:**  
Maximises the exact log-likelihood using the non-central chi-squared transition density. Provides statistically efficient, asymptotically unbiased starting parameters. Falls back to moment-matching (Euler OLS) if all restarts fail.

**Stage 2 — Cross-sectional SSE on full yield panel:**  
Minimises total squared error across all training dates × all 8 prediction tenors simultaneously. Seeds from Stage 1 result plus 9 perturbed variants (±30%). This ensures the calibrated model reproduces the full curve shape, not just the short-rate dynamics.

Post-calibration diagnostics: mean curve comparison, residual heatmap over time, and per-tenor RMSE bar chart.

### Section 6 — Prediction & Evaluation
`YieldCurvePredictor` wraps the calibrated model and supports an optional **per-maturity bias correction** $\hat{\varphi}(\tau)$:

$$\hat{\varphi}(\tau) = \frac{1}{T}\sum_{t=1}^{T}\left[y^{\text{market}}_t(\tau) - y^{\text{CIR}}_t(\tau)\right]$$

This scalar shift per tenor corrects for the term risk premium — the portion of long-term yields above the risk-neutral CIR expectation that the model structurally cannot capture. It is fitted on training data and applied at test time.

Three model variants are evaluated on the test set:

| Model | Description |
|---|---|
| Base CIR | Pure model, no correction |
| CIR + Bias | Per-maturity mean residual added |
| CIR++ | Same as bias correction, but mathematically motivated (see Section 7) |

Output: overall R², RMSE, MAE; per-tenor breakdown; time-series and scatter plots.

### Section 7 — CIR++ Extension
`CIRPlusPlusModel` implements the Brigo & Mercurio (2006) shift. The deterministic shift $\varphi(\tau)$ is defined as:

$$\varphi(\tau) = y^{\text{market}}_{\text{ref}}(\tau) - y^{\text{CIR}}_{\text{ref}}(\tau)$$

where the reference is the mean training yield curve. This is not just a heuristic correction — it has a rigorous theoretical basis:

- **Affine structure is preserved**: closed-form bond prices remain valid
- **Exact initial curve fit**: $\varphi(\tau)$ is uniquely determined from market data; no additional optimisation is needed
- **Captures the term risk premium**: the systematic component of long yields that risk-neutral CIR cannot price
- **Minimal extra parameters**: 8 (one $\varphi$ per tenor) vs. 6 continuous parameters for Two-Factor CIR or 3 extra for jump-diffusion

### Section 8 — Critical Analysis (9 Questions)

**Q1 — Parameter Sensitivity:** Yield curves are perturbed ±30% around each of $\kappa$, $\theta$, $\sigma$. Result: $\theta$ dominates the long-end level (parallel shift); $\kappa$ controls curve steepness and speed of convergence; $\sigma$ has a second-order effect through $\gamma = \sqrt{\kappa^2 + 2\sigma^2}$.

**Q2 — Feller Condition Stability:** Rolling-window recalibration across the training period. Charts whether $2\kappa\theta - \sigma^2$ stays positive. Violations indicate regimes where near-zero rates become accessible.

**Q3 — Mean-Reversion Interpretation:** Reports the half-life $\ln(2)/\kappa$ in trading days and months. Simulates impulse responses from below and above $\theta$. Compares the empirical 3M distribution to the CIR Gamma stationary distribution.

**Q4 — Per-Maturity Accuracy:** R², RMSE, and residual distributions for each tenor. Residuals are tested against normality.

**Q5 — Systematic Bias:** Mean prediction error by tenor for Base CIR vs CIR++. Base CIR structurally underestimates long-tenor yields (negative bias = missing term risk premium). CIR++ eliminates this by design.

**Q6 — Overfitting Check:** Training vs. test R² gap for Base CIR and CIR++. Since CIR++ adds only 8 parameters over hundreds of training dates, overfitting risk is negligible.

**Q7 — CIR++ Justification:** Compares CIR++ against Two-Factor CIR (rotation invariance problem, Kalman Filter requirement, 6D optimisation) and Jump-Diffusion (parameter identification from rare events, need for high-frequency data).

**Q8 — Jump Processes:** Derives qualitative effects of jumps on yield curve shape during stress events — sudden inversions, violent flattening, leptokurtic short-rate distributions. Simulates a CIR + Compound Poisson jump path and shows yield curves at calm vs. post-jump moments.

**Q9 — Two-Factor and Time-Dependent Extensions:** Explains rotation invariance and the need for Extended/Unscented Kalman Filtering in Two-Factor CIR; discusses piecewise calibration and Tikhonov regularisation challenges in fully time-dependent $\kappa(t), \theta(t)$ models.

---

## How to Run

### On Google Colab (recommended)

1. Upload `CIR_Model_FinClub_IIT_Roorkee.ipynb`, `train_data.csv`, `test_data.csv`, and `test_data_3M.csv` to your Colab session.
2. If using Google Drive, update the three path constants at the top of the Setup cell:
   ```python
   TRAIN_PATH   = '/content/drive/MyDrive/your_folder/train_data.csv'
   TEST_PATH    = '/content/drive/MyDrive/your_folder/test_data.csv'
   TEST_3M_PATH = '/content/drive/MyDrive/your_folder/test_data_3M.csv'
   ```
3. Run all cells top to bottom (**Runtime → Run all**).

### Locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
jupyter notebook CIR_Model_FinClub_IIT_Roorkee.ipynb
```

Run all cells in order.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Vectorised bond pricing, array operations |
| `pandas` | Date-indexed DataFrames, time-series interpolation |
| `scipy` | L-BFGS-B optimisation, non-central chi-squared density, Gamma distribution |
| `scikit-learn` | R², RMSE, MAE metrics |
| `matplotlib` | All visualisations |
| `seaborn` | Correlation heatmap, residual distributions |

No GPU or special hardware required. Full run takes approximately 2–5 minutes on Colab CPU.

---

## Key Design Decisions

**Why MLE over OLS?**  
OLS on the discretised SDE assumes Gaussian increments, which is wrong for CIR. It biases $\kappa$ downward. MLE uses the exact non-central chi-squared transition density and is asymptotically efficient.

**Why two-stage calibration?**  
Stage 1 (time-series MLE) gives statistically optimal parameters for the short-rate dynamics. Stage 2 (cross-sectional SSE) ensures the model also fits the full yield curve shape, which is what we actually need for prediction. Doing only Stage 2 risks getting stuck in poor local minima; doing only Stage 1 ignores the cross-sectional fit.

**Why CIR++ over Two-Factor CIR?**  
Two-Factor CIR has rotation invariance (any rotation of the two latent factors gives the same observable rate), requiring hard constraints for identification, plus a Kalman Filter for latent state estimation. CIR++ achieves exact initial curve fit with 8 extra parameters, no additional optimisation, and preserved closed-form pricing.

**Why the test date alignment?**  
The two test files (`test_data.csv` for actuals and `test_data_3M.csv` for prediction input) may cover different date ranges depending on the data source. The notebook intersects their date indices before building numpy arrays, preventing shape mismatches at the metrics computation step.

---

## Results Summary

The notebook produces a final summary cell reporting:

- Calibrated parameters: $\kappa$, $\theta$, $\sigma$, Feller value, half-life
- Out-of-sample R² for all three model variants (Base CIR, CIR+Bias, CIR++)
- Best model and whether it clears the R² > 0.85 acceptance threshold

A note is included: if CIR++ R² falls below Base CIR R², this signals a **regime shift** between the training and test periods (e.g., the 2022–2023 rate hiking cycle), which is itself a meaningful finding discussed in Section 8.

---

## References

- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). *A Theory of the Term Structure of Interest Rates.* Econometrica, 53(2), 385–407.
- Brigo, D., Mercurio, F. (2006). *Interest Rate Models — Theory and Practice.* Springer Finance. (CIR++ model: Chapter 3)
- Longstaff, F.A., Schwartz, E.S. (1992). *Interest Rate Volatility and the Term Structure.* Journal of Finance, 47(4), 1259–1282. (Two-Factor CIR)
