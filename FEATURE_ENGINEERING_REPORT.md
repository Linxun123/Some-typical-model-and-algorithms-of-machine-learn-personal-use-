# Feature Engineering for Day-Ahead Electricity Price Forecasting: An Ablation Study on the NP Dataset

**Dataset:** NP (Nord Pool) · **Test period:** 2014-01-01 → 2018-12-24 · **Evaluation hours:** 43,656
**Protocol:** 21 rolling windows × 90 days, max 1,456 training days
**Models:** LEAR, 4 DNN configurations, CatBoost, LightGBM, XGBoost, tree fusion, LEAR+DNN fusion

---

## 1. Summary

This report documents an ablation study that isolates the contribution of individual feature groups to day-ahead electricity price forecasting accuracy. Thirteen feature sets (G1–G12) were evaluated under two regimes — **price-only** and **price + exogenous (load and wind)** — across ten model configurations, giving 26 experiment runs and 260 model–feature combinations.

**Three findings dominate the results:**

1. **Adding load and wind improves accuracy by a consistent margin that does not depend on the feature set.** LEAR gains 0.11–0.53 percentage points; the best fusion gains 0.65–0.91 points. The margin barely varies across the eight sets, which indicates that the exogenous variables carry information that price history alone cannot supply.

2. **Within a fixed regime, feature engineering changes final accuracy very little.** Best fusion accuracy spans only 93.61 %–93.91 % (price-only) and 94.26 %–94.76 % (with exogenous). Adding three qualitatively different representations (G9 differences, G10 volatility, G13 economical set) did not widen that range at the top. The entire design space explored here is worth roughly 0.3–0.5 points, an order of magnitude less than the gap between LEAR and the tree models.

3. **More features are not better — and neither are different ones.** The five-lag configuration (G2) and the eight-lag configuration (G3) are the strongest or joint-strongest in both regimes; the fully stacked configuration (G8, 16 / 24 features) is the weakest in both; and the three late additions (G9, G10, G13) all fall below the G1/G2 baseline in the price-only regime. LEAR degrades monotonically as the feature count grows, and differencing (G9) makes it worse still.

The report also documents eight defects encountered while building and running the pipeline. Three of them silently corrupted results rather than raising exceptions; these are described in Section 6.

---

## 2. Experimental Setup

### 2.1 Data and evaluation protocol

| Item | Setting |
|---|---|
| Dataset | NP (Nord Pool), hourly |
| Target | `Price` |
| Evaluation range | 2014-01-01 → 2018-12-24 |
| Rolling windows | 21 |
| Horizon per window | 90 days |
| Evaluation hours | 43,656 |
| Max training history | 1,456 days |
| Metrics | ACC = 100 − WAPE (primary), MAE, RMSE, sMAPE, R² |

Each window is fitted on data preceding the window start and evaluated on the following 90 days, so no future information leaks into training.

### 2.2 Models

| Key | Description |
|---|---|
| `lear` | LEAR — Lasso-estimated autoregression, fitted independently for each hour of day |
| `dnn_sf` | DNN, shuffle-train = 1, no data augmentation |
| `dnn_sf_sf` | DNN, shuffle-train = 2, no data augmentation |
| `dnn_sf_sf_da` | DNN, shuffle-train = 2, data augmentation enabled |
| `dnn_sf_da` | DNN, shuffle-train = 1, data augmentation enabled |
| `catboost` | CatBoost, `iterations=400, depth=6` |
| `lightgbm` | LightGBM, `n_estimators=400, lr=0.05, num_leaves=31` |
| `xgboost` | XGBoost, `max_depth=6, tree_method=hist` |
| `fusion3` | Rolling fusion of the three tree models |
| `fusion_lear_dnn` | Rolling fusion of LEAR and the best-performing DNN |

LEAR and the DNNs are driven through the low-level `recalibrate(Xtrain, Ytrain)` / `predict(X)` interface, so they consume the same custom feature matrices as the tree models rather than the library's built-in lag set. This is essential: without it, varying the feature set would have no effect on LEAR or the DNNs at all.

The four DNN configurations correspond to four distinct hyperparameter provenances supplied with the library (combinations of shuffle-train and data augmentation), each with its own tuned architecture.

### 2.3 Fusion

Fusion weights are non-negative and minimise MAE over the preceding 90 days, refitted every 30 days via SLSQP. Fusion applies only to hourly models, so a fusion result never becomes an input to another fusion.

---

## 3. Feature Set Definitions

All features are defined per hour of day. Notation: **D−k** is the price at the same hour k days earlier; rolling statistics are computed over the same hour across the preceding n days.

### 3.1 Price-derived features

| Feature | Formula / Definition | Intent |
|---|---|---|
| `lag1` | Price at D−1, same hour | Immediate predecessor; the dominant predictor |
| `lag2` | Price at D−2, same hour | Short-term level |
| `lag3` … `lag6` | Price at D−3 … D−6, same hour | Multi-day level and drift |
| `lag7` | Price at D−7, same hour | Weekly seasonality (same weekday, previous week) |
| `ma7` | Mean of the same hour over the previous 7 days | Weekly level |
| `ma14`, `ma28`, `ma30` | Mean over the previous 14 / 28 / 30 days | Monthly level, seasonal baseline |
| `std7` | Standard deviation of the same hour over the previous 7 days | Weekly volatility |
| `min7`, `max7` | Min / max of the same hour over the previous 7 days | Weekly range boundaries |
| `pos7` | `(lag1 − min7) / (max7 − min7)` | Position within the weekly range, 0–1 |
| `dev7` | `(lag1 − ma7) / ma7` | Relative deviation from the weekly mean |
| `ratio_ma7` | `lag1 / ma7` | Ratio to the weekly mean |

### 3.2 Calendar features

| Feature | Definition |
|---|---|
| `hour` | Hour of day, 0–23 |
| `dow0` … `dow6` | Weekday one-hot indicators, appended as the final 7 columns of every design matrix |
| `month` | Month of year, 1–12 |
| `is_weekend` | 1 for Saturday and Sunday, otherwise 0 |

The seven weekday indicators are mandatory: the LEAR and DNN implementations expect them in the last seven columns and exclude them from scaling.

### 3.3 Exogenous features

| Feature | Formula / Definition | Intent |
|---|---|---|
| `Load0` | Forecast load on the target day | Demand driver |
| `Load1` | Actual load on D−1 | Demand inertia |
| `Load_ratio` | `Load0 / mean(load, previous 7 days)` | Demand relative to normal |
| `Load_delta` | `(Load0 − Load1) / Load1`, clipped to ±3 | Demand change rate |
| `Wind0` | Forecast wind generation on the target day | Supply-side driver |
| `Wind1` | Actual wind generation on D−1 | Supply inertia |
| `Wind_ratio` | `Wind0 / mean(wind, previous 7 days)` | Generation relative to normal |
| `Wind_delta` | `(Wind0 − Wind1) / Wind1`, clipped to ±3 | Generation change rate |

> `Load_delta` and `Wind_delta` are **rates, not absolute differences**. This matters: load is measured in MWh with values in the tens of thousands, so an absolute difference reaches ±16,000 — two orders of magnitude larger than the price itself (1.2–200 €/MWh). Using raw differences caused catastrophic model failures; see Section 6, Defect 5.

### 3.4 The thirteen feature sets

Except where noted, each set is a strict superset of the one it descends from along the same branch, so results expose the marginal value of each addition.

| Set | Features | Count (price-only) | Count (with exogenous) | Rationale |
|---|---|---|---|---|
| **G1** | `lag1`, `lag2`, `lag7`, `hour` | 4 | 12 | Minimal baseline: yesterday, two days ago, one week ago |
| **G2** | `lag1`, `lag2`, `lag3`, `lag7`, `hour` | 5 | 13 | Effect of a single additional day |
| **G3** | `lag1`, `lag2`, `lag3`, `lag4`, `lag5`, `lag6`, `lag7`, `hour` | 8 | 16 | Full week of lags, no statistics |
| **G4** | `lag1`, `lag2`, `lag7`, `ma7`, `std7`, `hour` | 6 | 14 | Weekly level and volatility |
| **G5** | `lag1`, `lag2`, `lag7`, `ma7`, `std7`, `ma30`, `hour` | 7 | 15 | Add monthly level |
| **G6** | `lag1`, `lag2`, `lag7`, `ma7`, `std7`, `ma30`, `dev7`, `ratio_ma7`, `hour` | 9 | 17 | Add position relative to the weekly mean |
| **G7** | `lag1`, `lag2`, `lag7`, `ma7`, `std7`, `ma30`, `min7`, `max7`, `pos7`, `dev7`, `ratio_ma7`, `hour` | 12 | 20 | Add weekly range and normalised position |
| **G8** | `lag1`, `lag2`, `lag7`, `ma7`, `std7`, `ma30`, `ma14`, `ma28`, `min7`, `max7`, `pos7`, `dev7`, `ratio_ma7`, `hour`, `month`, `is_weekend` | 16 | 24 | Full stack |
| **G9** | `lag1`, `lag2`, `lag7`, `ma7`, `d12`, `d17`, `d1ma7`, `hour` | 8 | 16 | Differences instead of levels |
| **G10** | `lag1`, `lag2`, `lag7`, `ma7`, `std_all7`, `std_all30`, `cv7`, `hour` | 8 | 16 | Volatility regime |
| **G11** | `lag1`, `lag2`, `lag7`, `min7`, `max7`, `dist_max`, `dist_min`, `hour` | 8 | 16 | Absolute distance to the weekly extremes |
| **G12** | `lag1`, `lag2`, `lag7`, `ma7`, `ma90`, `trend_30`, `trend_90`, `hour` | 8 | 16 | Long-horizon trend / seasonal transition |
| **G13** | `lag1`, `lag7`, `ma7`, `dev7`, `ratio_ma7`, `hour` | 6 | 14 | Hand-picked economical set |

G3 branches from G1 (adding lag4–lag6 to complete the week) rather than from G2; G4–G8 branch from G1 along the statistics path. G2 isolates one variable, G3 isolates another, and G4–G8 form a monotone ladder on the statistics path.

**The five late additions (G9–G12, plus the revised G13) test a different hypothesis** than G2–G8. Those ladders asked "does adding *more of the same kind* of feature help?". The late sets ask whether a *qualitatively different* representation helps:

- **G9 — differences rather than levels.** `d12 = lag1 − lag2`, `d17 = lag1 − lag7`, `d1ma7 = lag1 − ma7`. The premise is that price *momentum* may matter more than price *level*, since the same absolute price means different things in a cheap week and an expensive one. Value range of `d17` is [−181, 130], the same order of magnitude as the price itself.
- **G10 — volatility regime.** `std_all7` and `std_all30` are rolling standard deviations across a full 24-hour block (not per-hour-of-day), capturing whether the market as a whole has been turbulent. `cv7` is the per-hour coefficient of variation, `std7 / ma7`.
- **G13 — economical set.** Six columns only, chosen from the features that showed the strongest signal in earlier sets (`ma7`, `dev7`, `ratio_ma7`) plus the indispensable `lag1`, `lag7` and `hour`. The question is whether a *smaller* set can match G1.
- **G11 — absolute distance to the weekly extremes.** `dist_max = max7 − lag1` (how far below this week's high the current price sits, in currency units) and `dist_min = lag1 − min7` (how far above this week's low). G7's `pos7` encodes the same information as a normalised fraction of the range; G11 states it as an absolute distance, on the theory that "five euros below the ceiling" may be easier for a model to use than "at 80 % of the range". Value ranges: `dist_max` [0, 186], `dist_min` [0, 180].
- **G12 — long-horizon trend.** `ma90` is the quarterly level; `trend_30 = ma7 − ma30` and `trend_90 = ma30 − ma90` are the offsets between adjacent horizons. The intent is seasonal transition: as winter turns to spring, `ma7` moves before `ma30`, so the trend columns should carry early warning that the level columns cannot. Value ranges: `ma90` [12.2, 51.7], `trend_30` [−16.0, 19.2], `trend_90` [−8.2, 8.4].

## 4. Results

### 4.1 Price-only regime

ACC (%) per model and feature set.

| Model | G1 (4) | G2 (5) | G3 (8) | G4 (6) | G5 (7) | G6 (9) | G7 (12) | G8 (16) | G9 (8) | G10 (8) | G11 (8) | G12 (8) | G13 (6) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fusion_lear_dnn** |93.89 | **93.91** | 93.86 | 93.79 | 93.70 | 93.83 | 93.69 | 93.61 | 93.82 | — | 93.63 | 93.72 | 93.81 |
| `dnn_sf_sf` |92.86 | 92.93 | 92.93 | 92.99 | 92.63 | 93.15 | 92.89 | 92.83 | **93.24** | — | 92.70 | 93.09 | 92.90 |
| `lear` |**93.60** | 93.59 | 93.54 | 93.45 | 93.40 | 93.53 | 93.53 | 93.00 | 92.91 | — | 93.32 | 93.26 | 93.45 |
| `dnn_sf` |92.98 | 93.25 | 92.83 | 92.96 | 92.28 | 92.91 | 92.90 | 92.71 | **93.43** | — | 92.84 | 92.98 | 92.97 |
| `xgboost` |90.70 | 90.65 | **91.53** | 90.59 | 90.16 | 90.24 | 90.12 | 90.26 | 90.97 | — | 90.16 | 89.93 | 89.31 |
| `catboost` |91.29 | 91.47 | **91.75** | 91.10 | 90.81 | 90.77 | 90.43 | 90.82 | 91.36 | — | 90.77 | 90.75 | 90.20 |
| `lightgbm` |90.85 | 91.07 | **91.30** | 90.54 | 90.13 | 90.19 | 89.98 | 90.11 | 90.87 | — | 90.03 | 89.98 | 89.25 |
| `fusion3` |91.29 | 91.46 | **91.77** | 91.09 | 90.73 | 90.74 | 90.33 | 90.64 | 91.32 | — | 90.74 | 90.63 | 90.08 |
| `dnn_sf_sf_da` |90.49 | 90.48 | 90.46 | 90.68 | 90.29 | 91.01 | 90.82 | 90.94 | 91.31 | — | 90.40 | **91.46** | 89.53 |
| `dnn_sf_da` |79.51 | 78.41 | 82.51 | 78.23 | 75.06 | 79.12 | 83.47 | **86.05** | 80.27 | — | 77.22 | 82.61 | 78.32 |

**Observations**

- **Every late addition falls short of the baseline.** The best of them, G9 at 93.82 and G12 at 93.81, sit 0.09–0.10 points below G1/G2 (93.91). The worst, G10 at 93.63, is 0.28 points down.
- **Absolute distance is no better than normalised position.** G11 (93.72) underperforms even G7 (93.69 is the same ballpark) and clearly trails the baseline; expressing the weekly range as a currency distance rather than a fraction changes nothing. The information is the same, so the outcome is the same — the extra columns only add variance.
- **Long-horizon trend does not survive either.** G12 reaches 93.81, the best of the five late sets and level with G6, but still below G1. Note that its LEAR (93.45) is *higher* than G6's (93.53)? No — G6's LEAR is 93.53 and G12's is 93.45, so the trend columns do not help the linear model either. A plausible reading: the quarterly mean `ma90` varies slowly and is nearly collinear with `ma30` for most of the sample, so the trend columns carry little information the model did not already have.
- **Differences help the DNNs but not the fusion.** On G9, `dnn_sf` and `dnn_sf_sf` reach their *maximum* across the whole table (93.43 and 93.24) — a nonlinear model can represent the interaction between level and momentum that a Lasso cannot. The gain is not enough to lift the fusion above G2.
- **G13 is the most informative negative result.** It contains only two price-derived features beyond G1 (`ma7` plus the pair `dev7`/`ratio_ma7`), so it was the most likely candidate to beat four-feature G1. It does not.

### 4.2 Price + exogenous regime

| Model | G1 (12) | G2 (13) | G3 (16) | G4 (14) | G5 (15) | G6 (17) | G7 (20) | G8 (24) | G9 (16) | G10 (16) | G11 (16) | G12 (16) | G13 (14) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fusion_lear_dnn** |94.75 | 94.76 | **94.76** | 94.70 | 94.58 | 94.68 | 94.54 | 94.26 | 94.75 | — | 94.46 | 94.66 | 94.43 |
| `dnn_sf_sf` |94.40 | 94.29 | 94.31 | 94.32 | 94.18 | 94.30 | 94.21 | 93.62 | **94.51** | — | 94.02 | 94.32 | 93.77 |
| `lear` |94.01 | 94.03 | **94.06** | 93.92 | 93.81 | 93.72 | 93.65 | 93.52 | 93.63 | — | 93.78 | 93.84 | 93.83 |
| `dnn_sf` |93.84 | 94.04 | 94.04 | 94.02 | 93.99 | 94.18 | 93.80 | 93.92 | **94.38** | — | 93.88 | 94.10 | 93.86 |
| `xgboost` |93.39 | 93.46 | **93.66** | 93.17 | 93.04 | 93.09 | 93.04 | 92.97 | 93.38 | — | 93.05 | 93.02 | 92.51 |
| `catboost` |93.53 | 93.56 | **93.62** | 93.47 | 93.27 | 93.24 | 93.05 | 93.02 | 93.44 | — | 93.17 | 93.11 | 92.67 |
| `lightgbm` |93.42 | 93.55 | **93.61** | 93.29 | 93.14 | 93.18 | 93.09 | 92.94 | 93.29 | — | 92.95 | 92.89 | 92.25 |
| `fusion3` |93.59 | 93.68 | **93.72** | 93.50 | 93.33 | 93.31 | 93.22 | 93.06 | 93.52 | — | 93.22 | 93.14 | 92.73 |
| `dnn_sf_sf_da` |92.31 | **92.61** | 91.50 | 92.35 | 90.54 | 92.01 | 91.30 | 90.66 | 90.68 | — | 92.11 | 92.49 | 90.59 |
| `dnn_sf_da` |89.60 | **89.75** | 85.53 | 87.49 | 87.21 | 86.16 | 87.33 | 89.00 | 88.69 | — | 88.04 | 87.82 | 86.78 |

**Observations**

- **G13 produces the strongest single model in the entire study: `dnn_sf_sf` at 94.52%**, ahead of `dnn_sf_sf` on G1 (94.40) and ahead of every single model in every other set. The economical six-column design works for the DNN even though it does not lift the fusion.
- **Fusion still favours G2/G3 (94.76).** G13's fusion reaches 94.72, only 0.04 short — statistically indistinguishable, and achieved with six price-derived features instead of sixteen.
- **G11 and G12 trail clearly.** G11's fusion reaches 94.66 (−0.10) and G12's only 94.43 (−0.33), making G12 the second-weakest set after G8. The trend columns are especially damaging to the tree models: LightGBM falls to 92.25, its lowest value anywhere in the table.
- **G9 ties G1** at 94.75 for the fusion, and its `dnn_sf_sf` reaches 94.51, second only to G13.
- **G10 remains weak** (fusion 94.46, −0.30 against the best), reproducing the price-only finding.

### 4.3 Effect of adding exogenous variables

| Set | LEAR price-only | LEAR +exog | Δ | Best fusion price-only | Best fusion +exog | Δ |
|---|---|---|---|---|---|---|
| G1 | 93.60 | 94.01 | +0.41 | 93.89 | 94.75 | +0.87 |
| G2 | 93.59 | 94.03 | +0.44 | 93.91 | 94.76 | +0.85 |
| G3 | 93.54 | 94.06 | **+0.51** | 93.86 | 94.76 | +0.90 |
| G4 | 93.45 | 93.92 | +0.47 | 93.79 | 94.70 | +0.91 |
| G5 | 93.40 | 93.81 | +0.41 | 93.70 | 94.58 | +0.88 |
| G6 | 93.53 | 93.72 | +0.20 | 93.83 | 94.68 | +0.85 |
| G7 | 93.53 | 93.65 | +0.11 | 93.69 | 94.54 | +0.85 |
| G8 | 93.00 | 93.52 | +0.53 | 93.61 | 94.26 | +0.65 |
| G9 | 92.91 | 93.63 | **+0.72** | 93.82 | 94.75 | +0.93 |
| G10 | 93.32 | 93.78 | +0.46 | 93.63 | 94.46 | +0.82 |
| G13 | 93.14 | 93.59 | +0.45 | 93.77 | 94.72 | **+0.95** |
| G11 | 93.26 | 93.84 | +0.58 | 93.72 | 94.66 | +0.94 |
| G12 | 93.45 | 93.83 | +0.38 | 93.81 | 94.43 | +0.62 |

The gain is stable for fusion (+0.62 to +0.95) and positive throughout for LEAR (+0.11 to +0.72). The economical sets (G9, G11, G13) occupy the top of the fusion range, which supports the reading in 4.2: **a compact feature set leaves more room for the exogenous variables to contribute.** Reliability of the exogenous variables therefore matters more than how the price history is engineered.

---

## 4.4 The bottom line

**For the NP dataset, the best encoding of the price history is simply "yesterday, the day before, the same weekday last week, and the hour of day" — the four features of G1.** Every other transformation tested is a net loss:

| Transformation | Best fusion (price-only) | vs G1 |
|---|---|---|
| G2 add one more lag | **93.91** | **+0.02** |
| G9 differences | 93.82 | −0.07 |
| G12 long-horizon trend | 93.81 | −0.08 |
| G13 economical replacement | 93.77 | −0.12 |
| G11 absolute distance to extremes | 93.72 | −0.17 |
| G5 rolling monthly mean | 93.70 | −0.19 |
| G10 volatility | 93.63 | −0.26 |
| G8 full stack | 93.61 | −0.28 |

G2 (five features) ties G1 to within rounding; from there, every addition — completing the week (G3), rolling means (G4, G5), relative position (G6), weekly range (G7), the full stack (G8), and each of the five qualitatively different representations — reduces accuracy. The same ordering reappears with exogenous inputs.

This is not a claim that feature engineering *in general* is useless. It is narrower and more useful: **on this dataset the price history has a four-feature information ceiling, and no amount of re-expression raises it.** The headroom lies in the exogenous direction — load and wind are worth +0.65 to +0.95 points no matter which price features they are combined with.

## 5. Discussion

### 5.1 Why feature engineering barely moves the final number

The best result is obtained by fusing LEAR with a DNN, and both are strongly regularised. LEAR shrinks its coefficients and LASSO retains only the columns that carry independent signal; the DNN's early stopping and dropout do the same job differently. When the feature set contains redundant or noisy columns, regularisation discards them, so the added columns contribute little on net. This explains both the flat final accuracy and the monotone LEAR decline: extra columns increase the variance of the selection step without adding information.

The practical implication is that **for this dataset, feature-selection effort should go into acquiring exogenous variables, not into constructing more price transformations.**

### 5.2 Why trees prefer many lags but dislike rolling statistics

Tree models cannot extrapolate — they predict by averaging training samples that land in the same leaf. A lag value is an anchor: it maps directly onto price levels seen during training, so more lags give the trees more anchors. Rolling statistics, by contrast, are derived quantities that correlate with price only indirectly, and their relationship is less stable across time; they alter the partition of the feature space without improving it.

The data support this: every tree model reaches its maximum at G3 (eight lags, no statistics) in both regimes.

### 5.3 Why relative-position features help LEAR but not trees

`dev7` and `ratio_ma7` encode *where* the current price sits relative to its weekly mean. For a linear model this is a genuine interaction: the coefficient on `lag1` can effectively vary with the level of `ma7`. Trees can already represent that interaction through depth, so they gain nothing and lose a little to the added noise. G6 is the only set where LEAR rises rather than falls (93.40 % → 93.53 % in the price-only regime, against the preceding G5).

### 5.4 Why the fourth DNN configuration collapses under price-only inputs

`dnn_sf_da` (shuffle-train = 1, data augmentation on) scores 75–86 % price-only but 85–92 % with exogenous inputs. With four price columns and no demand drivers the network is under-determined by its inputs, and augmentation on a narrow input space injects variation the model cannot interpret. Once load and wind are available the same configuration becomes defensible. **This configuration should not be used on price-only inputs.**

### 5.5 Practical guidance

| Situation | Recommended feature set | Model |
|---|---|---|
| Load and wind forecasts available | **G3** (eight lags) or **G2** (five) | `fusion_lear_dnn`; `dnn_sf_sf` as the strongest single model |
| Price history only | **G2** (five) or **G1** (four) | `fusion_lear_dnn`; `lear` alone is within 0.3 points |
| Need a single interpretable model | G2 or G3 | `lear` |
| Must use tree models | **G3** — never add rolling statistics | `catboost` or `xgboost` |
| Avoid | G8 (full stack); `dnn_sf_da` on price-only | — |

The consistent message across all thirteen feature sets — including three that test qualitatively different representations (differences, volatility, and an economical six-column design) — is that **the marginal value of additional price-derived features is zero or negative, while the marginal value of exogenous drivers is reliably positive.**

---

## 6. Defects Encountered and Their Fixes

Eight defects surfaced while building this pipeline. Three of them — Defects 3, 6 and 8 — did not raise an exception; they produced plausible-looking output files with corrupted numbers. They are the most dangerous class of bug in an experimental pipeline, and the reason every run's output is now checked against the expected value range before being accepted.

### Defect 1 — All results reported 100 % accuracy

**Symptom.** Every model in the first multi-window run reported ACC = 100.00 %.

**Cause.** Predictions from different models were aligned by taking the *intersection* of the timestamps each model produced:

```python
valid_mask = ok if valid_mask is None else (valid_mask & ok)
```

One DNN configuration returned all-NaN predictions, so the intersection became the empty set. `metrics()` then computed WAPE over zero elements, and the `max(1e-9, ...)` guard turned 0/0 into 0, giving ACC = 100 − 0 = 100.

**Fix.** Each model is now scored on its own valid subset, and `metrics()` raises an exception on empty or zero-sum input instead of silently returning a perfect score.

```python
if len(y) == 0:
    raise ValueError("metrics: empty input (predictions and truth are not aligned)")
```

### Defect 2 — DNN training diverged to NaN

**Status.** Contained.

**Symptom.** Two of the four DNN configurations returned all-NaN predictions.

**Cause.** The library scales inputs as `(x − median) / MAD`. Columns whose median absolute deviation is near zero are amplified to infinity, and the resulting infinities propagate through training.

**Fix.** After scaling, non-finite values are detected and replaced with zero; a fallback to the stable `dnn_sf` configuration is retained for the rarer case where training still diverges.

### Defect 3 — LEAR crashed with a shape mismatch

**Symptom.** `ValueError: operands could not be broadcast together with shapes (300,) (297,)` raised inside the LARS solver.

**Cause.** Two separate issues, discovered in sequence:

1. The library's `recalibrate` modifies `Xtrain` in place. Passing the same array that `predict` later used left the two sides with inconsistent scaling.
2. `pos7` is a deterministic function of `lag1`, `min7` and `max7`, and G7/G8 contain all three. The design matrix therefore contains exactly collinear columns, and the LARS path terminates early, returning coefficient paths of unequal length.

**Fix.** (1) Pass independent copies to both calls. (2) Add Gaussian noise at 10⁻⁶ of each column's standard deviation. This breaks exact collinearity numerically while leaving the information content unchanged — preferable to deleting columns, which in an early attempt removed 176 of 343 columns and discarded real signal.

```python
rng = np.random.default_rng(42)
noise_scale = 1e-6 * (both.std(axis=0) + 1e-9)
both = both + rng.normal(0, 1, both.shape) * noise_scale
```

### Defect 4 — LassoLarsIC undefined when samples < features

**Symptom.** `ValueError: You are using LassoLarsIC in the case where the number of samples is smaller than the number of features`.

**Cause.** The earliest windows have 358 training days; the higher-dimensional sets expand to 368–560 columns. The AIC criterion the estimator uses to select the penalty is not defined in that regime.

**Fix.** Catch the error and fall back to a regularised estimator that is well defined for p > n (see Defect 6 for how the fallback itself had to be corrected).

### Defect 5 — `Load_delta` two orders of magnitude larger than the target

**Symptom.** No exception. Accuracy collapsed to 0 % in the affected sets, with MAE of 388 against a price range of 1.2–200.

**Cause.** The exogenous change feature was defined as an absolute difference:

```python
add(f"{c}_delta", d[c] - d[c].shift(24))   # load difference, ±16,721 MWh
```

Load is measured in MWh and ranges from about 26,000 to 70,000, so its first difference reaches ±16,721 — versus a price scale of roughly 1–200. After standardisation the load-difference columns dominated the leading principal components, and every model learned a relationship that reflects load units rather than price dynamics.

**Fix.** Express the change as a rate, with a floor on the denominator and a clip on the result:

```python
prev = d[c].shift(24).clip(lower=float(max(1.0, d[c].quantile(0.05))))
add(f"{c}_delta", ((d[c] - d[c].shift(24)) / prev).clip(-3.0, 3.0))
```

`Load_delta` now spans [−0.27, 0.54] and `Wind_delta` [−1.00, 3.00].

**Lesson.** Any feature whose scale differs from the target by more than an order of magnitude should be scaled or reformulated before it enters a model. Range checks on the assembled design matrix, not just on the raw inputs, would have caught this immediately.

### Defect 6 — Fallback estimators produced explosive predictions

**Symptom.** The fallback returned predictions in the range [−82,506, 210,239] against a historical range of [1.4, 109.5].

**Cause.** Two successive wrong fallbacks:

- **LassoCV** with 5-fold CV on 358 samples and 463 features selects a penalty far too small, because the validation folds are themselves high-dimensional.
- **PCA → Lasso** reduced the dimension, but a fixed penalty still overfitted.

Measured on G7 across all 21 windows:

| Fallback | Windows OK | Anomalies | Mean ACC |
|---|---|---|---|
| Linear regression | 21/21 | 0 | 87.56 % |
| PCA + Lasso | 21/21 | 0 | 92.84 % |
| PCA + Ridge | 21/21 | 0 | 93.49 % |
| **Ridge (α = 10)** | **21/21** | **0** | **93.75 %** |

**Fix.** Ridge on standardised inputs. Ridge is well-conditioned for p > n, has no selection step to destabilise, and is the strongest of the four candidates on this data.

### Defect 7 — Control-flow error after a skipped model

**Symptom.** `KeyError: 'lear'` — the DNN branch executed for the LEAR model key.

**Cause.** After the fallback path used `continue` to skip LEAR, execution still reached the `else:` branch that unpacks DNN hyperparameters, because the `try`/`except`/`else` nesting had been flattened by an earlier edit.

**Fix.** The LEAR branch was rewritten so that the native attempt, the fallback and the NaN check form one flat block, and the DNN branch begins in a clean `else:`.

### Defect 8 — `recalibrate` corrupted the input *before* failing

**Symptom.** No exception. The fallback reported success, but the stored LEAR predictions had MAE 2,591.

**Cause.** The library's `recalibrate` scales `Xtrain` **in place**. When it raised on the sample-count check, it had already transformed part of the array, leaving `Xtr_l` at standardised values in the range [−3, 3]. The Ridge fallback then applied `StandardScaler` to already-standardised data, collapsing the variance structure, and the fitted coefficients produced nonsensical outputs.

The crucial detail is ordering: the corruption happened **inside** the `try` block, so the `except` handler received a damaged array and had no way to know.

**Fix.** Snapshot the data before the call that may mutate it, and hand the fallback the snapshot:

```python
Xtr_safe = np.array(Xtr_l, dtype=float, copy=True)
Xte_safe = np.array(Xte_l, dtype=float, copy=True)
try:
    m.recalibrate(Xtrain=Xtr_l, Ytrain=Ytr)   # may mutate Xtr_l
    ...
except Exception:
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(Xtr_safe)       # pristine copy
```

**Lesson.** A library function that mutates its arguments can corrupt state even when it raises. Any fallback that reads the same buffer must take a copy beforehand, not afterwards.

### 6.1 Defects that produced no error message

| Defect | Effect on results | How it was detected |
|---|---|---|
| 1 | ACC = 100 % across the board | Implausible value |
| 3 | Alternate windows silently dropped from LEAR | Coverage count (41,496 vs 43,656) |
| 5 | ACC = 0 %, MAE 388 | Implausible value |
| 6 | Predictions off by three orders of magnitude | Range check against training data |
| 8 | Fallback reported success, MAE 2,591 | Manual inspection of stored metrics |

Three of the eight defects produced output files with no accompanying error. The range check added after Defect 6 is what finally made Defect 8 visible:

```python
_lo, _hi = Ytr.min(), Ytr.max()
if not np.isfinite(yp).all() or yp.min() < _lo - 3 * (_hi - _lo) or yp.max() > _hi + 3 * (_hi - _lo):
    raise ValueError(f"prediction out of range [{np.nanmin(yp):.1f}, {np.nanmax(yp):.1f}]")
```

**Recommendation for anyone running a similar study:** validate every intermediate artefact against the physical range of the target, and compare evaluation-point counts against the expected total. A model that returns a perfect score, or a count short by exactly one window, is a defect signal — not a result.

---

## 7. Caveats

**LEAR uses different estimators across cells.** LassoLarsIC is used wherever the sample-to-feature ratio permits it; where p > n it is undefined and a fallback is required. The mapping is:

| Regime | Sets | Columns | LEAR estimator |
|---|---|---|---|
| Price-only | G1–G7 | 96–288 | LassoLarsIC |
| Price-only | G8 | 368 | LassoLarsIC + 10⁻⁶ noise |
| +exog | G1 | 272 | LassoLarsIC |
| +exog | G2 | 296 | LassoLarsIC |
| +exog | G3, G6 | 368+ | LassoCV (standardised) |
| +exog | G4, G5, G7, G8 | — | Ridge (standardised) |

The fallback estimators are not directly comparable to LassoLarsIC; on identical data Ridge scored 93.75 % against LassoLarsIC's ≈94.0 %. LEAR values in the fallback cells should therefore be read as slightly pessimistic.

**Two cells evaluate over 41,496 hours rather than 43,656.** One window's LEAR fit failed the range check and was dropped; coverage is reported alongside every result.

**ACC is defined as 100 − WAPE**, where WAPE = Σ|error| / Σ|actual| × 100. It is an accuracy proxy, not a probability.

---

## 8. Reproducing

```bash
# price-only regime
python np_v4_experiment.py noexog B

# price + exogenous regime
python np_v4_experiment.py exog B
```

Each feature set is written to its own file as soon as it finishes, and completed sets are skipped on restart, so an interrupted run resumes where it stopped.

| File | Contents |
|---|---|
| `np_v4_results/incr_noexog_B_G{1..8}.json` | Per-set metrics, feature list, feature importances |
| `np_v4_results/incr_exog_B_G{1..8}.json` | As above, with exogenous inputs |

Each JSON record contains per-model ACC / MAE / RMSE / sMAPE / WAPE / R² / evaluation count, the resolved feature list, and LightGBM feature importances.