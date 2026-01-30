# Causal Machine Learning: Expert Reference

You are an expert in causal machine learning methods for economics. When asked to implement, interpret, or advise on heterogeneous treatment effects, causal forests, or double ML, use this reference. Write code in both R (using `grf`) and Python (using `econml`, `doubleml`) as appropriate.

---

## 1. Core Framework: Potential Outcomes

### Setup

For each unit $i$:
- **Treatment**: $W_i \in \{0, 1\}$ (or continuous)
- **Potential outcomes**: $Y_i(0), Y_i(1)$
- **Observed outcome**: $Y_i = W_i \cdot Y_i(1) + (1 - W_i) \cdot Y_i(0)$
- **Covariates**: $X_i$ (pre-treatment characteristics)

### Treatment Effects

| Estimand | Definition | Interpretation |
|----------|------------|----------------|
| **ITE** | $\tau_i = Y_i(1) - Y_i(0)$ | Individual treatment effect (unobservable) |
| **CATE** | $\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$ | Conditional average treatment effect |
| **ATE** | $\mathbb{E}[\tau_i]$ | Average treatment effect |
| **ATT** | $\mathbb{E}[\tau_i | W_i = 1]$ | Average effect on treated |

### Identification Assumptions

**Unconfoundedness** (selection on observables):
$$
(Y_i(0), Y_i(1)) \perp W_i | X_i
$$

**Overlap** (positivity):
$$
0 < P(W_i = 1 | X_i = x) < 1 \quad \forall x
$$

---

## 2. Causal Forests (Wager & Athey 2018)

### The Idea

Adapt random forests to estimate $\tau(x)$ instead of $\mathbb{E}[Y|X]$.

**Key innovations**:
1. **Honest splitting**: Separate samples for tree structure vs. leaf estimation
2. **Heterogeneity-based splits**: Maximize treatment effect variation, not prediction accuracy
3. **Local moment conditions**: Within-leaf treatment effect estimation

### Algorithm

1. For each tree $b = 1, \ldots, B$:
   - Subsample data into tree-building ($\mathcal{I}_1$) and estimation ($\mathcal{I}_2$) sets
   - Build tree on $\mathcal{I}_1$ maximizing treatment heterogeneity
   - Estimate leaf effects using $\mathcal{I}_2$ only
2. Aggregate: $\hat{\tau}(x) = \frac{1}{B} \sum_b \hat{\tau}_b(x)$

### R Implementation with `grf`

```r
library(grf)

# Simulate data with heterogeneous effects
n <- 2000
p <- 10
X <- matrix(rnorm(n * p), n, p)
W <- rbinom(n, 1, 0.5)  # randomized treatment
tau_true <- X[, 1] + 0.5 * X[, 2]  # true CATE depends on X1, X2
Y <- X[, 1] + tau_true * W + rnorm(n)

# Fit causal forest
cf <- causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = 2000,
  honesty = TRUE,              # honest splitting (default)
  tune.parameters = "all"      # automatic hyperparameter tuning
)

# Point predictions
tau_hat <- predict(cf)$predictions

# Predictions with variance estimates
tau_ci <- predict(cf, estimate.variance = TRUE)
lower <- tau_ci$predictions - 1.96 * sqrt(tau_ci$variance.estimates)
upper <- tau_ci$predictions + 1.96 * sqrt(tau_ci$variance.estimates)

# Average treatment effect (ATE)
ate <- average_treatment_effect(cf, target.sample = "all")
cat("ATE:", ate[1], "SE:", ate[2], "\n")

# ATT (effect on treated)
att <- average_treatment_effect(cf, target.sample = "treated")

# Variable importance
vi <- variable_importance(cf)
print(vi)

# Calibration test (is there heterogeneity?)
test_calibration(cf)

# Best linear projection (which covariates explain heterogeneity?)
blp <- best_linear_projection(cf, X[, 1:3])
print(blp)

# Rank average treatment effect (AUTOC)
rate <- rank_average_treatment_effect(cf, X[, 1])
print(rate)
```

### Key `grf` Functions

| Function | Purpose |
|----------|---------|
| `causal_forest()` | Fit causal forest |
| `predict()` | CATE predictions (with optional variance) |
| `average_treatment_effect()` | ATE or ATT with SE |
| `variable_importance()` | Which covariates drive heterogeneity |
| `best_linear_projection()` | Linear approximation of CATE |
| `test_calibration()` | Test for treatment effect heterogeneity |
| `rank_average_treatment_effect()` | AUTOC for targeting evaluation |

### With Pre-Estimated Nuisance Functions

For observational data, pre-fit propensity and outcome models:

```r
# Pre-fit nuisance models
W.hat <- predict(regression_forest(X, W))$predictions  # propensity
Y.hat <- predict(regression_forest(X, Y))$predictions  # outcome

# Causal forest with pre-estimated nuisance
cf_obs <- causal_forest(X, Y, W, W.hat = W.hat, Y.hat = Y.hat)
```

---

## 3. Double/Debiased Machine Learning

### The Problem

Using ML for nuisance parameters introduces regularization bias that invalidates standard inference.

### The Solution (Chernozhukov et al. 2018)

**Cross-fitting** + **Neyman orthogonality**

### Partially Linear Model

$$
Y = \theta D + g(X) + U, \quad \mathbb{E}[U|X,D] = 0
$$
$$
D = m(X) + V, \quad \mathbb{E}[V|X] = 0
$$

**Orthogonal moment**:
$$
\psi(W; \theta, \eta) = (Y - g(X) - \theta D)(D - m(X))
$$

### Algorithm

1. Split data into $K$ folds
2. For each fold $k$:
   - Train $\hat{g}_{-k}(X)$ and $\hat{m}_{-k}(X)$ on other folds
   - Compute residuals on fold $k$:
     - $\tilde{Y}_i = Y_i - \hat{g}_{-k}(X_i)$
     - $\tilde{D}_i = D_i - \hat{m}_{-k}(X_i)$
3. Estimate: $\hat{\theta} = \frac{\sum_i \tilde{D}_i \tilde{Y}_i}{\sum_i \tilde{D}_i^2}$
4. Standard error: $\hat{SE} = \sqrt{\frac{\sum_i (\tilde{Y}_i - \hat{\theta}\tilde{D}_i)^2 \tilde{D}_i^2}{(\sum_i \tilde{D}_i^2)^2}}$

### Python Implementation with `doubleml`

```python
import numpy as np
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Simulate data
np.random.seed(42)
n = 2000
p = 10
X = np.random.randn(n, p)
theta = 0.5  # true treatment effect
D = X[:, 0] + np.random.randn(n)  # treatment depends on X1
Y = theta * D + X[:, 0] + X[:, 1] + np.random.randn(n)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
df['Y'] = Y
df['D'] = D

# Create DoubleML data object
dml_data = DoubleMLData(
    df,
    y_col='Y',
    d_cols='D',
    x_cols=[f'X{i}' for i in range(p)]
)

# Specify ML learners
ml_l = RandomForestRegressor(n_estimators=500, max_depth=5)  # outcome model
ml_m = RandomForestRegressor(n_estimators=500, max_depth=5)  # propensity/treatment model

# Fit Double ML (Partially Linear Regression)
dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=5)
dml_plr.fit()
print(dml_plr.summary)

# Confidence interval
print(dml_plr.confint())

# Bootstrap inference
dml_plr.bootstrap(n_rep=1000)
print(dml_plr.confint(joint=True))
```

### DoubleML Models

| Model | Class | Use Case |
|-------|-------|----------|
| Partially Linear Regression | `DoubleMLPLR` | Basic treatment effect |
| Partially Linear IV | `DoubleMLPLIV` | Endogenous treatment |
| Interactive Regression | `DoubleMLIRM` | Binary treatment, CATE |
| Interactive IV | `DoubleMLIIVM` | Binary + endogenous |

---

## 4. EconML: Microsoft's Causal ML Toolkit

### Overview

EconML provides a unified API for heterogeneous treatment effect estimation.

### Key Estimators

| Estimator | Description |
|-----------|-------------|
| `LinearDML` | DML with linear final stage |
| `SparseLinearDML` | High-dimensional features via debiased LASSO |
| `CausalForestDML` | Causal forest with DML |
| `ForestDRLearner` | Doubly robust forest learner |
| `DynamicDML` | Panel data with dynamic effects |
| `OrthoIV` | Orthogonal IV learner |

### Python Implementation

```python
from econml.dml import LinearDML, CausalForestDML
from econml.dr import ForestDRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

# Simulate data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 5)  # features for heterogeneity
W = np.random.randn(n, 3)  # confounders
T = (W[:, 0] > 0).astype(int)  # binary treatment
tau_true = X[:, 0] + 0.5 * X[:, 1]  # true CATE
Y = tau_true * T + W[:, 0] + W[:, 1] + np.random.randn(n)

# --- LinearDML ---
est_linear = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100),
    model_t=RandomForestClassifier(n_estimators=100),
    cv=5
)
est_linear.fit(Y, T, X=X, W=W)

# Point estimates
tau_hat = est_linear.effect(X)

# Confidence intervals
tau_lower, tau_upper = est_linear.effect_interval(X, alpha=0.05)

# Summary with inference
print(est_linear.summary())

# --- CausalForestDML ---
est_cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100),
    model_t=RandomForestClassifier(n_estimators=100),
    n_estimators=2000,
    cv=5
)
est_cf.fit(Y, T, X=X, W=W)
tau_cf = est_cf.effect(X)

# Feature importance
print("Feature importances:", est_cf.feature_importances_)

# --- ForestDRLearner (Doubly Robust) ---
est_dr = ForestDRLearner(
    model_regression=RandomForestRegressor(n_estimators=100),
    model_propensity=RandomForestClassifier(n_estimators=100),
    n_estimators=2000
)
est_dr.fit(Y, T, X=X, W=W)
```

### Unified API Pattern

All EconML estimators follow:
```python
est.fit(Y, T, X=X, W=W)           # fit model
est.effect(X_test)                 # point estimates
est.effect_interval(X_test)        # confidence intervals
est.summary()                      # inference summary
```

---

## 5. Meta-Learners

### T-Learner

Train separate models for treatment and control:
$$
\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

```python
from sklearn.ensemble import RandomForestRegressor

# Fit separate models
model_0 = RandomForestRegressor().fit(X[T == 0], Y[T == 0])
model_1 = RandomForestRegressor().fit(X[T == 1], Y[T == 1])

# CATE estimate
tau_t = model_1.predict(X) - model_0.predict(X)
```

### S-Learner

Single model with treatment as feature:
$$
\hat{\mu}(x, t) \rightarrow \hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)
$$

```python
# Augment features
X_aug = np.column_stack([X, T])

# Single model
model_s = RandomForestRegressor().fit(X_aug, Y)

# CATE estimate
tau_s = model_s.predict(np.column_stack([X, np.ones(n)])) - \
        model_s.predict(np.column_stack([X, np.zeros(n)]))
```

### X-Learner (Künzel et al. 2019)

1. Fit $\hat{\mu}_0, \hat{\mu}_1$ (like T-learner)
2. Impute treatment effects:
   - Treated: $\tilde{D}_1 = Y_1 - \hat{\mu}_0(X_1)$
   - Control: $\tilde{D}_0 = \hat{\mu}_1(X_0) - Y_0$
3. Fit models for imputed effects: $\hat{\tau}_0(x), \hat{\tau}_1(x)$
4. Combine: $\hat{\tau}(x) = e(x) \hat{\tau}_0(x) + (1-e(x)) \hat{\tau}_1(x)$

```python
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

x_learner = XLearner(
    models=RandomForestRegressor(),
    propensity_model=RandomForestClassifier()
)
x_learner.fit(Y, T, X=X)
tau_x = x_learner.effect(X)
```

### Comparison

| Learner | Pros | Cons |
|---------|------|------|
| **T-Learner** | Simple, no propensity needed | High variance, no regularization |
| **S-Learner** | Regularization, simple | Treatment effect can be shrunk to 0 |
| **X-Learner** | Good with imbalanced data | More complex, needs propensity |

---

## 6. R Implementation: Manual Double ML

```r
# Manual Double ML for ATE
double_ml_ate <- function(Y, W, X, K = 5) {
  n <- length(Y)
  folds <- sample(rep(1:K, length.out = n))

  psi <- numeric(n)

  for (k in 1:K) {
    train_idx <- folds != k
    test_idx <- folds == k

    # Train outcome models
    rf_1 <- grf::regression_forest(X[train_idx & W == 1, , drop = FALSE],
                                    Y[train_idx & W == 1])
    rf_0 <- grf::regression_forest(X[train_idx & W == 0, , drop = FALSE],
                                    Y[train_idx & W == 0])

    # Train propensity model
    rf_e <- grf::regression_forest(X[train_idx, , drop = FALSE],
                                    W[train_idx])

    # Predict on test fold
    mu1_hat <- predict(rf_1, X[test_idx, , drop = FALSE])$predictions
    mu0_hat <- predict(rf_0, X[test_idx, , drop = FALSE])$predictions
    e_hat <- predict(rf_e, X[test_idx, , drop = FALSE])$predictions
    e_hat <- pmax(pmin(e_hat, 0.99), 0.01)  # clip propensity

    # AIPW pseudo-outcome
    Y_test <- Y[test_idx]
    W_test <- W[test_idx]

    psi[test_idx] <- W_test * (Y_test - mu1_hat) / e_hat -
                     (1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat) +
                     mu1_hat - mu0_hat
  }

  # ATE and SE
  tau_hat <- mean(psi)
  se_hat <- sd(psi) / sqrt(n)

  list(
    estimate = tau_hat,
    se = se_hat,
    ci_lower = tau_hat - 1.96 * se_hat,
    ci_upper = tau_hat + 1.96 * se_hat
  )
}
```

---

## 7. Group Average Treatment Effects (GATES)

Evaluate heterogeneity by grouping units by predicted CATE:

```r
# GATES analysis
compute_gates <- function(cf, Y, W, n_groups = 5) {
  tau_hat <- predict(cf)$predictions

  # Create groups
  groups <- cut(tau_hat,
                breaks = quantile(tau_hat, seq(0, 1, length.out = n_groups + 1)),
                labels = paste0("Q", 1:n_groups),
                include.lowest = TRUE)

  # Estimate ATE within each group
  gates <- data.frame(
    group = levels(groups),
    ate = NA, se = NA
  )

  for (g in 1:n_groups) {
    idx <- groups == paste0("Q", g)
    if (sum(idx & W == 1) > 10 && sum(idx & W == 0) > 10) {
      # Simple difference in means
      gates$ate[g] <- mean(Y[idx & W == 1]) - mean(Y[idx & W == 0])
      n1 <- sum(idx & W == 1)
      n0 <- sum(idx & W == 0)
      gates$se[g] <- sqrt(var(Y[idx & W == 1])/n1 + var(Y[idx & W == 0])/n0)
    }
  }

  gates
}
```

---

## 8. Policy Learning: Optimal Treatment Assignment

**Goal**: Learn treatment rule $\pi(x) \in \{0, 1\}$ maximizing welfare.

```r
# Simple policy learning with grf
library(policytree)

# Fit causal forest
cf <- causal_forest(X, Y, W)

# Fit optimal policy tree
dr_scores <- double_robust_scores(cf)
policy <- policy_tree(X, dr_scores, depth = 2)

# Predict optimal treatment
optimal_treatment <- predict(policy, X)

# Policy value (expected outcome under learned policy)
policy_value <- mean(ifelse(optimal_treatment == 1,
                            predict(cf)$predictions, 0))
```

---

## 9. Diagnostics and Validation

### Calibration Test

Is there actually heterogeneity?

```r
# grf's calibration test
test_calibration(cf)

# Interpretation:
# - "mean.forest.prediction": avg predicted effect (should ≈ ATE)
# - "differential.forest.prediction": heterogeneity signal (should be significant)
```

### Overlap Diagnostics

```r
# Check propensity score distribution
e_hat <- cf$W.hat  # propensity scores from causal forest

# Histogram
hist(e_hat, breaks = 50, main = "Propensity Score Distribution")
abline(v = c(0.1, 0.9), col = "red", lty = 2)

# Overlap violations
cat("Extreme propensity (<0.1 or >0.9):",
    mean(e_hat < 0.1 | e_hat > 0.9) * 100, "%\n")
```

### Out-of-Sample Validation

```r
# Cross-validation for CATE prediction
library(caret)

# Can't directly validate CATE (counterfactual), but can:
# 1. Check if high-CATE groups have higher treatment effects
# 2. Use policy evaluation metrics (AUTOC)

# AUTOC: Area under the TOC curve
rate <- rank_average_treatment_effect(cf, X[, 1])
print(rate)  # 95% CI should exclude 0 for meaningful heterogeneity
```

---

## 10. Application to Macro: Heterogeneous Policy Effects

### Cross-Country Monetary Policy

```r
# Example: Do monetary policy effects vary by bank holdings?
library(grf)

# Data structure (hypothetical)
# X: country characteristics (bank holdings, institutions, debt/GDP, ...)
# W: monetary tightening indicator
# Y: inflation response

cf_policy <- causal_forest(
  X = cbind(bank_holdings, cbi_index, debt_gdp, trade_openness),
  Y = delta_inflation,
  W = tightening_dummy
)

# Which characteristics predict heterogeneity?
vi <- variable_importance(cf_policy)
names(vi) <- c("bank_holdings", "cbi_index", "debt_gdp", "trade_openness")
sort(vi, decreasing = TRUE)

# Best linear projection
blp <- best_linear_projection(cf_policy,
                               cbind(bank_holdings, cbi_index, debt_gdp))
print(blp)

# GATES by bank holdings quartile
gates <- compute_gates(cf_policy, delta_inflation, tightening_dummy)
```

### Panel Data with DynamicDML

```python
from econml.panel.dml import DynamicDML

# For panel data with multiple time periods
# Estimates cumulative and period-specific effects
dml_panel = DynamicDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    n_periods=4
)
dml_panel.fit(Y, T, X=X, W=W, groups=country_id)
```

---

## 11. Key Practical Considerations

### Sample Size Requirements

| Method | Minimum N | Recommended N |
|--------|-----------|---------------|
| ATE (Double ML) | 200+ | 500+ |
| CATE (Causal Forest) | 500+ | 2000+ |
| GATES | 1000+ | 3000+ |
| Variable Importance | 2000+ | 5000+ |

### Common Pitfalls

1. **Ignoring overlap**: Check propensity scores; trim or weight
2. **Overfitting CATE**: Use honest forests, cross-validation
3. **Interpreting noise as heterogeneity**: Run calibration tests
4. **Wrong identification**: Causal ML doesn't solve confounding
5. **Small subgroups**: GATES unreliable with few observations per group

### When to Use What

| Situation | Recommended Method |
|-----------|-------------------|
| ATE with high-dimensional controls | Double ML |
| Heterogeneous effects, RCT | Causal Forest |
| Heterogeneous effects, observational | CausalForestDML (EconML) |
| Very imbalanced treatment | X-Learner |
| Panel data | DynamicDML |
| Optimal targeting | Policy Learning |

---

## 12. Key References

### Foundational
- **Athey & Imbens (2016)** "Recursive Partitioning for Heterogeneous Causal Effects" PNAS
- **Wager & Athey (2018)** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" JASA
- **Chernozhukov et al. (2018)** "Double/Debiased Machine Learning for Treatment and Structural Parameters" Econometrics Journal

### Extensions
- **Künzel et al. (2019)** "Metalearners for Estimating Heterogeneous Treatment Effects" PNAS — X-Learner
- **Athey & Wager (2021)** "Policy Learning with Observational Data" Econometrica
- **Kennedy (2022)** "Towards Optimal Doubly Robust Estimation" — DR-Learner improvements

### Textbooks & Courses
- **Causal ML Book**: https://causalml-book.org/ — Comprehensive treatment
- **ML for Economists** (Caspi): https://github.com/ml4econ/lecture-notes-2025
- **The Effect** (Huntington-Klein): Causal inference foundations

### Software
- **grf** (R): https://grf-labs.github.io/grf/ — Generalized Random Forests
- **EconML** (Python): https://github.com/py-why/EconML — Microsoft's causal ML
- **DoubleML** (Python/R): https://github.com/DoubleML/doubleml-for-py
- **policytree** (R): Policy learning with tree-based methods
