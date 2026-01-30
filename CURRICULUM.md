# Mastering Empirical Macroeconometrics

**Author**: Bijoy Ratan Ghosh (University of Virginia)
**Goal**: Deep understanding of modern empirical methods in macroeconomics
**Languages**: R (panel, LP, DiD, VAR), Python (Bayesian), MATLAB (DSGE)
**Format**: Interactive notebooks — theory + Monte Carlo + applied examples + diagnostics

---

## Curriculum Overview

| Phase | Topic | Modules | Language |
|-------|-------|---------|----------|
| 1 | Foundations | 1-2 | R |
| 2 | Dynamic Methods | 3-4 | R |
| 3 | Treatment Effects | 5-6 | R |
| 4 | Bayesian Econometrics | 7-9 | Python |
| 5 | Structural Macro | 10-11 | MATLAB/Python |
| 6 | Causal ML | 12-13 | Python |

---

## Phase 1: Foundations (R)

### Module 1: Panel Data Econometrics
**File**: `chapters/01_panel_econometrics.qmd`

- **From OLS to Panel FE**
  - What pooled OLS assumes (and when it fails)
  - The omitted variable bias argument for fixed effects
  - Within transformation: what FE literally does to your data
  - Between vs. within variation — which your coefficient captures
  - Monte Carlo: simulate panel with correlated unobservables, show OLS bias vs. FE consistency

- **Two-Way Fixed Effects**
  - Country FE: absorbs time-invariant characteristics
  - Time FE: absorbs common shocks
  - What's left: unit-specific deviations from unit mean and time mean
  - Diagnostics: F-test for FE significance, Hausman test (FE vs RE)

- **Interactions in Panels**
  - What an interaction term identifies (differential effect, not the level)
  - Standard errors: clustering, why IID is wrong for panels
  - Monte Carlo: simulate the true DGP and show that clustered SEs have correct coverage

- **Diagnostics and Threats**
  - Multicollinearity in interaction models
  - Influential observations (Cook's distance, DFBETAS)
  - Serial correlation in panel residuals
  - Cross-sectional dependence (Pesaran CD test)

### Module 2: Identification in Macro
**File**: `chapters/02_identification.qmd`

- **The Fundamental Problem**
  - Simultaneity: Y affects X, X affects Y
  - Selection: treated units differ from controls
  - Omitted variables: unobserved confounders
  - Why "controlling for" doesn't solve endogeneity

- **DID as an Identification Strategy**
  - Parallel trends assumption: what it requires, when it fails
  - Pre-trends testing: what it does and doesn't tell you (Roth 2022)
  - Natural experiments in macro

- **Instruments and GMM (Overview)**
  - The logic of IV: relevance, exclusion, monotonicity
  - Why finding instruments in macro is hard
  - Shift-share designs for macro

- **Wild Cluster Bootstrap**
  - Small-cluster inference problem
  - Cameron-Gelbach-Miller approach
  - When to use WCB vs. analytical corrections

---

## Phase 2: Dynamic Methods (R)

### Module 3: Local Projections
**File**: `chapters/03_local_projections.qmd`

- **LP vs. VAR: The Core Trade-off**
  - VAR: estimate a system, iterate forward for IRFs (parametric, efficient, fragile)
  - LP: directly regress y_{t+h} on shock at t (nonparametric in horizon, robust, noisy)
  - Mathematical connection: LP and VAR give same IRFs under correct specification
  - When they diverge: misspecification, nonlinearity, small samples
  - Monte Carlo: simulate a VAR DGP, estimate both, compare coverage

- **State-Dependent LP**
  - Interaction approach: LP with regime indicator × shock
  - Smooth transition (Auerbach-Gorodnichenko): F(z) weighting function
  - The horizon profile: why delay matters (not just magnitude)
  - Inference: Newey-West HAC SEs, why standard errors grow with h

- **LP-DID**
  - Combining LP dynamics with DID identification
  - Cross-sectional treatment vs. panel variation
  - Connection to event study literature

- **Diagnostics**
  - Pre-trend equivalents in LP (pre-shock coefficients)
  - Sensitivity to controls (lag structure, additional covariates)
  - Comparison: LP vs. VAR IRFs side by side

### Module 4: Vector Autoregressions
**File**: `chapters/04_var_svar.qmd`

- **Reduced-Form VAR**
  - VAR(p) as a system of equations
  - Estimation: OLS equation-by-equation
  - Lag selection: AIC, BIC, HQ
  - Stability: eigenvalues inside unit circle

- **Structural Identification: Why Ordering Matters**
  - The fundamental problem: reduced-form residuals ≠ structural shocks
  - Cholesky decomposition: what it assumes
  - What changes if you reorder
  - Monte Carlo: simulate structural model, show that wrong ordering gives wrong IRFs

- **IRFs, FEVDs, and Granger Causality**
  - IRF: response of variable j to a 1-SD shock in variable k
  - Bootstrap confidence intervals
  - FEVD: fraction of forecast error variance explained
  - Granger causality: predictive precedence, NOT true causality

- **Panel VAR Approaches**
  - Split-sample: estimate separate VARs by group, compare IRFs
  - Mean-group: estimate unit-level VARs, average IRFs
  - Pooled panel VAR: common dynamics with unit FE

- **Beyond Cholesky**
  - Sign restrictions: impose direction (not magnitude) of responses
  - Narrative restrictions: use known historical events
  - Long-run restrictions (Blanchard-Quah)

---

## Phase 3: Modern Treatment Effects (R)

### Module 5: Staggered Difference-in-Differences
**File**: `chapters/05_staggered_did.qmd`

- **The TWFE Problem**
  - Standard TWFE with staggered treatment: what goes wrong
  - Negative weights: early-treated units act as "controls" for late-treated
  - Goodman-Bacon decomposition: decompose TWFE into 2×2 DID comparisons
  - Monte Carlo: simulate heterogeneous treatment effects, show TWFE bias

- **Modern DiD Estimators**
  - Sun-Abraham (2021): interaction-weighted estimator, cohort-specific ATTs
  - Callaway-Sant'Anna (2021): group-time treatment effects
  - de Chaisemartin-D'Haultfoeuille (2020): fuzzy designs
  - When each is appropriate, what assumptions they require

- **Event Study Design**
  - Pre-trend testing (and its limitations — Roth 2022)
  - Heterogeneous event studies
  - Intensity-weighted design: continuous treatment

### Module 6: Synthetic Control Methods
**File**: `chapters/06_synthetic_control.qmd`

- **The SCM Framework**
  - Abadie-Diamond-Hainmueller: constructing the counterfactual
  - Choosing predictors and pre-treatment fit
  - Donor pool selection
  - Convex weight restriction vs. augmented SCM

- **Inference**
  - Placebo (permutation) tests
  - RMSPE ratios
  - Conformal inference for SC (Chernozhukov et al.)
  - Power: when can SC detect an effect?

- **Practical Issues**
  - Pre-treatment period length
  - Handling structural breaks
  - Robustness: leave-one-out, alternative donor pools

---

## Phase 4: Bayesian Econometrics (Python)

### Module 7: Bayesian Foundations
**File**: `chapters/07_bayesian_foundations.qmd`

- **Bayes' Theorem in Practice**
  - Prior × Likelihood = Posterior (up to normalizing constant)
  - Conjugate priors: Normal-Normal, Normal-InverseGamma
  - Informative vs. weakly informative vs. flat priors
  - Hands-on: Bayesian linear regression on simulated data

- **MCMC Methods**
  - Why we need sampling (posterior isn't analytic in general)
  - Metropolis-Hastings: the random walk proposal
  - Gibbs sampling: exploit conditional conjugacy
  - Hamiltonian Monte Carlo (HMC): the modern workhorse
  - Implementation: code each from scratch, then use PyMC/NumPyro

- **Convergence Diagnostics**
  - Trace plots: visual inspection
  - R-hat (Gelman-Rubin): target < 1.01
  - Effective sample size (ESS)
  - Divergent transitions (HMC-specific)

- **Model Comparison**
  - Marginal likelihood
  - Bayes factors
  - Information criteria: WAIC, LOO-CV

### Module 8: Bayesian VARs
**File**: `chapters/08_bayesian_var.qmd`

- **The Minnesota Prior**
  - Motivation: VAR has too many parameters for macro samples
  - The Litterman (1986) idea: shrink toward random walk
  - Hyperparameters: overall tightness, cross-variable shrinkage, lag decay
  - Implementation: Normal-Inverse-Wishart posterior, Gibbs sampler

- **Sign Restriction Identification**
  - Motivation: Cholesky is too restrictive
  - Algorithm: draw rotation matrices, keep those satisfying sign restrictions
  - Set-identification: multiple admissible models → credible sets

- **Narrative Restrictions**
  - Antolín-Díaz and Rubio-Ramírez (2018): combine sign + narrative
  - Sharpens identification vs. pure sign restrictions

### Module 9: Bayesian Panel Methods
**File**: `chapters/09_bayesian_panel.qmd`

- **Hierarchical/Multilevel Models**
  - Pooled → Fixed Effects → Random Effects → Hierarchical
  - Partial pooling: the Bayesian middle ground
  - Shrinkage toward group mean

- **Random Coefficients Panel**
  - Each unit has its own β drawn from a common distribution
  - Posterior distribution of unit-specific effects

- **Bayesian Panel VAR**
  - Hierarchical priors on VAR parameters
  - Unit-level VARs with cross-unit shrinkage

---

## Phase 5: Structural Macro (MATLAB + Python)

### Module 10: DSGE Foundations
**File**: `chapters/10_dsge_foundations.qmd`

- **The 3-Equation NK Model**
  - IS curve, Phillips curve, Taylor rule — from micro-foundations to linear system
  - Log-linearization: from nonlinear to linear rational expectations
  - Blanchard-Kahn conditions: existence and uniqueness of solution
  - Active vs. passive monetary policy (Leeper 1991)

- **Solution Methods**
  - Method of undetermined coefficients (simple models)
  - QZ (generalized Schur) decomposition (general method)
  - Perturbation: first-order, second-order, third-order
  - Dynare basics: .mod file syntax, steady state, stochastic simulation

- **Adding Financial Frictions**
  - Bernanke-Gertler-Gilchrist: financial accelerator
  - Bank capital constraints → amplification of shocks

### Module 11: DSGE Estimation
**File**: `chapters/11_dsge_estimation.qmd`

- **State-Space Form and the Kalman Filter**
  - DSGE solution → state-space representation
  - Kalman filter: optimal linear filter for latent states
  - Likelihood function: prediction error decomposition

- **Bayesian DSGE Estimation**
  - Prior × likelihood via MCMC
  - Choosing priors for structural parameters
  - Convergence: mode check plots, multiple chains
  - Dynare's Bayesian estimation tools

---

## Phase 6: Causal Machine Learning (Python)

### Module 12: Regularization for Macro
**File**: `chapters/12_regularization.qmd`

- **LASSO, Ridge, Elastic Net**
  - The bias-variance trade-off
  - L1 (LASSO): sparsity, variable selection
  - L2 (Ridge): shrinkage without sparsity
  - Elastic Net: the hybrid

- **Double/Debiased LASSO**
  - Chernozhukov et al. (2018): using LASSO for causal inference
  - Why naive LASSO gives biased treatment effects

- **Tree-Based Methods**
  - Random Forest: bagging + random feature subsets
  - Variable importance: permutation, impurity-based, SHAP
  - Decision trees for interpretability

### Module 13: Causal Machine Learning
**File**: `chapters/13_causal_ml.qmd`

- **Causal Forests (Athey-Imbens)**
  - Heterogeneous treatment effects
  - Honest estimation: sample splitting

- **Double ML for Panel Data**
  - Extending debiased LASSO to panel settings
  - High-dimensional controls with causal target parameter

---

## Each Module Includes

1. **Theory Block**: Mathematical foundations, key theorems, identification assumptions
2. **Monte Carlo Lab**: Simulate known DGP → estimate → verify properties
3. **Applied Example**: Apply methods to public macroeconomic data
4. **Diagnostics Checklist**: What to check, what can go wrong, red flags
5. **FAQ**: Common conceptual questions and answers

---

## Technical Requirements

- **R** (≥4.0) with packages: `fixest`, `did`, `synthdid`, `vars`, `lmtest`, `sandwich`
- **Python** (≥3.9) with packages: `numpy`, `pandas`, `statsmodels`, `pymc`, `arviz`, `econml`
- **MATLAB** (optional) for DSGE modules, or use Dynare with Octave
- **Quarto** (≥1.3) for rendering

---

*See [SOURCES.md](SOURCES.md) for references and acknowledgments.*
