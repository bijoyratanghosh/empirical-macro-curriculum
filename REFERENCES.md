# External Resources and References

This document tracks external resources used to develop the curriculum.

---

## Textbooks and Core References

### Panel Econometrics & Causal Inference
- Wooldridge (2010) *Econometric Analysis of Cross Section and Panel Data*
- Cunningham (2021) *Causal Inference: The Mixtape*
- Angrist & Pischke (2009) *Mostly Harmless Econometrics*

### Time Series and VARs
- Lutkepohl (2005) *New Introduction to Multiple Time Series Analysis*
- Kilian & Lutkepohl (2017) *Structural Vector Autoregressive Analysis*
- Hamilton (1994) *Time Series Analysis*

### Bayesian Econometrics
- Koop (2003) *Bayesian Econometrics*
- Greenberg (2008) *Introduction to Bayesian Econometrics*
- Blake & Mumtaz (2012) *Applied Bayesian Econometrics for Central Bankers*
- **Herbst & Schorfheide (2016)** *Bayesian Estimation of DSGE Models*

### DSGE Models
- Gali (2015) *Monetary Policy, Inflation, and the Business Cycle*
- Woodford (2003) *Interest and Prices*
- DeJong & Dave (2011) *Structural Macroeconometrics*

---

## Online Course Materials

### Vladislav Morozov (GitHub)
- **econometrics-heterogeneity**: https://github.com/vladislav-morozov/econometrics-heterogeneity
  - Causal inference with unobserved heterogeneity (Master's/PhD level)
  - Covers: heterogeneous treatment effects, unobserved confounding

- **econometrics-2**: https://github.com/vladislav-morozov/econometrics-2
  - Advanced Econometrics course materials
  - Full lectures, codes, problem sets in TeX

- **simulations-course**: https://github.com/vladislav-morozov/simulations-course
  - Monte Carlo simulations for data science
  - Practical computational methods

### Herbst & Schorfheide DSGE Companion
**URL**: https://web.sas.upenn.edu/schorf/companion-web-site-bayesian-estimation-of-dsge-models/

| Resource | Language | Description |
|----------|----------|-------------|
| DSGE Estimation | MATLAB | Random walk Metropolis-Hastings algorithm |
| SMC | MATLAB | Sequential Monte Carlo for state-space models |
| DSGE SMC | MATLAB | SMC for small-scale DSGE model |
| Nonlinear Filtering | MATLAB | Bootstrap + conditionally optimal particle filters |
| Alternative Code | Python/FORTRAN | http://edherbst.net/bayesian-book |
| Julia Implementation | Julia | FRBNY (Del Negro, Cai, et al.) on GitHub |
| Lecture Slides | PDF | 10 chapters in 3 parts |
| Datasets | ZIP | Empirical illustration data |

---

## Software Packages by Module

### Module 1-2: Panel Econometrics & Identification
| Package | Language | Purpose |
|---------|----------|---------|
| `fixest` | R | Fast fixed effects, multi-way clustering |
| `did` | R | Callaway-Sant'Anna staggered DiD |
| `fwildclusterboot` | R | Wild cluster bootstrap |
| `HonestDiD` | R | Sensitivity analysis for DiD |

### Module 3: Local Projections
| Package | Language | Purpose |
|---------|----------|---------|
| `lpirfs` | R | Local projections IRFs |
| `fixest` | R | Manual LP with panel() |
| `lpdid` | Stata | LP-DiD implementation |

**LP-DiD (Girardi)**: https://github.com/danielegirardi/lpdid
- Implements Dube, Girardi, Jordà & Taylor (2024)
- Sample restriction approach to avoid TWFE bias
- Handles: absorbing, non-absorbing, shock treatments

### Module 4: VAR/SVAR
| Package | Language | Purpose |
|---------|----------|---------|
| `vars` | R | VAR estimation, IRFs, FEVD |
| `svars` | R | Structural VAR identification |
| `VARsignR` | R | Sign restrictions |

### Module 5-6: Treatment Effects
| Package | Language | Purpose |
|---------|----------|---------|
| `augsynth` | R | Augmented synthetic control |
| `synthdid` | R | Synthetic difference-in-differences |
| `Synth` | R | Original synthetic control |

### Module 7-9: Bayesian Econometrics
| Package | Language | Purpose |
|---------|----------|---------|
| `BVAR` | R | Minnesota prior with GLP optimization |
| `bvartools` | R | Flexible BVAR toolkit |
| `bvarsv` | R | TVP-VAR with stochastic volatility |
| `stochvol` | R | Univariate stochastic volatility |

### Module 10-11: DSGE
| Package | Language | Purpose |
|---------|----------|---------|
| Dynare | MATLAB/Octave | Standard DSGE solver/estimator |
| `gEcon` | R | Symbolic DSGE derivation |
| DSGE.jl | Julia | FRBNY DSGE implementation |
| IRIS Toolbox | MATLAB | Alternative DSGE toolkit |

### Module 12-13: Causal ML
| Package | Language | Purpose |
|---------|----------|---------|
| `glmnet` | R | LASSO/ridge/elastic net |
| `grf` | R | Generalized random forests |
| `DoubleML` | R/Python | Double machine learning |
| `causalForest` | R | Causal forests |

---

## Key Methodological Papers by Topic

### Difference-in-Differences (Modern)
- Callaway & Sant'Anna (2021) JoE — Staggered DiD
- Sun & Abraham (2021) JoE — Interaction-weighted estimator
- Goodman-Bacon (2021) JoE — TWFE decomposition
- de Chaisemartin & D'Haultfoeuille (2020) AER — DID_M
- Borusyak, Jaravel & Spiess (2024) REStat — Imputation approach
- Roth (2022) REStat — Pre-trends testing power

### Local Projections
- Jordà (2005) AER — Original LP paper
- Jordà & Taylor (2025) JEL — Comprehensive survey
- Plagborg-Møller & Wolf (2021) Econometrica — LP-VAR equivalence
- Dube, Girardi, Jordà & Taylor (2024) JAE — LP-DiD

### Bayesian VARs
- Litterman (1986) JBES — Minnesota prior
- Giannone, Lenza & Primiceri (2015) REStat — GLP optimization
- Primiceri (2005) RES — TVP-VAR-SV

### DSGE Estimation
- Smets & Wouters (2007) AER — Medium-scale estimated NK
- An & Schorfheide (2007) Econometric Reviews — Bayesian DSGE
- Herbst & Schorfheide (2016) — Modern textbook

---

## Data Sources

### Macroeconomic Data
- FRED (Federal Reserve Economic Data)
- IMF WEO (World Economic Outlook)
- World Bank WDI (World Development Indicators)
- BIS Statistics (Bank for International Settlements)

### Monetary Policy
- Miranda-Agrippino & Rey — Global Financial Cycle factor
- USMPD — US Monetary Policy Surprises
- ISOM — International Spillover of Monetary Policy

### Institutional Data
- Romelli (2022) — Central Bank Independence indices
- FKRSU — Capital Controls indices
- World Bank WGI — Worldwide Governance Indicators

---

## Contributing

If you find useful resources that should be added to this curriculum, please open an issue or PR on the GitHub repository.
