# Data-Driven Foresight

**From historical evidence to what may happen next.** 

This repository presents a data-driven foresight: using longitudinal data, statistical modeling, validation, and predictive analysis to move from historical evidence toward forward-looking intelligence.

The case study focuses on **enrollment continuation forecasting**, but the underlying approach is broader. The same reasoning can be applied wherever organizations need to understand what happened, identify changing patterns, and anticipate what may happen next.

 [📊 View Foresight Modeling Guide](https://github.com/panxiaoyan225-sudo/Data-Driven-Foresight/blob/main/FORESIGHT_MODELING_GUIDE.md)

## Why This Matters

Predictive analytics is not only about producing a forecast.

The more important question is:

> **What can we know about what may happen next, before it becomes visible in the reported numbers?**

A useful foresight system connects:

**historical evidence → patterns → prediction → validation → decision**

The model is one component of that process.

## The Foresight Approach

**Understand the problem → understand the underlying data → define the prediction unit → model historical patterns → validate against the future → compare predictive approaches → communicate what may happen next.**

The emphasis is not simply on selecting a predictive model. Reliable foresight depends on first understanding the problem and the structure of the data, defining the population correctly, choosing validation methods that respect the temporal nature of the problem, and interpreting model results in a decision context.

## Case Study: Enrollment Continuation Forecasting

This project uses enrollment data as a demonstration of how historical student pathways can be transformed into forward-looking continuation probabilities.

The analysis considers students within their respective qualification levels and uses the first term within each qualification as the cohort reference point. This avoids treating a student's earliest university enrollment as the cohort anchor when the same student may subsequently enter a different level of study.

The objective is to estimate the probability of second-year continuation from historical enrollment patterns.

## Modeling


I  train four candidate models and select the best balance of accuracy, interpretability, and maintainability:

| Model | Type | Strengths | Trade-offs |
|---|---|---|---|
| **Logistic Regression** | Linear | Transparent coefficients; fast; easy to audit | Assumes additive, linear feature effects |
| **Random Forest** | Tree ensemble | Captures interactions; robust | Harder to explain; slower to update |
| **XGBoost** | Gradient-boosted trees | High accuracy on structured data | More complex; needs careful monitoring |
| **LightGBM** | Gradient-boosted trees | Fast on large datasets | Similar to XGBoost |

### How the final model is chosen

Selection uses a weighted score (`selection_score`):

- **70%** — test-set performance (ROC-AUC, F1, recall, precision)  
- **15%** — interpretability (Logistic Regression scores highest)  
- **15%** — operational maintainability (simplicity of deployment and monitoring)

The purpose of comparing models is not to assume that the most complex model is automatically the best model. Model performance must be evaluated against the structure and purpose of the prediction problem.

## Temporal Validation

Enrollment data is inherently temporal. The project therefore uses **historical forward validation** to evaluate predictive performance.

Rather than randomly mixing observations from different periods, the validation approach respects the direction of time:

**Past → model → future period → evaluate**

This provides a more realistic assessment of how a predictive model may perform when used to anticipate future enrollment outcomes.

## Model Evaluation

The repository includes visual evidence of model evaluation, including confusion matrices and forecast outputs.

These results are provided without exposing underlying student-level data.

### Included evidence

- `confusion_logistic_regression.png`
- `confusion_random_forest.png`
- `confusion_xgboost.png`
- `logistic_regression_forecast.png`

Additional figures may be added as the analysis develops.

## Methodology Guide

The detailed methodology is documented in:

**`FORESIGHT_MODELING_GUIDE.md`**

The guide focuses on the reasoning behind the prediction:

1. Problem definition
2. Data structure
3. Cohort definition
4. Prediction target
5. Modeling approaches
6. Temporal validation
7. Model evaluation
8. Forecast interpretation

The goal is to make the analytical reasoning visible without exposing confidential or restricted institutional data.


## Data and Privacy

No student-level data, confidential institutional records, or restricted source data are included in this public repository.

The repository contains methodology documentation and selected analytical outputs intended to demonstrate the modeling and foresight approach.

## Broader Application

Although the case study is based on enrollment continuation, the underlying approach is not limited to higher education.

The same analytical discipline can support questions such as:

- What may happen to demand?
- Which risks are emerging?
- Where might capacity become constrained?
- Which outcomes may change?
- What signals appear before a decline becomes visible?
- What should decision-makers watch next?

The domain changes. The reasoning remains.

---

**Data-Driven Foresight**

*From historical evidence to what may happen next.*


**Step-by-Step Prediction Workflow**

### End-to-end ASCII flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │ lifecycle_data       │    │ External indicators (StatsCan / IRCC /   │   │
│  │ (student snapshots)  │    │ ESDC proxies by cohort year)             │   │
│  └──────────┬───────────┘    └──────────────────┬───────────────────────┘   │
└─────────────┼───────────────────────────────────┼───────────────────────────┘
              │                                   │
              ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              SQL FEATURE ENGINEERING  (01_feature_engineering.sql)          │
│  • One row per StudentID + studylevel                                          │
│  • Entry-time (COHORTE) snapshot only                                       │
│  • Lagged historical trends (no leakage)                                    │
│  • Join external indicators by coh_year                                       │
│  • Label: target_year2_continuation                                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              CLEANING & ENCODING  (enrollment_prediction.py)                  │
│  • Impute missing values                                                    │
│  • StandardScaler  →  Z-scores for numeric features                         │
│  • OneHotEncoder   →  0/1 flags for categorical features                    │
│  • Exclude IDs, targets, and post-outcome fields                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         HISTORICAL WALK-FORWARD VALIDATION                                  │
│  Fold 1: Train 2010-2014 → Test 2015                                        │
│  Fold 2: Train 2010-2015 → Test 2016                                        │
│  ...                                                                          │
│  Fold 9: Train 2010-2022 → Test 2023                                        │
│  → Measure ROC-AUC stability & model drift over time                        │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         TEMPORAL TRAIN / VALIDATION / TEST SPLIT                            │
│  Train: 2010-2020  |  Validation: 2021  |  Test: 2022-2023                 │
│  → Fit Logistic Regression, Random Forest, XGBoost, LightGBM                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY SCORING                                      │
│  Each EMPLID + COH_QUALIF  →  P(continue to Year 2)  =  P_i                │
│  Program headcount estimate  →  Σ P_i  (expected continuers)                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RISK GROUPING                                            │
│  P_i ≥ 0.70  →  Low Risk                                                    │
│  0.40 ≤ P_i < 0.70  →  Moderate Risk                                        │
│  P_i < 0.40  →  High Risk                                                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         SHAP / COEFFICIENT DRIVER AGGREGATION                               │
│  Logistic Regression  →  β coefficients (global + per-feature direction)    │
│  Tree models          →  TreeSHAP per student (dashboard recommended)       │
│  Dashboard view       →  Average drivers among Moderate + High Risk only    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         EXECUTIVE DASHBOARD & INTERVENTIONS                                 │
│  • Cohort/program demand forecasts (Σ P_i by year × qualif × program)       │
│  • At-risk student lists for advising outreach                                │
│  • Top risk drivers for policy conversations                                  │
│  • Walk-forward stability charts for model governance                       │
└─────────────────────────────────────────────────────────────────────────────┘
```