
#  FORESIGHT MODELING GUIDE

 
**Audience:** Institutional stakeholders, academic planners, and analysts  
**Prerequisite knowledge:** Basic algebra and percentages — no prior machine learning experience required

---

## 1. Executive Summary & Core Objective

### Why we predict Year 2 continuation

Every fall, a new cohort of students begins their program. By the end of Year 1, some students leave — and **Year 1 → Year 2 is the single highest-risk transition** in most undergraduate lifecycles. If we can estimate *who* is likely to continue and *how many* students a program will retain, we can:

| Planning need | How the model helps |
|---|---|
| **Early intervention** | Advisors can reach students *before* they disengage, while there is still time to offer support. |
| **Capacity planning** | Departments can estimate how many Year 2 seats, sections, and lab spots they will need. |
| **Tuition & budget forecasting** | Finance teams can project revenue from continuing students rather than relying on last year's headcount. |
| **Program health monitoring** | Leaders can spot whether a specific qualification pathway (e.g., a degree vs. a certificate) is retaining students at expected rates. |

Think of the model as a **weather forecast for enrolment**: it does not tell us with certainty whether any one student will stay, but it gives us a well-calibrated probability for each student — and those probabilities roll up into reliable program-level numbers.

### What we predict (and at what grain)

Each prediction is made for a unique combination of:

- **the_student's_institutional_ID**
- **StudyLevel** — the qualification pathway they entered (e.g., Bachelor, Master)

A single student can appear more than once if they pursued multiple qualification pathways over time. Each pathway is modeled independently because the risk profile of "first-year Economics" may differ from "first-year Engineering transfer."

**Target label:** `target_year2_continuation`  
- `1` = the student continued into Year 2 (or beyond) on that qualification pathway  
- `0` = the student did not continue into Year 2 on that pathway

### From individual probabilities to program headcount

The model outputs a probability $P_i$ for each student pathway $i$. To estimate how many students will continue, we **sum the probabilities** rather than simply counting yes/no predictions:

$$
\text{Expected Year 2 continuers} = \sum_{i=1}^{N} P_i
$$

**Concrete example:** Suppose a program has 4 entering students with these model scores:

| Student | $P_i$ (continue) | Hard prediction ($P_i \geq 0.5$) |
|---|---|---|
| Alex | 0.90 | Yes (1) |
| Blake | 0.70 | Yes (1) |
| Casey | 0.40 | No (0) |
| Dana | 0.30 | No (0) |

- **Hard count** (threshold at 0.5): $1 + 1 + 0 + 0 = \mathbf{2}$ continuers  
- **Expected value** (sum of probabilities): $0.90 + 0.70 + 0.40 + 0.30 = \mathbf{2.30}$ continuers

The expected-value approach is smoother and more accurate at the program level because it respects uncertainty. A student at 40% risk still contributes partial "expected" enrolment — which better reflects reality than treating them as a definite departure.

This rollup is exactly what the pipeline produces in files like `outputs/enrolment_demand_forecast_test.csv` and `outputs/logistic_regression_forecast.csv`, where `predicted_y2_expected` is the sum of $\sum P_i$ by cohort year, qualification, and program.

---

## 2. Data Preparation & Feature Engineering

### Overview: two data worlds, one modeling table

Our pipeline combines **internal student records** with **external environmental indicators** into a single analysis-ready table. The work happens in two stages:

1. **SQL feature engineering** (`01_feature_engineering.sql`) — run inside DuckDB  
2. **Python model training** (`enrollment_prediction.py`) — encoding, scaling, and prediction

```
lifecycle_data.csv  ──►  SQL (DuckDB)  ──►  enrollment_modeling_dataset.csv
                                    │
                                    └──►  enrollment_prediction.py  ──►  outputs/
```

### Database sources

#### Internal: student lifecycle snapshot (`lifecycle_data.csv`)

This file contains one row per student per term in their academic journey. Key fields include:

- **Identity**
- **Cohort timing**
- **Program structure:** 
- **Demographics:** residency (`Intern_domest`), sex, immigration group  
- **Outcome tracking:** 

The SQL layer collapses this raw history into **one row per student ID + level of study**, using only information available at or near entry (the "COHORTE" snapshot — the student's first recorded term on that qualification).

#### External: environmental indicators (`external_indicators` table)

Macro-level conditions that may influence enrolment patterns are joined by **cohort entry year**. In production, these would be sourced from official series such as:

| Indicator in our model | Real-world source (production target) |
|---|---|
| `labour_market_unemployment` | Statistics Canada Labour Force Survey |
| `immigration_policy_tightness` | IRCC policy / study permit processing trends |
| `intl_student_cap_pressure` | Federal/provincial international student cap announcements |
| `youth_demographic_index` | StatsCan population / youth cohort projections |
| `geopolitical_risk_index` | Composite economic / global risk proxy (e.g., ESDC, World Bank) |

> **Note:** In the current `Enrollment_revised` prototype, these external series are **placeholder values** embedded in SQL. Before production deployment, they should be replaced with live feeds from StatsCan, IRCC, and ESDC.

#### Historical enrolment trends (lagged, leakage-safe)

For each qualification and program, the SQL computes **prior-year** headcount and Year 2 continuation rates — never using the current year's outcomes to predict itself. These become features like `hist_prior_y2_rate` and `hist_roll3_y2_rate`.

### Data preprocessing made simple

Raw data contains numbers on different scales (e.g., unemployment rate ≈ 5–10, headcount ≈ 0–500) and text labels (e.g., `Faculty = Science`). Models need these in a consistent numeric format.

#### Step 1: Missing value imputation

Before scaling or encoding, gaps are filled:

- **Numeric columns:** median imputation (the middle value — robust to outliers)  
- **Categorical columns:** most frequent category

#### Step 2: Z-score standardization (`StandardScaler`)

Continuous numeric features are rescaled so the training data has **mean $\mu = 0$** and **standard deviation $\sigma = 1$**:

$$
Z = \frac{x - \mu}{\sigma}
$$

**Why?** Features like "years in post-secondary" (range 0–5) and "historical headcount" (range 0–500) would otherwise dominate the model simply because their numbers are larger. Z-scoring puts them on equal footing.

**Intuition:** If the average unemployment rate is 6% with little variation, a year at 9% becomes a "high" Z-score — the model learns *relative* unusualness, not raw units.

Numeric features scaled in our pipeline include:

Cohort/Program inforamtion, , study-duration fields, international/coop/load flags, historical trend fields, and all five external indicators.

#### Step 3: One-hot encoding (`OneHotEncoder`)

Text categories cannot be fed directly into math equations. **One-hot encoding** converts each category into a set of 0/1 columns.

**Example — Residency status (`intern_domest`):**

| Original value | `intern_domest_International` | `intern_domest_Domestic` |
|---|---|---|
| International | 1 | 0 |
| Domestic | 0 | 1 |

**Example — Faculty** (if a student is in Engineering):

| `faculty_Arts` | `faculty_Engineering` | `faculty_Science` | … |
|---|---|---|---|
| 0 | 1 | 0 | … |

Only one faculty column is `1` for any given student; all others are `0`. This prevents the model from incorrectly treating faculty names as ordered numbers (Engineering ≠ "2 × Arts").

Categorical features encoded in our pipeline include: application categories, immigration group, residency, sex, subject, program language, load, coop, program, faculty, academic unit, program code, academic load, and coop indicator.

### Columns excluded from model training

Not every column in the dataset should be used as a model input. Using the wrong columns causes **data leakage** (explained below) or meaningless predictions.

| Column type | Examples | Why excluded |
|---|---|---|
| **Student identifiers** | IDs carry no predictive pattern; including them would memorize individuals |
| **Target label** | `target_year2_continuation` | This is what we are trying to predict — using it as input would be cheating |
| **Post-outcome fields** | `max_years_observed`, `ever_graduated`, `lifecycle_row_count` | These are only known *after* the student has progressed — not available at prediction time |
| **Maturity flags** | `is_mature_cohort` | Used to filter training data, not as a feature |
| **Descriptive metadata** | `plan`, `program`, `entering_subject`, `reg_subject`, `acad_year` | Redundant with encoded features or not used in the current model spec |

> **Important:** `coh_year` *is* included as a numeric feature because cohort-era effects (e.g., COVID-19 entry year) matter for prediction. However, it is also the basis for our **temporal train/test splits** — the model never trains on future years to predict past years.

### Data leakage prevention

**Data leakage** occurs when information from the *future* accidentally slips into the model's inputs, making offline performance look great but failing in real deployment.

**Student example:**

> **Wrong (leaky):** Use "total credits earned in Year 1 Winter term" to predict whether a student shows up on Day 1 of Year 2.  
> By Winter of Year 1, we already know the student survived Fall — that is partially the answer we are trying to predict early.

> **Right (leakage-safe):** Use only information known at **entry** — program, residency, coop intent, prior institutional trends, and macro indicators for that entry year.

Our SQL enforces this by:

1. Taking the **COHORTE** (entry) snapshot per StudentID + StudyLevel 
2. Computing historical trend features with **lagged** windows (prior years only)  
3. Training only on **mature cohorts** where Year 2 outcomes have actually been observed

---

## 3. Model Training & Evaluation (Walk-Forward Validation)

### Why not standard K-fold cross-validation?

In a typical K-fold cross-validation, the dataset is shuffled and split into $K$ random folds. Each fold takes a turn being the "test" set. This works well for static problems (e.g., classifying flower species from measurements).

**Enrolment data is not static — it is temporal.** Student cohorts arrive year by year, policies change, and economic conditions shift. If we randomly shuffle 2015 students into the same training pool as 2023 students, the model can implicitly "peek" at the future:

- It learns patterns from 2022–2023 while being tested on 2014.  
- This creates **look-ahead bias** — performance looks artificially high.  
- In production, we only ever predict *forward* in time, so our validation must mirror that.

**Analogy:** Random K-fold is like studying for a history exam by reading the answer key first. Walk-forward validation is like taking practice exams in chronological order — the only honest way to know if you are ready for next year.

### Historical walk-forward validation

Our pipeline (`historical_walk_forward_validate` in `enrollment_prediction.py`) simulates how the model would have performed if deployed year by year.

**Rule:** At every step, training data comes from cohort years **strictly before** the test year:

$$
T_{\text{train}} < T_{\text{test}}
$$

#### Step-by-step walk-through (2010–2023 cohorts)

We require at least **5 cohort years** of training history before the first test fold (`MIN_TRAIN_COHORTS = 5`).

| Fold | Training cohorts | Test cohort | What this simulates |
|---|---|---|---|
| 1 | 2010–2014 | 2015 | "If we deployed in 2015, how would we score the 2015 entrants?" |
| 2 | 2010–2015 | 2016 | Expanding window — more history each year |
| 3 | 2010–2016 | 2017 | … |
| 4 | 2010–2017 | 2018 | … |
| 5 | 2010–2018 | 2019 | … |
| 6 | 2010–2019 | 2020 | Includes COVID-era entry cohort |
| 7 | 2010–2020 | 2021 | … |
| 8 | 2010–2021 | 2022 | … |
| 9 | 2010–2022 | 2023 | Most recent mature cohort |

Each fold trains a fresh model, scores the held-out year, and records ROC-AUC, precision, recall, and F1. The **mean and standard deviation** across folds (reported as `wf_roc_auc_mean` in `outputs/model_comparison.csv`) tell us whether the model is **stable over time** or drifting.

Fold-level detail is saved to `outputs/walk_forward_validation_folds.csv`.

### Data splitting: training, validation, and test sets

After walk-forward validation establishes historical stability, we fit the final candidate models using a **temporal holdout structure**. Think of it like school:

| Set | Role | Analogy | Our pipeline (cohort years) |
|---|---|---|---|
| **Training** | Learn patterns from the past | Textbook + lectures | 2010 – 2020 (~4,920 pathways) |
| **Validation** | Tune decisions; practice quiz | Practice quiz before the final | 2021 (~438 pathways) |
| **Test** | Final unbiased exam — never touched during training | Final exam | 2022 – 2023 (~649 pathways) |

#### About the "60 / 20 / 20" guideline

Many textbooks recommend splitting data **60% train / 20% validation / 20% test** when rows are exchangeable (i.i.d.). That proportion is a useful **conceptual benchmark** for the three roles above.

**Our enrolment pipeline uses a time-based split instead of a random 60/20/20**, because student cohorts must stay in chronological order. With 14 mature cohort years (2010–2023), the approximate row shares are:

- **Training:** ~82% (oldest 11 cohort years)  
- **Validation:** ~7% (1 recent cohort year)  
- **Test:** ~11% (2 most recent cohort years)

This is intentional: the test set represents **the most recent real-world conditions** the model would face in production — including post-pandemic and policy-shift cohorts. A random 60/20/20 split would scatter recent years into training and destroy the temporal integrity we need.

### Evaluation metric: ROC-AUC explained

**ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) measures how well the model **ranks** students by risk.

- **0.5** = no better than a coin flip (random ordering)  
- **1.0** = perfect separation (every continuer scored higher than every non-continuer)  
- **0.7 – 0.8** = good, useful discrimination for planning purposes  
- **Our selected model (Logistic Regression):** walk-forward ROC-AUC ≈ **0.757**, test ROC-AUC ≈ **0.722**

#### Concrete 2-student comparison

| Student | Actual outcome | Model score $P_i$ |
|---|---|---|
| Student A | Continued (1) | 0.85 |
| Student B | Did not continue (0) | 0.30 |

The model ranked A above B — **correct ordering**. ROC-AUC captures whether this ordering holds across *all* pairs of students, not just whether the 0.5 threshold was perfect.

**Why we care about ranking:** Advising teams often work top-down — contact the highest-risk students first. A model that ranks well is valuable even if the exact 0.5 cutoff is imperfect.

#### Other metrics reported

| Metric | Plain-language meaning |
|---|---|
| **Precision** | Of students flagged "at risk," how many actually left? |
| **Recall** | Of students who actually left, how many did we flag? |
| **F1** | Balance between precision and recall |
| **Confusion matrix** | Table of true/false positives and negatives at the 0.5 threshold |

---

## 4. Model Selection & Explainability Mechanics

### Models compared

We train four candidate models and select the best balance of accuracy, interpretability, and maintainability:

| Model | Type | Strengths | Trade-offs |
|---|---|---|---|
| **Logistic Regression** | Linear | Transparent coefficients; fast; easy to audit | Assumes additive, linear feature effects |
| **Random Forest** | Tree ensemble | Captures interactions; robust | Harder to explain; slower to update |
| **XGBoost** | Gradient-boosted trees | High accuracy on structured data | More complex; needs careful monitoring |
| **LightGBM** | Gradient-boosted trees | Fast on large datasets | Similar to XGBoost |

**Linearity vs. interactions — intuitive example:**

- *Logistic Regression* might learn: "Being international adds +0.3 log-odds of risk, and coop adds −0.2" — a fixed recipe.  
- *Tree models* might learn: "IF international AND program = X AND unemployment > 7%, THEN risk jumps" — conditional recipes.

### How the final model is chosen

Selection uses a weighted score (`selection_score`):

- **70%** — test-set performance (ROC-AUC, F1, recall, precision)  
- **15%** — interpretability (Logistic Regression scores highest)  
- **15%** — operational maintainability (simplicity of deployment and monitoring)

**Current selection:** Logistic Regression (test ROC-AUC = 0.722, walk-forward ROC-AUC = 0.757)

Full comparison: `outputs/model_comparison.csv`  
Selection rationale: `outputs/final_model_selection.json`

### Explaining risk drivers

Stakeholders rightly ask: *"Why is this student flagged?"* We answer with **feature contributions** — how each input pushed the score up or down.

#### Logistic Regression: coefficient weights ($\beta$)

Logistic Regression converts a weighted sum of features into a probability:

$$
P(\text{continue}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots)}}
$$

Each coefficient $\beta_j$ tells us the **direction and strength** of feature $j$:

- **Positive $\beta$** → higher values of that feature increase continuation probability  
- **Negative $\beta$** → higher values decrease continuation probability

Top coefficients for the fitted model are exported to `outputs/model_interpretation.csv`.

#### Tree models: feature importances and SHAP (TreeSHAP)

For Random Forest, XGBoost, and LightGBM, global **feature importances** show which inputs the trees relied on most often. For individual-level explanations suitable for advising dashboards, **SHAP values** (SHapley Additive exPlanations) are the industry standard:

- Each feature gets a contribution score for *that specific student*  
- Contributions sum to the difference between the student's score and the average score  
- TreeSHAP is the efficient SHAP algorithm for tree-based models

> **Current pipeline status:** `enrollment_prediction.py` exports global coefficients (Logistic Regression) and feature importances (tree models). **Per-student SHAP scoring is the recommended next step** for dashboard deployment.

### Risk grouping and dashboard driver aggregation

Once each student pathway has a probability $P_i$, we assign a **risk tier** for operational use:

| Risk tier | Typical threshold | Suggested action |
|---|---|---|
| **Low risk** | $P_i \geq 0.70$ | Standard engagement |
| **Moderate risk** | $0.40 \leq P_i < 0.70$ | Proactive outreach, tutoring referral |
| **High risk** | $P_i < 0.40$ | Priority advising, retention case management |

*(Thresholds are configurable by institutional policy; 0.5 is the default hard classifier in the pipeline.)*

#### Why driver rankings focus on flagged students

A common dashboard mistake is averaging feature contributions across **all** students — including low-risk students who dilute the signal. Our recommended approach:

1. Score every studemt + level of study pathway → $P_i$  
2. Filter to **flagged at-risk students only** (Moderate + High Risk)  
3. Average SHAP values (or absolute coefficients × standardized feature values) **within that filtered group**  
4. Display the top drivers — e.g., "Among at-risk Economics entrants, high unemployment year and part-time load are the dominant factors"

This answers the question planners actually ask: *"What is driving risk among the students we need to help?"* — not *"What is average across everyone?"*

Per-student scores for the held-out test period are available in `outputs/student_predictions_test.csv`:

```
StudentID | Studylevel | cohort_year | program | y_true | y_prob | y_pred
```

---

## 5. Step-by-Step Prediction Workflow

### End-to-end ASCII flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │ lifecycle_data2.csv  │    │ External indicators (StatsCan / IRCC /   │   │
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

### How to run the pipeline

```powershell
cd c:\Python\File_revised
python enrollment_prediction.py
```

### Key output files

| File | Contents |
|---|---|
| `outputs/enrollment_modeling_dataset.csv` | Full feature matrix for mature cohorts |
| `outputs/student_predictions_test.csv` | Per-student probabilities  |
| `outputs/model_comparison.csv` | Walk-forward and test metrics for all models |
| `outputs/walk_forward_validation_folds.csv` | Year-by-year fold performance |
| `outputs/model_interpretation.csv` | Top feature coefficients / importances |
| `outputs/enrolment_demand_forecast_test.csv` | Expected continuers by cohort × qualif × program |
| `outputs/logistic_regression_forecast.csv` | Institution-level demand forecast series |
| `outputs/final_model_selection.json` | Selected model and rationale |

---

## Glossary

| Term | Definition |
|---|---|
| **StudentID** | Unique student identifier in the institutional system |
| **StudyLevel** | Qualification pathway code (degree, certificate, etc.) |
| **COHORTE** | The entry-term snapshot row in lifecycle data |
| **Walk-forward validation** | Train on the past, test on the next future period, repeat |
| **ROC-AUC** | Ranking quality score from 0.5 (random) to 1.0 (perfect) |
| **Data leakage** | Using future information to predict the past |
| **Expected value ($\sum P_i$)** | Sum of continuation probabilities → program headcount estimate |
| **SHAP** | Method to explain each feature's contribution to an individual prediction |
| **Z-score** | Standardized value: how many standard deviations from the mean |

---

*Document version: 1.0 — aligned with `File_revised` pipeline (prediction grain: studentID + StudyLevel; validation: Historical Walk-Forward)*
