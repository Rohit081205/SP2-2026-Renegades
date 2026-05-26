# Sona Power Predict — 2026

| | |
|---|---|
| **College** | Sri Krishna College of Technology |
| **Team Name** | Renegades |

---

## Team Members

| Name | Year | Department |
|---|---|---|
| Shaheenur Rahman M | Year 3 | B.Tech Artificial Intelligence and Data Science |
| Saravanakumar M | Year 3 | B.Tech Artificial Intelligence and Data Science |
| Rohit T J | Year 3 | B.Tech Artificial Intelligence and Data Science |
| Praveen Kumar K | Year 3 | B.Tech Artificial Intelligence and Data Science |

---

## Libraries Used in Model

The following libraries are used in `mymodelfile.py`, categorized by their purpose.

### Data Analysis & Modeling Libraries

These are the core libraries used specifically for manipulating data, performing mathematical operations, and building the predictive model.

#### `pandas` (`pd`) — Data Analysis & Manipulation

The primary tool for data analysis and manipulation.

**Purpose:** Handles data wrangling tasks such as:
- Loading CSV datasets into DataFrames (`pd.read_csv`)
- Filtering rows and handling missing values
- Merging tables (e.g., matches and deliveries)
- Grouping data to calculate statistics (e.g., aggregating total runs per batsman or calculating average strike rates)

---

#### `numpy` (`np`) — Numerical Computation

The core library for numerical computations.

**Purpose:** Supports analysis by handling underlying mathematical operations and array manipulations, including:
- Formatting input data for the machine learning model (`np.array`)
- Safely checking for missing numeric values (`np.isnan`)
- Constraining calculated heuristic scores within logical bounds (`np.clip`)

---

#### `lightgbm` (`lgb`) — Predictive Modeling

A fast, distributed gradient boosting framework used for predictive analysis.

**Purpose:** Takes the engineered features prepared by `pandas` and `numpy` to:
- Train a regression model (`lgb.train`)
- Find complex patterns in data to predict score adjustments based on the current match state (`lgb.predict`)
