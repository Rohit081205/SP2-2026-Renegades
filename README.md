# Sona Power Predict - 2026

# College Name: Sri Krishna College of Technology
#Team Name: Renegades

# Library Analysis for `mymodelfile.py`

Below is a breakdown of the libraries used in `mymodelfile.py`, categorized by their purpose.

## Data Analysis & Modeling Libraries

These are the core libraries used specifically for manipulating data, performing mathematical operations, and building the predictive model.

*   **`pandas` (`pd`)**: The primary tool for **data analysis** and manipulation. 
    *   **Purpose**: Handles data wrangling tasks such as loading CSV datasets into DataFrames (`pd.read_csv`), filtering rows, handling missing values, merging tables (like matches and deliveries), and grouping data to calculate statistics (e.g., aggregating total runs per batsman or calculating average strike rates).
*   **`numpy` (`np`)**: The core library for numerical computations.
    *   **Purpose**: Supports the analysis by handling underlying mathematical operations and array manipulations. It is used to format input data for the machine learning model (`np.array`), safely check for missing numeric values (`np.isnan`), and constrain calculated heuristic scores within logical bounds (`np.clip`).
*   **`lightgbm` (`lgb`)**: A fast, distributed gradient boosting framework used for **predictive analysis**.
    *   **Purpose**: Takes the engineered features prepared by pandas and numpy to train a regression model (`lgb.train`). It finds complex patterns in the data to predict score adjustments based on the current match state (`lgb.predict`).
