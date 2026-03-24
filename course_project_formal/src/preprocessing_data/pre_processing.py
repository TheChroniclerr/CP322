import numpy as np
import pandas as pd

column_names = [
    "fice",
    "college_name",
    "state",
    "public_private",
    "avg_math_sat",
    "avg_verbal_sat",
    "avg_combined_sat",
    "avg_act",
    "math_sat_q1",
    "math_sat_q3",
    "verbal_sat_q1",
    "verbal_sat_q3",
    "act_q1",
    "act_q3",
    "applications",
    "accepted",
    "enrolled",
    "pct_top10",
    "pct_top25",
    "fulltime_undergrad",
    "parttime_undergrad",
    "tuition_in_state",
    "tuition_out_state",
    "room_board_cost",
    "room_cost",
    "board_cost",
    "additional_fees",
    "book_cost",
    "personal_spending",
    "pct_faculty_phd",
    "pct_faculty_terminal",
    "student_faculty_ratio",
    "pct_alumni_donate",
    "instructional_expenditure",
    "graduation_rate"
]

drops = [
    # Identifiers
    "fice",
    "college_name",
    "state",
    # Missing
    "act_q3",
    "act_q1",
    "avg_act",
    "math_sat_q1",
    "math_sat_q3",
    "verbal_sat_q1",
    "verbal_sat_q3",
    "avg_math_sat",
    "avg_verbal_sat",
    "avg_combined_sat",
    # Redundant
    "room_cost",
    "board_cost"
]

def preprocessing(df):
    """
    Preprocesses a pandas DataFrame by cleaning, converting, and imputing data.

    Steps performed:
    1. Assigns predefined column names to the DataFrame.
    2. Replaces '*' entries with NaN (missing values).
    3. Converts all data to numeric format, coercing invalid values to NaN.
    4. Drops specified columns (e.g., identifiers, redundant features, and those with excessive missing values).
    5. Fills remaining missing values using the median of each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input raw DataFrame to be preprocessed.

    Returns
    -------
    pandas.DataFrame
        Cleaned and preprocessed DataFrame with numeric values and no missing data.

    Notes
    -----
    - Requires global variables:
        * column_names : list of column names to assign
        * drops : list of column names to remove
    - Columns with more than 500 missing values should be included in `drops`.
    - Median imputation is applied only after column removal.
    """
    # Set feature names
    df.columns = column_names

    # Replace '*' with nan
    df.replace("*", np.nan, inplace=True)

    # Convert data to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop identifiers,redundant features, and feature with too many missing values (>500)
    df = df.drop(columns=drops)

    # Median imputate (<500)
    df.fillna(df.median(), inplace=True)

    return df