def build(df):
    """
    Performs feature engineering and feature selection on a DataFrame.

    Steps performed:
    1. Creates new ratio-based features:
        - acceptance_rate: proportion of accepted applications
        - yield_rate: proportion of accepted students who enroll
    2. Removes original columns used to compute the new features to avoid redundancy.
    3. Drops highly correlated features (correlation > 0.70) to reduce multicollinearity.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing admissions and institutional data.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with engineered features and reduced multicollinearity.

    Feature Engineering
    -------------------
    - acceptance_rate = accepted / applications
    - yield_rate = enrolled / accepted

    Feature Selection
    -----------------
    The following columns are removed due to high correlation:
    - pct_faculty_terminal : highly correlated with pct_faculty_phd (~0.86)
    - tuition_in_state     : highly correlated with other tuition-related variables (~0.92)
    - pct_top25            : highly correlated with pct_top10 (~0.88)

    Notes
    -----
    - Assumes no division-by-zero issues; ensure 'applications' and 'accepted' contain non-zero values.
    - Removing original columns helps prevent multicollinearity in downstream models.
    - Correlation thresholds are based on prior analysis and may need adjustment for new datasets.
    """
    # Acceptance rate
    df["acceptance_rate"] = df["accepted"] / df["applications"]

    # Yield rate
    df["yield_rate"] = df["enrolled"] / df["accepted"]

    # Drop original data to avoid multicollinearity
    df = df.drop(columns=["accepted", "applications", "enrolled"])

    # Drop highly correlated feature (>0.70)
    df = df.drop(columns=[
        "pct_faculty_terminal",     # 0.86 with pct_faculty_phd
        "tuition_in_state",         # 0.92 with tuition_in_state, 0.77 with public_private
        "pct_top25"                 # 0.88 with pct_top10
    ])

    return df