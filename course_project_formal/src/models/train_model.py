from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def train(df):
    """
    Splits data, scales features, and trains multiple regression models.

    Steps performed:
    1. Splits the dataset into features (X) and target (y), where the target is 'graduation_rate'.
    2. Performs an 80/20 train-test split.
    3. Applies standard scaling to feature data for models sensitive to feature magnitude.
    4. Trains three regression models:
        - Linear Regression
        - K-Nearest Neighbors Regressor
        - Random Forest Regressor
    5. Returns trained models along with raw and scaled datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed DataFrame containing features and the target variable 'graduation_rate'.

    Returns
    -------
    tuple
        (
            models : list
                [LinearRegression, KNeighborsRegressor, RandomForestRegressor],
            data_splits : list
                [X_train, X_test, y_train, y_test],
            scaled_data : list
                [X_train_scaled, X_test_scaled]
        )

    Notes
    -----
    - Scaling is applied only to Linear Regression and KNN, as Random Forest does not require feature scaling.
    - Random state is fixed for reproducibility.
    - KNN uses 12 neighbors and Random Forest uses 100 trees with a max depth of 15.
    - Assumes 'graduation_rate' exists in the DataFrame.
    """
    # 80/20 split
    X = df.drop("graduation_rate", axis=1)
    y = df["graduation_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit models
    lr = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=12)
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

    lr.fit(X_train_scaled, y_train)
    knn.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)

    return ([lr, knn, rf], 
        [X_train, X_test, y_train, y_test], 
        [X_train_scaled, X_test_scaled])