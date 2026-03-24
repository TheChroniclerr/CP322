def predict(models, unscaled, scaled):
    """
    Generates predictions on test data using trained regression models.

    Steps performed:
    1. Unpacks trained models and dataset splits.
    2. Applies each model to the appropriate test data:
        - Scaled data for Linear Regression and KNN
        - Unscaled data for Random Forest
    3. Returns predictions from all models.

    Parameters
    ----------
    models : list
        Trained models in the order:
        [LinearRegression, KNeighborsRegressor, RandomForestRegressor]

    unscaled : list
        Original (unscaled) dataset splits:
        [X_train, X_test, y_train, y_test]

    scaled : list
        Scaled dataset splits:
        [X_train_scaled, X_test_scaled]

    Returns
    -------
    list
        Predictions from each model in the order:
        [lr_pred, knn_pred, rf_pred]

    Notes
    -----
    - Linear Regression and KNN require scaled inputs for accurate predictions.
    - Random Forest operates on unscaled data.
    - Assumes input lists are in the correct order and consistent with the `train()` function output.
    """
    # Unpack lists
    lr, knn, rf = models
    X_train, X_test, y_train, y_test = unscaled
    X_train_scaled, X_test_scaled = scaled

    # Predict using models
    lr_pred = lr.predict(X_test_scaled)
    knn_pred = knn.predict(X_test_scaled)
    rf_pred = rf.predict(X_test)

    return [lr_pred, knn_pred, rf_pred]