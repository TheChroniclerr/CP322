import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def correlation_heatmap(name, df):
    """
    Plots a correlation heatmap to identify relationships between features.

    Useful for detecting multicollinearity.

    Parameters
    ----------
    name : str
        Figure identifier.
    df : pandas.DataFrame
        Dataset to analyze.
    """
    plt.figure(num=name, figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    return


def knn_hptuning(name, models, unscaled, scaled):
    """
    Visualizes KNN performance across different values of k.

    Evaluates RMSE for k in range [1, 39] and plots results.

    Helps identify the optimal number of neighbors.

    Parameters
    ----------
    name : str
        Figure identifier.
    models : list
        Trained models (unused except for structure consistency).
    unscaled : list
        Dataset splits.
    scaled : list
        Scaled dataset splits.
    """
    lr, knn, rf = models
    X_train, X_test, y_train, y_test = unscaled
    X_train_scaled, X_test_scaled = scaled

    rmse_values = []
    for k in range(1, 40):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)

        pred = knn.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_values.append(rmse)

    plt.figure(num=name)
    plt.plot(range(1,40), rmse_values)
    plt.xlabel("k")
    plt.ylabel("RMSE")
    plt.title("KNN Performance vs k")
    return


def rf_hptuning(name, models, unscaled):
    """
    Performs grid search visualization for Random Forest hyperparameters.

    Parameters tested:
    - n_estimators: [50, 100, 200]
    - max_depth: [5, 10, 15, None]

    Displays results as a heatmap of RMSE values.

    Parameters
    ----------
    name : str
        Figure identifier.
    models : list
        Trained models.
    unscaled : list
        Dataset splits.
    """
    lr, knn, rf = models
    X_train, X_test, y_train, y_test = unscaled

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
    }

    rmse_results = []
    for n in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            rf = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state=42
            )
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            rmse_results.append({"n_estimators": n, "max_depth": depth, "RMSE": rmse})

    rmse_df = pd.DataFrame(rmse_results)
    heatmap_data = rmse_df.pivot(index="max_depth", columns="n_estimators", values="RMSE")

    plt.figure(num=name, figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues_r", cbar_kws={'label': 'RMSE'})
    plt.title("Random Forest Hyperparameter Tuning")
    plt.ylabel("max_depth")
    plt.xlabel("n_estimators")
    return


def perf_comp_bar(name, unscaled, predictions):
    """
    Compares model performance using RMSE and R² metrics.

    Displays a bar plot of RMSE across models.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    # Model comparison table
    results = pd.DataFrame({
        "Model": ["Linear Regression", "KNN", "Random Forest"],
        "R2": [
            r2_score(y_test, lr_pred),
            r2_score(y_test, knn_pred),
            r2_score(y_test, rf_pred)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, lr_pred)),
            np.sqrt(mean_squared_error(y_test, knn_pred)),
            np.sqrt(mean_squared_error(y_test, rf_pred))
        ]
    })

    plt.figure(num=name)
    sns.barplot(data=results, x="Model", y="RMSE")
    plt.title("Model Performance Comparison")
    return


def graduation_rate_dist(name, df):
    """
    Plots the distribution of the target variable 'graduation_rate'.

    Displays a histogram to visualize the spread, skewness,
    and potential outliers in graduation rates.

    Parameters
    ----------
    name : str
        Figure identifier.
    df : pandas.DataFrame
        Dataset containing the 'graduation_rate' column.
    """
    plt.figure(num=name)
    sns.histplot(df["graduation_rate"], bins=20)
    plt.title("Distribution of Graduation Rates")
    return


def lr_feature_effects(name, df, models):
    """
    Visualizes feature effects using Linear Regression coefficients.

    Features are ranked by the absolute magnitude of their coefficients,
    indicating their relative influence on the target variable.

    Parameters
    ----------
    name : str
        Figure identifier.
    df : pandas.DataFrame
        Dataset including features and 'graduation_rate'.
    models : list
        Trained models [LinearRegression, KNN, RandomForest].

    Notes
    -----
    - Only applicable to linear models.
    - Coefficients assume input features were properly scaled.
    """
    lr, knn, rf = models
    X = df.drop("graduation_rate", axis=1)

    coeff_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lr.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    plt.figure(num=name)
    sns.barplot(data=coeff_df, x="Coefficient", y="Feature")
    plt.title("Linear Regression Feature Effects")
    return 


def lr_avp(name, unscaled, predictions):
    """
    Plots actual vs predicted values for Linear Regression.

    Used to assess how well predictions align with true values.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits [X_train, X_test, y_train, y_test].
    predictions : list
        Model predictions [lr_pred, knn_pred, rf_pred].
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    plt.figure(num=name)
    plt.scatter(y_test, lr_pred)
    plt.xlabel("Actual Graduation Rate")
    plt.ylabel("Predicted Graduation Rate")
    plt.title("Actual vs Predicted (Linear Regression)")
    return


def lr_residual(name, unscaled, predictions):
    """
    Plots residuals for Linear Regression predictions.

    Residuals (actual - predicted) are plotted against predicted values
    to assess model assumptions such as homoscedasticity.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    residuals = y_test - lr_pred

    plt.figure(num=name)
    plt.scatter(lr_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (Linear Regression)")
    return


def knn_avp(name, unscaled, predictions):
    """
    Plots actual vs predicted values for KNN regression.

    Helps evaluate prediction accuracy and detect systematic errors.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    plt.figure(num=name)
    plt.scatter(y_test, knn_pred)
    plt.xlabel("Actual Graduation Rate")
    plt.ylabel("Predicted Graduation Rate")
    plt.title("Actual vs Predicted (KNN)")
    return


def knn_residual(name, unscaled, predictions):
    """
    Plots residuals for KNN regression.

    Useful for identifying prediction patterns or bias.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    residuals = y_test - knn_pred

    plt.figure(num=name)
    plt.scatter(knn_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (KNN)")
    return


def rf_feature_importance(name, df, models):
    """
    Displays feature importance from the Random Forest model.

    Features are ranked based on their contribution to reducing
    prediction error across decision trees.

    Parameters
    ----------
    name : str
        Figure identifier.
    df : pandas.DataFrame
        Dataset including features and target variable.
    models : list
        Trained models.

    Notes
    -----
    - Importance values are relative and sum to 1.
    - Tree-based importance may be biased toward high-cardinality features.
    """
    lr, knn, rf = models
    X = df.drop("graduation_rate", axis=1)

    importances = rf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(num=name, figsize=(10,6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Random Forest Feature Importance")
    return


def rf_avp(name, unscaled, predictions):
    """
    Plots actual vs predicted values for Random Forest regression.

    Evaluates how closely predictions match true values.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    plt.figure(num=name)
    plt.scatter(y_test, rf_pred)
    plt.xlabel("Actual Graduation Rate")
    plt.ylabel("Predicted Graduation Rate")
    plt.title("Actual vs Predicted (Random Forest)")
    return


def rf_residual(name, unscaled, predictions):
    """
    Plots residuals for Random Forest predictions.

    Helps assess prediction errors and identify patterns
    that may indicate model limitations.

    Parameters
    ----------
    name : str
        Figure identifier.
    unscaled : list
        Dataset splits.
    predictions : list
        Model predictions.
    """
    X_train, X_test, y_train, y_test = unscaled
    lr_pred, knn_pred, rf_pred = predictions

    residuals = y_test - rf_pred

    plt.figure(num=name)
    plt.scatter(rf_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot (Random Forest)")
    return

def visualize(df, models, unscaled, scaled, predictions):
    """
    Runs the complete visualization pipeline.

    Generates plots for:
    - Feature correlations
    - Hyperparameter tuning (KNN, Random Forest)
    - Model performance comparison
    - Target distribution
    - Model diagnostics (actual vs predicted, residuals)
    - Feature importance and coefficients

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset including features and target variable.

    models : list
        Trained models [LinearRegression, KNN, RandomForest].

    unscaled : list
        [X_train, X_test, y_train, y_test]

    scaled : list
        [X_train_scaled, X_test_scaled]

    predictions : list
        [lr_pred, knn_pred, rf_pred]

    Returns
    -------
    matplotlib.pyplot
        Plotting object containing all generated figures.
    """
    correlation_heatmap("correlation_heatmap.png", df)
    knn_hptuning("knn_hptuning.png", models, unscaled, scaled)
    rf_hptuning("rf_hptuning.png", models, unscaled)
    perf_comp_bar("perf_comp_bar.png", unscaled, predictions)
    graduation_rate_dist("graduation_rate_dist.png", df)
    lr_feature_effects("lr_feature_effects.png", df, models)
    lr_avp("lr_avp.png", unscaled, predictions)
    lr_residual("lr_residual.png", unscaled, predictions)
    knn_avp("knn_avp.png", unscaled, predictions)
    knn_residual("knn_residual.png", unscaled, predictions)
    rf_feature_importance("rf_feature_importance.png", df, models)
    rf_avp("rf_avp.png", unscaled, predictions)
    rf_residual("rf_residual.png", unscaled, predictions)
    return plt