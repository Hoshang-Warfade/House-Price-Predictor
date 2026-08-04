from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


def evaluate_regression_model(
    y_true,
    y_pred,
    n_features=None,
    adjusted_r2=False,
):
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.

    y_pred : array-like
        Predicted target values.

    n_features : int, default=None
        Number of predictor variables used by the model.
        Required only when adjusted_r2=True.

    adjusted_r2 : bool, default=False
        Whether to compute the Adjusted R² score.

    Returns
    -------
    dict
        Dictionary containing regression evaluation metrics.
    """

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }

    if adjusted_r2:
        if n_features is None:
            raise ValueError(
                "n_features must be provided when adjusted_r2=True."
            )

        n_samples = len(y_true)

        metrics["Adjusted R²"] = (
            1
            - (1 - metrics["R²"])
            * (n_samples - 1)
            / (n_samples - n_features - 1)
        )

    return metrics



from sklearn.model_selection import cross_validate
import pandas as pd


def evaluate_cv_model(
    pipeline,
    X,
    y,
    cv,
    scoring,
):
    """
    Evaluate a regression model using cross-validation.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Complete preprocessing and regression pipeline.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    cv : Cross-validation splitter
        Cross-validation strategy.

    scoring : dict
        Dictionary of evaluation metrics.

    Returns
    -------
    pandas.DataFrame
        Cross-validation performance summary containing the
        mean and standard deviation of both training and
        validation metrics.
    """

    # Perform cross-validation
    cv_results = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # Convert negative error metrics to positive values
    train_mae = -cv_results["train_mae"]
    valid_mae = -cv_results["test_mae"]

    train_rmse = -cv_results["train_rmse"]
    valid_rmse = -cv_results["test_rmse"]

    train_r2 = cv_results["train_r2"]
    valid_r2 = cv_results["test_r2"]

    # Performance summary
    evaluation_summary = pd.DataFrame(
        {
            "Train Mean": [
                train_mae.mean(),
                train_rmse.mean(),
                train_r2.mean(),
            ],
            "Train Std": [
                train_mae.std(),
                train_rmse.std(),
                train_r2.std(),
            ],
            "Validation Mean": [
                valid_mae.mean(),
                valid_rmse.mean(),
                valid_r2.mean(),
            ],
            "Validation Std": [
                valid_mae.std(),
                valid_rmse.std(),
                valid_r2.std(),
            ],
        },
        index=[
            "MAE",
            "RMSE",
            "R²",
        ],
    )

    return evaluation_summary.round(4)