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