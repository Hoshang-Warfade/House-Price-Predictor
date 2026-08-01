"""
Pipeline creation utilities for the House Price Prediction project.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.transformers import (
    NumericalMissingValueTransformer,
    CategoricalMissingValueTransformer,
    FeatureEngineeringTransformer,
)


def create_pipeline(model):
    """
    Create the complete machine learning pipeline.

    Parameters
    ----------
    model : sklearn estimator
        Regression model to be used as the final estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing and modeling pipeline.
    """

    # -------------------------------------------------------------
    # Dataset-level preprocessing
    # -------------------------------------------------------------

    dataset_preprocessing_pipeline = Pipeline(
        steps=[
            (
                "numerical_missing_values",
                NumericalMissingValueTransformer(),
            ),
            (
                "categorical_missing_values",
                CategoricalMissingValueTransformer(),
            ),
            (
                "feature_engineering",
                FeatureEngineeringTransformer(),
            ),
        ]
    )

    # -------------------------------------------------------------
    # Column-wise preprocessing
    # -------------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                make_column_selector(dtype_include="object"),
            ),
        ]
    )

    # -------------------------------------------------------------
    # Complete Pipeline
    # -------------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "dataset_preprocessing",
                dataset_preprocessing_pipeline,
            ),
            (
                "column_transformer",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline