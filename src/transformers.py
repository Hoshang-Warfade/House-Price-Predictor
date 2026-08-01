"""
Custom scikit-learn transformers for the House Price Prediction project.

These transformers perform:
- Numerical missing value imputation
- Categorical missing value imputation
- Feature engineering

All transformers are compatible with scikit-learn Pipelines.
"""

from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

FEATURE_ABSENCE_COLUMNS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]


# ---------------------------------------------------------------------
# Numerical Missing Value Transformer
# ---------------------------------------------------------------------

class NumericalMissingValueTransformer(BaseEstimator, TransformerMixin):
    """
    Impute missing numerical features.

    Operations
    ----------
    - MasVnrArea   -> 0
    - GarageYrBlt -> 0
    - LotFrontage -> Neighborhood-wise median
    """

    def fit(self, X, y=None):
        """
        Learn Neighborhood-wise median LotFrontage
        from the training data.
        """
        self.lotfrontage_median_ = (
            X.groupby("Neighborhood")["LotFrontage"]
            .median()
        )

        return self

    def transform(self, X):

        X = X.copy()

        # Houses without masonry veneer
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)

        # Houses without garages
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)

        # Impute LotFrontage using Neighborhood-wise median
        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.lotfrontage_median_)
        )

        return X


# ---------------------------------------------------------------------
# Categorical Missing Value Transformer
# ---------------------------------------------------------------------

class CategoricalMissingValueTransformer(BaseEstimator, TransformerMixin):
    """
    Impute missing categorical features.

    Operations
    ----------
    - Correct ambiguous basement records identified during EDA.
    - Replace feature absence with "None".
    - Impute Electrical using training-set mode.
    """

    def fit(self, X, y=None):
        """
        Learn categorical statistics from training data.
        """
        self.bsmtfintype2_mode_ = X["BsmtFinType2"].mode()[0]
        self.electrical_mode_ = X["Electrical"].mode()[0]

        return self

    def transform(self, X):

        X = X.copy()

        # -------------------------------------------------------------
        # Correct ambiguous basement records identified during EDA
        # -------------------------------------------------------------

        X.loc[
            X["BsmtExposure"].isna() & X["BsmtQual"].notna(),
            "BsmtExposure"
        ] = "No"

        X.loc[
            X["BsmtFinType2"].isna() & X["BsmtQual"].notna(),
            "BsmtFinType2"
        ] = self.bsmtfintype2_mode_

        # -------------------------------------------------------------
        # Missing value represents feature absence
        # -------------------------------------------------------------

        X[FEATURE_ABSENCE_COLUMNS] = (
            X[FEATURE_ABSENCE_COLUMNS]
            .fillna("None")
        )

        # Impute Electrical using training-set mode
        X["Electrical"] = X["Electrical"].fillna(
            self.electrical_mode_
        )

        return X


# ---------------------------------------------------------------------
# Feature Engineering Transformer
# ---------------------------------------------------------------------

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Create new features from existing attributes.

    Features Created
    ----------------
    - HouseAge
    - TotalIndoorArea
    - TotalBathrooms

    Feature Dropped
    ---------------
    - YrSold
    """

    def fit(self, X, y=None):
        """
        No parameters are learned.

        The fitted attribute is stored to satisfy
        scikit-learn's fitted-state validation.
        """
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):

        X = X.copy()

        # Age of the house at the time of sale
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        # Total finished living area
        X["TotalIndoorArea"] = (
            X["GrLivArea"]
            + X["TotalBsmtSF"]
        )

        # Total bathrooms (half bath counted as 0.5)
        X["TotalBathrooms"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"]
            + 0.5 * X["BsmtHalfBath"]
        )

        # Remove redundant feature after engineering
        X = X.drop(columns="YrSold")

        return X