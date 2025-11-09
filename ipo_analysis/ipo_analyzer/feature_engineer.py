import pandas as pd
import numpy as np
from typing import List

from ipo_analyzer.config import Columns, SKEWNESS_THRESHOLD, TARGET_COLUMN, NumericFeatures, CategoricalFeatures


class FeatureEngineer:
    """
    Handles the feature engineering process for the IPO dataset.
    Each step is a public method to be called sequentially.
    """

    def __init__(self):
        self.numerical_features: NumericFeatures = []
        self.categorical_features: CategoricalFeatures = []
        self.highly_skewed_features: List[str] = []

    def define_feature_types(self, df: pd.DataFrame) -> None:
        """
        Manually defines which features are categorical and which are numerical.
        """
        df.columns = [str(col) for col in df.columns]

        # Manually define categorical features
        self.categorical_features = [Columns.INDUSTRY_SECTOR.value, Columns.VENTURE_BACKED.value, Columns.IPO_MONTH.value, Columns.LOCKUP_PERIOD_DAYS.value]

        # Define numerical features as all other columns that are not the target or categorical
        self.numerical_features = [col for col in df.select_dtypes(include=np.number).columns if col not in self.categorical_features and col != TARGET_COLUMN]

        print(f"Manually defined categorical features: {self.categorical_features}")
        print(f"Remaining numerical features: {self.numerical_features}")

    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features. Does not impute, allows NaNs to propagate.
        """
        if not self.numerical_features:
            raise ValueError("Feature types not defined. Run define_feature_types() first.")

        print("Creating new features: [ipo_size_M, revenue_per_employee]")

        # ipo_size_M
        offer_price_str = str(Columns.OFFER_PRICE)
        shares_str = str(Columns.SHARES_OFFERED_MILLIONS)
        ipo_size_str = str(Columns.IPO_SIZE_M)

        df[ipo_size_str] = df[offer_price_str] * df[shares_str]

        # revenue_per_employee
        revenue_millions_str = str(Columns.REVENUE_MILLIONS)
        employees_str = str(Columns.EMPLOYEES)
        revenue_per_employee_str = "revenue_per_employee"

        # Replace 0 with NaN before division to avoid infinity, then divide.
        df[revenue_per_employee_str] = df[revenue_millions_str] / df[employees_str].replace(0, np.nan)

        # Add new features to the numerical features list
        for new_feat in [ipo_size_str, revenue_per_employee_str]:
            if new_feat not in self.numerical_features:
                self.numerical_features.append(new_feat)

        print(f"New features created. Numerical features are now: {self.numerical_features}")
        return df

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log1p transformation to highly skewed numerical features.
        """
        if not self.numerical_features:
            raise ValueError("Feature types not defined. Run define_feature_types() first.")

        df.columns = [str(col) for col in df.columns]

        # Calculate skew only on the defined numerical features
        skewed_feats = df[self.numerical_features].skew()
        self.highly_skewed_features = list(skewed_feats[abs(skewed_feats) > SKEWNESS_THRESHOLD].index)

        if self.highly_skewed_features:
            print(f"Applying log transform to highly skewed features: {self.highly_skewed_features}")
            for col in self.highly_skewed_features:
                df[col] = np.log1p(df[col])
        else:
            print("No highly skewed features detected to transform.")

        return df
