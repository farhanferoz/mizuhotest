import pandas as pd
import numpy as np
import shap
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ipo_analyzer.config import RANDOM_STATE, CV_SPLITS, SHAP_SUMMARY_PLOT_PATH, NumericFeatures, CategoricalFeatures, Models


class ModelTrainer:
    """
    Handles the model training, validation, and interpretation pipeline for a given model.
    """

    def __init__(self, model_name: str, numerical_features: NumericFeatures, categorical_features: CategoricalFeatures):
        self.model_name = model_name
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.pipeline: Pipeline = self._create_pipeline()
        self.mean_cv_auc: float = 0.0
        self.sem_cv_auc: float = 0.0  # Standard Error of the Mean
        self.feature_importances: pd.DataFrame = pd.DataFrame()

    def _create_pipeline(self) -> Pipeline:
        """
        Creates the scikit-learn preprocessing and modeling pipeline based on the model name.
        """
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, self.numerical_features), ("cat", categorical_transformer, self.categorical_features)], remainder="passthrough"
        )

        if self.model_name == Models.LOGISTIC:
            classifier = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
        elif self.model_name == Models.XGB:
            classifier = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=10, learning_rate=0.3, max_depth=6)  # Removed use_label_encoder=False
        else:
            raise ValueError(f"Unknown model_name: {self.model_name.name}")

        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        return model_pipeline

    def run_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Performs stratified k-fold cross-validation and calculates the mean ROC AUC score and its standard error.
        """
        print(f"--- Running Stratified K-Fold CV for {self.model_name} ---")
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = []

        for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.pipeline.fit(X_train, y_train)
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)

        self.mean_cv_auc = np.mean(auc_scores)
        self.sem_cv_auc = np.std(auc_scores) / np.sqrt(len(auc_scores))
        print(f"Mean ROC AUC: {self.mean_cv_auc:.4f} (SEM: {self.sem_cv_auc:.4f})")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the final model on the provided training data.
        """
        print(f"--- Training final {self.model_name} model ---")
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, np.ndarray]:
        """
        Evaluates the trained model on the hold-out test set.
        """
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        return report, matrix

    def generate_feature_importance(self, X_test_for_shap: Optional[pd.DataFrame] = None) -> None:
        """
        Generates and stores normalized feature importances based on the model type.
        """
        print(f"--- Generating Feature Importance for {self.model_name} ---")

        try:
            ohe_cols = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(self.categorical_features)
            final_feature_names = self.numerical_features + list(ohe_cols)
        except Exception:
            print("Could not retrieve feature names.")
            return

        if self.model_name == Models.XGB:
            if X_test_for_shap is None:
                print("X_test data is required for SHAP analysis.")
                return
            explainer = shap.TreeExplainer(self.pipeline.named_steps["classifier"])
            X_test_transformed = self.pipeline.named_steps["preprocessor"].transform(X_test_for_shap)
            shap_values = explainer.shap_values(X_test_transformed)
            importances = np.abs(shap_values).mean(0)
        elif self.model_name == Models.LOGISTIC:
            coeffs = self.pipeline.named_steps["classifier"].coef_[0]
            importances = np.abs(coeffs)

        # Normalize the importances to sum to 1
        normalized_importances = importances / np.sum(importances)

        self.feature_importances = pd.DataFrame(list(zip(final_feature_names, normalized_importances)), columns=["feature", "importance"]).sort_values(
            by="importance", ascending=False
        )
