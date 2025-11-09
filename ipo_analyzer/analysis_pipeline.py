import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from ipo_analyzer.config import TARGET_COLUMN, TEST_SET_SIZE, RANDOM_STATE, Columns
from ipo_analyzer.data_preprocessor import DataPreprocessor
from ipo_analyzer.feature_engineer import FeatureEngineer
from ipo_analyzer.model_trainer import ModelTrainer


class AnalysisPipeline:
    """
    Orchestrates the entire IPO analysis workflow from data loading to model interpretation.
    This class is designed to be called sequentially from a notebook.
    """

    def __init__(self, data_file_path: Optional[str] = None):
        self.data_preprocessor = DataPreprocessor(file_path=data_file_path)
        self.feature_engineer = FeatureEngineer()
        self.trainers: Dict[str, ModelTrainer] = {}

        # --- Data Attributes ---
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None
        self.X_train_val: Optional[pd.DataFrame] = None
        self.X_holdout_test: Optional[pd.DataFrame] = None
        self.y_train_val: Optional[pd.Series] = None
        self.y_holdout_test: Optional[pd.Series] = None

        # --- Results Attributes ---
        self.cv_results: Dict[str, float] = {}
        self.cv_results_sem: Dict[str, float] = {}
        self.holdout_reports: Dict[str, str] = {}
        self.holdout_matrices: Dict[str, np.ndarray] = {}
        self.feature_importances: Dict[str, pd.DataFrame] = {}

    # --- Sequential Execution Methods ---

    def load_and_clean_data(self):
        """Loads the raw data and runs the cleaning process."""
        print("--- Loading and Cleaning Data ---")
        self.raw_df = self.data_preprocessor.load_data()
        self.cleaned_data = self.data_preprocessor.run_cleaning(self.raw_df)
        self.final_df = self.cleaned_data.copy()  # Initialize final_df
        print("Data loading and cleaning complete.")

    def define_feature_types(self):
        """Defines feature types using the feature engineer."""
        if self.final_df is None:
            raise ValueError("Data not loaded. Please run load_and_clean_data() first.")
        print("\n--- Defining Feature Types ---")
        self.feature_engineer.define_feature_types(self.final_df)

    def create_new_features(self):
        """Creates new features using the feature engineer."""
        if self.final_df is None:
            raise ValueError("Data not loaded. Please run load_and_clean_data() first.")
        print("\n--- Creating New Features ---")
        self.final_df = self.feature_engineer.create_new_features(self.final_df)

    def apply_log_transform(self):
        """Applies log transform using the feature engineer."""
        if self.final_df is None:
            raise ValueError("Data not loaded. Please run load_and_clean_data() first.")
        print("\n--- Applying Log Transformations ---")
        self.final_df = self.feature_engineer.apply_log_transform(self.final_df)

    def split_data(self):
        """Splits the data into a training/validation set and a hold-out test set."""
        if self.final_df is None:
            raise ValueError("Feature engineering not complete. Please run feature engineering steps first.")

        print(f"\n--- Splitting Data into Train/Test Sets ---")
        X = self.final_df.drop(TARGET_COLUMN, axis=1)
        y = self.final_df[TARGET_COLUMN]
        self.X_train_val, self.X_holdout_test, self.y_train_val, self.y_holdout_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y)
        print(f"Training/Validation set size: {len(self.X_train_val)}")
        print(f"Hold-out Test set size: {len(self.X_holdout_test)}")

    def train_and_evaluate_single_model(self, model_name: str):
        """Trains and evaluates a single specified model."""
        if self.X_train_val is None:
            raise ValueError("Data not split. Please run split_data() first.")

        print(f"\n{'='*20} Training & Evaluating: {model_name.upper()} {'='*20}")
        trainer = ModelTrainer(model_name, self.feature_engineer.numerical_features, self.feature_engineer.categorical_features)
        self.trainers[model_name] = trainer

        # CV on the training/validation set
        trainer.run_cross_validation(self.X_train_val, self.y_train_val)
        self.cv_results[model_name] = trainer.mean_cv_auc
        self.cv_results_sem[model_name] = trainer.sem_cv_auc

        # Train final model on the full training/validation set
        trainer.train_model(self.X_train_val, self.y_train_val)

        # Evaluate on the hold-out test set
        report, matrix = trainer.evaluate_model(self.X_holdout_test, self.y_holdout_test)
        self.holdout_reports[model_name] = report
        self.holdout_matrices[model_name] = matrix

        # Generate feature importances
        trainer.generate_feature_importance(X_test_for_shap=self.X_holdout_test)
        self.feature_importances[model_name] = trainer.feature_importances
        print(f"--- Finished workflow for {model_name} ---")

    # --- Plotting and Helper Methods ---

    def get_missing_data_fraction(self) -> pd.DataFrame:
        """Calculates and returns the fraction of missing data."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Please run load_and_clean_data() first.")
        missing_fractions = self.raw_df.isnull().mean().sort_values(ascending=False)
        return pd.DataFrame(missing_fractions, columns=["missing_fraction"])

    def _plot_combined_roc_curve(self) -> None:
        """Plots the ROC curves for all trained models on the hold-out test set."""
        plt.figure(figsize=(10, 8))
        for model_name, trainer in self.trainers.items():
            y_pred_proba = trainer.pipeline.predict_proba(self.X_holdout_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_holdout_test, y_pred_proba)
            auc = roc_auc_score(self.y_holdout_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison on Hold-out Test Data")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_feature_importances(self) -> None:
        """Plots the normalized feature importances for all trained models."""
        num_models = len(self.feature_importances)
        fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 8), squeeze=False)
        fig.suptitle("Normalized Feature Importance Comparison", fontsize=16)

        for i, (model_name, importances) in enumerate(self.feature_importances.items()):
            ax = axes[0, i]
            if not importances.empty:
                top_15 = importances.head(15)
                sns.barplot(x="importance", y="feature", data=top_15, ax=ax)
                ax.set_title(f"Feature Importances for {model_name}")
                ax.set_xlabel("Normalized Importance")
                ax.set_ylabel("Feature")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
