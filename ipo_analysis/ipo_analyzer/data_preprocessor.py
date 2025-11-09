import pandas as pd
from typing import Optional

from ipo_analyzer.config import Columns, TARGET_COLUMN


class DataPreprocessor:
    """
    Handles loading and initial cleaning of the IPO data.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataPreprocessor.

        Args:
            file_path (Optional[str]): The path to the data file. If None, uses default.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the IPO data from the CSV file into a pandas DataFrame.
        """
        try:
            print(f"Loading data from: {self.file_path}")
            data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file was not found at {self.file_path}")

        print("Data loaded successfully.")
        return data

    def run_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial data cleaning.
        - Drops rows where the target variable is missing.
        """
        initial_rows = len(df)
        cleaned_df = df.dropna(subset=[TARGET_COLUMN]).copy()
        rows_dropped = initial_rows - len(cleaned_df)

        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing target variable ('{TARGET_COLUMN}').")

        return cleaned_df

    def run(self) -> pd.DataFrame:
        """
        Executes the full preprocessing workflow: load and clean.
        """
        raw_data = self.load_data()
        cleaned_data = self.run_cleaning(raw_data)
        return cleaned_data
