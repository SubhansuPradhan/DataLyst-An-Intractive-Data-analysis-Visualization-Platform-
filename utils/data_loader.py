"""
Simple data loading helpers.
"""
import pandas as pd
from typing import Tuple, Optional

def load_dataframe_from_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel into a DataFrame. Returns None if no file.
    """
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def limit_rows(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    """Limit DataFrame size for UI responsiveness."""
    if df is None:
        return df
    if len(df) > max_rows:
        return df.head(max_rows)
    return df
