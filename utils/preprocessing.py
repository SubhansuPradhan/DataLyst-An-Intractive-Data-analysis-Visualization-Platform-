"""
Preprocessing helpers used across the app.
Provides safe cleaning and light encoding utilities.
"""
from typing import List, Tuple

import pandas as pd


def clean_missing_values(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """
    method: "drop" | "mean" | "zero"
    """
    if df is None:
        return df
    if method == "drop":
        return df.dropna()
    if method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if method == "zero":
        return df.fillna(0)
    return df

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return DataFrame with selected columns (if exist)."""
    if df is None:
        return df
    cols = [c for c in columns if c in df.columns]
    return df[cols]

def get_numeric_and_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols)."""
    if df is None:
        return [], []
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric, cat

def safe_to_datetime(df: pd.DataFrame, cols: List[str]):
    """Try to convert provided columns to datetime in-place (silently skip)."""
    for c in cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="raise")
        except Exception:
            pass
    return df
