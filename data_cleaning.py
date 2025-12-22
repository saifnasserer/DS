import re
import pandas as pd
from typing import Union


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if 'text' not in df.columns or 'target' not in df.columns:
        raise KeyError("DataFrame must contain 'text' and 'target' columns")
    
    df = df[['text', 'target']].copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=['text'])
    return df


def clean_text(text: Union[str, float]) -> str:
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"(http\S+|www\S+)", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
