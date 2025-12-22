import pandas as pd
from typing import Optional


def load_data(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def extract_volume_data(df: pd.DataFrame, keyword: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    
    if 'keyword' in df.columns and keyword:
        df = df[df['keyword'].str.contains(keyword, case=False, na=False)]
    
    if 'id' in df.columns:
        df['date'] = pd.to_datetime(df['id'], errors='coerce', origin='2020-01-01', unit='D')
    else:
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    daily_volume = df.groupby(df['date'].dt.date).size().reset_index()
    daily_volume.columns = ['date', 'volume']
    daily_volume['date'] = pd.to_datetime(daily_volume['date'])
    
    return daily_volume
