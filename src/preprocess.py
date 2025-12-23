import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional, Union

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


def load_data(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


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


def preprocess_text(text: Union[str, float]) -> str:
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    words = text.split()
    words = [_lemmatizer.lemmatize(w) for w in words if w.lower() not in _stop_words and len(w) > 1]
    return " ".join(words)


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

