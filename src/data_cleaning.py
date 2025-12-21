import re
import pandas as pd

def clean_dataframe(df):
    df = df.drop(['ids', 'date', 'flag', 'user'], axis=1)
    df['target'] = df['target'].replace(4, 1)
    df = df.drop_duplicates()
    df = df.dropna(subset=['text'])
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"(http\S+|www\S+)", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text
