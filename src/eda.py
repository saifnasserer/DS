def add_text_length(df):
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    return df
