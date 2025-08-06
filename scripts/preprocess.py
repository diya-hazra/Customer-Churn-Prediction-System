import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df
