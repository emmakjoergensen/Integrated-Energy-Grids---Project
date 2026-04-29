import pandas as pd

def load_generation(path):
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(None)
    return df

def load_load(path):
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert(None)
    return df