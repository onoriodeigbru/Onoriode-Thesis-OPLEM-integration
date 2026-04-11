import pandas as pd

REQUIRED_COLUMNS = {
    "time": ["time", "hour"],
    "ghi": ["ghi", "GHI", "irradiance"],
    "temp_air": ["temp_air", "temperature", "T2m"],
    "wind_speed": ["wind_speed", "WS10m", "wind"]
}

def load_weather_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    for standard, aliases in REQUIRED_COLUMNS.items():
        for col in aliases:
            if col in df.columns:
                rename_map[col] = standard

    df = df.rename(columns=rename_map)

    missing = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[["time", "ghi", "temp_air", "wind_speed"]]

