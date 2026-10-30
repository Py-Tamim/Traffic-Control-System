import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("dhaka_traffic_dataset.csv")

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(exclude="number").columns:
    most_common = df[col].mode()
    replacement = most_common.iloc[0] if not most_common.empty else "Unknown"
    df[col] = df[col].fillna(replacement)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour_of_day"] = df["hour_of_day"].fillna(df["timestamp"].dt.hour).fillna(12)
df["day_of_week"] = df["day_of_week"].fillna(df["timestamp"].dt.dayofweek).fillna(0)
traffic_labels = {
    "A": "No traffic",
    "B": "Light traffic",
    "C": "Moderate traffic",
    "D": "Heavy traffic",
    "E": "Very heavy traffic",
    "F": "Extreme traffic",
}

df["los_category"] = df["los_category"].astype(str).str.strip().str.upper().str[0]
df["traffic_level"] = df["los_category"].map(traffic_labels)
df = df.dropna(subset=["corridor_name", "traffic_level"]).copy()

features = [
    "corridor_name",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "weather_condition",
    "special_event_flag",
]

numeric_features = ["hour_of_day", "day_of_week", "is_weekend"] # Separate numeric fields from text/category fields for preprocessing.
categorical_features = [
    "corridor_name",
    "weather_condition",
    "special_event_flag",
]

