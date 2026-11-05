import re
from datetime import datetime
from tabnanny import check
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("dhaka_traffic_dataset.csv") # Load Data

for column in df.select_dtypes(include="number").columns:     # Fill Missing Values
    df[column] = df[column].fillna(df[column].median())

for column in df.select_dtypes(exclude="number").columns:
    most_common = df[column].mode()
    df[column] = df[column].fillna(
        most_common.iloc[0] if not most_common.empty else "Unknown"
    )

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

severity_order = {
    "No traffic": 0,
    "Light traffic": 1,
    "Moderate traffic": 2,
    "Heavy traffic": 3,
    "Very heavy traffic": 4,
    "Extreme traffic": 5,
}
#__________Train the Model__________
numeric_features = ["hour_of_day", "day_of_week", "is_weekend", "sin_time", "cos_time"]
categorical_features = ["corridor_name"]
features = categorical_features + numeric_features
time_columns = ["sin_time", "cos_time"]
target_columns = [
    "flow_q_vph",
    "density_k_vpk",
    "speed_v_kmh",
    "vc_ratio",
    "non_motorized_pct",
    "cng_three_wheeler_pct",
    "bus_minibus_pct",
    "private_car_suv_pct",
    "motorcycle_pct",
    "cav_v2x_penetration_rate",
]
column_titles = {
    "sin_time": "sin_time",
    "cos_time": "cos_time",
    "flow_q_vph": "Flow (vehicles/hour)",
    "density_k_vpk": "Density (vehicles/km)",
    "speed_v_kmh": "Speed (km/h)",
    "vc_ratio": "Volume/Capacity ratio",
    "non_motorized_pct": "Non-motorized (%)",
    "cng_three_wheeler_pct": "CNG three-wheeler (%)",
    "bus_minibus_pct": "Bus / minibus (%)",
    "private_car_suv_pct": "Private car / SUV (%)",
    "motorcycle_pct": "Motorcycle (%)",
    "cav_v2x_penetration_rate": "CAV V2X penetration rate",
    "severity": "Traffic severity (0-5)",
}

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numbers", SimpleImputer(strategy="median"), numeric_features),
            (
                "categories",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

classifier = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced", min_samples_leaf=10, n_jobs=-1
            ),
        ),
    ]
)
regressor = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        (
            "model",
            RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=10, n_jobs=-1),
        ),
    ]
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
classifier.fit(train_df[features], train_df["traffic_level"])
regressor.fit(train_df[features], train_df[target_columns])


accuracy = accuracy_score(                                    # Check model accuracy on unseen data (Comment this part if not needed). 
    test_df["traffic_level"], classifier.predict(test_df[features])
)
r2 = r2_score(test_df[target_columns], regressor.predict(test_df[features]))
print(f"Model check on unseen data: traffic-category accuracy = {accuracy:.0%}")  # (fOR r2 score(, "f"numeric-column R2 = {r2:.2f}))
classifier.fit(df[features], df["traffic_level"])                  # Retraon model for final prediction. 
regressor.fit(df[features], df[target_columns])
corridors = sorted(df["corridor_name"].astype(str).unique())         #Part end

def parse_date(date_text):
    cleaned = date_text.strip().lower()
    cleaned = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.replace(",", " ")).strip()

    # Try supported day-first and ISO date formats.
    date_formats = [
        "%d %B %Y", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y",
        "%d-%m-%y", "%d/%m/%y", "%Y-%m-%d",
    ]

    # Return the first format that matches the user's date.
    for date_format in date_formats:
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue

    # Tell the user which input styles are supported if none matched.
    raise ValueError(
        "Could not read that date. Try 16th January, 2027, "
        "18-11-2026, or 18/11/26."
    )

#=============================================== Will be update soon =========================================