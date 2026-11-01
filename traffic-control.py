import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("dhaka_traffic_dataset.csv")   # For load the CSV file into a pandas DataFrame.
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

preprocessor = ColumnTransformer(
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
# Preprocessing 
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

model.fit(df[features], df["traffic_level"])
corridors = sorted(df["corridor_name"].astype(str).unique())
weather_values = sorted(df["weather_condition"].astype(str).unique())
weather_default = (
    df["weather_condition"].mode().iloc[0]
    if not df["weather_condition"].mode().empty
    else "Unknown"
)
df.groupby("corridor_name")["speed_v_kmh"].mean().sort_values().plot(
    kind="barh",
    title="Historical Average Speed by Corridor",
    xlabel="Average speed (km/h)",
)
plt.tight_layout()
plt.show()
def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

def match_corridors(text):
    query = normalize(text)
    exact_matches = [
        corridor
        for corridor in corridors
        if normalize(corridor).replace("_", " ") in query
    ]
    if exact_matches:
        return exact_matches

    query_words = set(query.split())
    scored_matches = []
    for corridor in corridors:
        corridor_words = set(normalize(corridor).replace("_", " ").split())
        shared_word_count = len(query_words & corridor_words)
        if shared_word_count:
            scored_matches.append((shared_word_count, corridor))
    scored_matches.sort(reverse=True)
    return [scored_matches[0][1]] if scored_matches else []

def predict_corridors(candidate_corridors, hour, weekday, weather):
    prediction_rows = pd.DataFrame(
        [
            {
                "corridor_name": corridor,
                "hour_of_day": hour,
                "day_of_week": weekday,
                "is_weekend": int(weekday >= 5),
                "weather_condition": weather,
                "special_event_flag": False,
            }
            for corridor in candidate_corridors
        ]
    )
    probabilities = model.predict_proba(prediction_rows[features])
    traffic_classes = model.named_steps["classifier"].classes_
    severity_order = {
        "No traffic": 0,
        "Light traffic": 1,
        "Moderate traffic": 2,
        "Heavy traffic": 3,
        "Very heavy traffic": 4,
        "Extreme traffic": 5,
    }
    results = []
    for corridor, class_probabilities in zip(candidate_corridors, probabilities):
        severity_score = sum(
            severity_order[label] * probability
            for label, probability in zip(traffic_classes, class_probabilities)
        )
        median_speed = df.loc[
            df["corridor_name"] == corridor, "speed_v_kmh"
        ].median()
        results.append((corridor, severity_score, median_speed))
    return sorted(results, key=lambda result: (result[1], -result[2]))

