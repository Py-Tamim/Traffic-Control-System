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
    date_formats = [
        "%d %B %Y", "%d %b %Y", "%d-%m-%Y", "%d/%m/%Y",
        "%d-%m-%y", "%d/%m/%y", "%Y-%m-%d",
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
    raise ValueError(
        "Could not read that date. Try 16th January, 2027, "
        "18-11-2026, or 18/11/26."
    )

def get_prediction_dates(date_input):
    date_parts = re.split(r"\s+to\s+", date_input.strip(), flags=re.IGNORECASE)
    start_date = parse_date(date_parts[0])
    if len(date_parts) == 1:
        return [pd.Timestamp(start_date)]

    # Parse the end of the range and reject reversed ranges.
    if len(date_parts) != 2:
        raise ValueError("Enter one date or a range in the format date to date.")
    end_date = parse_date(date_parts[1])
    if end_date < start_date:
        raise ValueError("The end date must be the same as or later than the start date.")

    return list(pd.date_range(start=start_date, end=end_date, freq="D"))

def parse_single_hour(hour_text):
    match = re.fullmatch(r"(\d{1,2})(?::\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?", hour_text.strip().lower())
    if not match:
        raise ValueError("Could not read the hour. Try 7, 7 am, or 7 am to 11 am.")
    hour = int(match.group(1))
    period = match.group(2)
    if period:
        if not 1 <= hour <= 12:
            raise ValueError("With am/pm, the hour must be between 1 and 12.")
        is_pm = period.startswith("p")
        hour = hour % 12 + (12 if is_pm else 0)
    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")
    return hour

def get_prediction_hours(hour_input):                    # Prediction
    if not hour_input.strip():
        return [12]
    parts = re.split(r"\s+to\s+|\s*[-–]\s*", hour_input.strip(), flags=re.IGNORECASE)

    if len(parts) == 1:
        return [parse_single_hour(parts[0])]
    if len(parts) != 2:
        raise ValueError("Enter one hour or a range such as 7 am to 11 am.")
    start_text, end_text = parts
    suffix = re.search(r"(am|pm|a\.m\.|p\.m\.)$", end_text.strip().lower())
    if suffix and not re.search(r"(am|pm|a\.m\.|p\.m\.)$", start_text.strip().lower()):
        start_text = f"{start_text} {suffix.group(1)}"
    start_hour = parse_single_hour(start_text)
    end_hour = parse_single_hour(end_text)
    if end_hour < start_hour:
        raise ValueError("The end hour must be the same as or later than the start hour.")
    
    return list(range(start_hour, end_hour + 1))

def predict_all(dates, hours):
    rows = []
    for date in dates:
        for hour in hours:
            for corridor in corridors:
                rows.append(
                    {
                        "date": date,
                        "corridor_name": corridor,
                        "hour_of_day": hour,
                        "day_of_week": date.dayofweek,
                        "is_weekend": int(date.dayofweek >= 5),
                        # Same formula as the CSV: the hour as an angle on a 24h circle.
                        "sin_time": round(float(np.sin(2 * np.pi * hour / 24)), 5),
                        "cos_time": round(float(np.cos(2 * np.pi * hour / 24)), 5),
                    }
                )
    result = pd.DataFrame(rows)

    numeric_predictions = pd.DataFrame(
        regressor.predict(result[features]), columns=target_columns, index=result.index
    )
    result = pd.concat([result, numeric_predictions.round(3)], axis=1)
    probabilities = classifier.predict_proba(result[features])
    classes = classifier.named_steps["model"].classes_
    result["traffic"] = classifier.predict(result[features])
    result["severity"] = [
        sum(severity_order[label] * p for label, p in zip(classes, row))
        for row in probabilities
    ]
    return result

def plot_predictions(plot_df, x_column, x_label, title, marker_hour=None):     # Graphical representation of the prediction result.
    panels = time_columns + target_columns + ["severity"]
    fig, axes = plt.subplots(5, 3, figsize=(18, 20), sharex=True)
    fig.suptitle(title, fontsize=15)

    for axis, column in zip(axes.flat, panels):
        if column in time_columns:
            single = plot_df[plot_df["corridor_name"] == corridors[0]]
            axis.plot(single[x_column], single[column], color="black", marker="o", markersize=3)
        else:
            for corridor in corridors:
                part = plot_df[plot_df["corridor_name"] == corridor]
                axis.plot(part[x_column], part[column], marker="o", markersize=3, label=corridor)

        if marker_hour is not None:
            axis.axvline(marker_hour, color="red", linestyle="--", alpha=0.6)

        axis.set_title(column_titles[column], fontsize=10)
        axis.grid(alpha=0.3)
        if x_column == "hour_of_day":
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    handles, labels = axes.flat[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=1, fontsize=11)
    for axis in axes[-1]:
        axis.set_xlabel(x_label)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main():               # MAin Program.
    date_input = input(
        "Enter a date or date range (example: 16th January, 2027 "
        "or 16-11-2026 to 16-02-2027): "
    )
    hour_input = input(
        "Enter prediction hour or hour range (example: 7 | 7 am | 7 am to 11 am | "
        "7-11 | 14 to 18) [12]: "
    )

    try:
        dates = get_prediction_dates(date_input)
        hours = get_prediction_hours(hour_input)
    except ValueError as error:
        print(error)
        return

    results = predict_all(dates, hours)
    results.drop(columns=["severity"]).to_csv("predicted_traffic.csv", index=False)

    show_columns = ["corridor_name", "traffic"] + time_columns + target_columns
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    if len(dates) * len(hours) <= 6:
        for date in dates:
            for hour in hours:
                subset = results[(results["date"] == date) & (results["hour_of_day"] == hour)]
                subset = subset.sort_values("severity", ascending=False)
                print(f"\nTraffic predictions for {date:%d %B %Y} at {hour:02d}:00")
                print(subset[show_columns].to_string(index=False))
                best = subset.iloc[-1]
                print(f"Suggested corridor: {best['corridor_name']} ({best['traffic']})")
    else:
        summary = results.groupby("corridor_name")[target_columns + ["severity"]].mean()
        most_common = results.groupby("corridor_name")["traffic"].agg(lambda s: s.mode().iloc[0])
        summary.insert(0, "most_common_traffic", most_common)
        summary = summary.sort_values("severity", ascending=False).round(2)
        print(f"\nAverage predictions over {len(dates)} day(s) and hours {hours[0]:02d}:00-{hours[-1]:02d}:00")
        print(summary.to_string())
        print(f"\nSuggested corridor: {summary.index[-1]} ({summary.iloc[-1]['most_common_traffic']})")
    print("\nFull detailed results were saved to predicted_traffic.csv")

    if len(dates) == 1 and len(hours) == 1:
        day = predict_all(dates, list(range(24)))
        plot_predictions(
            day, "hour_of_day", "Hour of day",
            f"Predicted traffic on {dates[0]:%d %B %Y} (red line = {hours[0]:02d}:00)",
            marker_hour=hours[0],
        )
    elif len(dates) == 1:
        plot_predictions(
            results, "hour_of_day", "Hour of day",
            f"Predicted traffic on {dates[0]:%d %B %Y}, {hours[0]:02d}:00-{hours[-1]:02d}:00",
        )
    else:
        daily = (
            results.groupby(["date", "corridor_name"])[time_columns + target_columns + ["severity"]]
            .mean()
            .reset_index()
        )
        plot_predictions(
            daily, "date", "Date",
            f"Predicted traffic {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y} "
            f"(average of {hours[0]:02d}:00-{hours[-1]:02d}:00)",
        )

if __name__ == "__main__":
    main()
