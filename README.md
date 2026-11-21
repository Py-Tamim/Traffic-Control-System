# Dhaka Traffic Prediction System 🚦

A Machine Learning-based traffic prediction system designed to analyze and forecast traffic conditions across major roads in Dhaka, Bangladesh. The system uses **Random Forest Classification and Regression** to predict traffic congestion levels and estimate important traffic parameters based on historical traffic patterns, road corridors, dates, and time of day.
Users can enter a specific date, a date range, a single hour, or an hourly range to generate traffic predictions for all available road corridors.

## 📌 Project Overview

Traffic congestion is a major problem in Dhaka, Bangladesh. This project aims to provide a data-driven approach to understanding and predicting traffic conditions across selected major road corridors.

The system analyzes historical traffic data and predicts:

- Traffic congestion levels.
- Traffic flow (vehicles per hour).
- Traffic density (vehicles per kilometer).
- Average vehicle speed (km/h).
- Volume-to-capacity ratio.
- Vehicle composition percentages.
- CAV V2X penetration rate.
- Overall traffic severity.

It also compares traffic conditions across different corridors to suggest the road with the lowest predicted congestion.

## ✨ Key Features

- **Traffic Classification:** Classifies traffic into six categories, from No Traffic to Extreme Traffic.
- **Traffic Parameter Prediction:** Predicts multiple numerical traffic indicators using machine learning regression.
- **Date-Based Prediction:** Accepts individual dates and date ranges.
- **Hourly Prediction:** Supports individual hours and hourly ranges in both 12-hour and 24-hour formats.
- **Multiple Corridor Analysis:** Predicts traffic conditions across all available road corridors.
- **Corridor Recommendation:** Identifies the corridor with the lowest predicted traffic severity.
- **Model Evaluation:** Evaluates classification accuracy and regression performance using unseen test data.
- **Data Visualization:** Generates multiple line graphs to compare predicted traffic conditions.
- **CSV Export:** Saves detailed prediction results into `predicted_traffic.csv`.

## 🛣️ Supported Road Corridors

The current dataset includes five major road corridors:

1. Airport Road – Uttara
2. Gulshan Avenue
3. Kazi Nazrul Islam Avenue
4. Mirpur Road
5. VIP Road – Tejgaon

## 🤖 Machine Learning Models

This project uses two Random Forest models.

### 1. Random Forest Classifier

The classification model predicts the traffic congestion category.

Traffic categories are mapped as follows:

| Code | Traffic Level | Severity |
|---|---|---:|
| A | No traffic | 0 |
| B | Light traffic | 1 |
| C | Moderate traffic | 2 |
| D | Heavy traffic | 3 |
| E | Very heavy traffic | 4 |
| F | Extreme traffic | 5 |

**Model configuration:**
- Algorithm: RandomForestClassifier
- Number of estimators: 200
- Class weighting: Balanced
- Minimum samples per leaf: 10
- Random state: 42

### 2. Random Forest Regressor

The regression model predicts multiple numerical traffic parameters simultaneously.

**Model configuration:**
- Algorithm: RandomForestRegressor
- Number of estimators: 200
- Minimum samples per leaf: 10
- Random state: 42

### Feature Engineering

The system uses the following input features:

| Feature | Description |
|---|---|
| corridor_name | Selected road corridor |
| hour_of_day | Hour of prediction |
| day_of_week | Day of the week |
| is_weekend | Weekend indicator |
| sin_time | Sine transformation of time |
| cos_time | Cosine transformation of time |

Sine and cosine transformations represent time cyclically. This helps the model understand that 23:00 and 00:00 are consecutive hours rather than completely different time values.

### Predicted Parameters

| Parameter | Description |
|---|---|
| flow_q_vph | Traffic flow (vehicles/hour) |
| density_k_vpk | Traffic density (vehicles/km) |
| speed_v_kmh | Average vehicle speed (km/h) |
| vc_ratio | Volume-to-capacity ratio |
| non_motorized_pct | Non-motorized vehicle percentage |
| cng_three_wheeler_pct | CNG three-wheeler percentage |
| bus_minibus_pct | Bus and minibus percentage |
| private_car_suv_pct | Private car and SUV percentage |
| motorcycle_pct | Motorcycle percentage |
| cav_v2x_penetration_rate | CAV V2X penetration rate |

## 📊 Model Evaluation

The dataset is divided into training and testing sets:

- Training data: 80%
- Testing data: 20%

The models are evaluated using:

**Classification:**
- Accuracy Score

**Regression:**
- R² Score

Example model evaluation from a test run:

| Evaluation Metric | Result |
|---|---:|
| Traffic Classification Accuracy | 54% |

*Note: This is an example result from one execution. Actual performance depends on the dataset and training configuration.*

## 📈 Data Visualization

The project generates multiple line graphs to visualize predicted traffic conditions.

The graphs include:

- Time-based traffic flow.
- Traffic density.
- Average vehicle speed.
- Volume-to-capacity ratio.
- Vehicle composition.
- CAV V2X penetration rate.
- Overall traffic severity.

Each graph compares the available road corridors, making it easier to identify differences in predicted traffic conditions.

## 🖥️ Sample Output

**Example Input:**

```text
Enter a date or date range:
16-01-2027

Enter prediction hour or hour range:
12 to 17
```

**Example Prediction:**

```text
Traffic predictions for 16 January 2027 at 12:00

        corridor_name          traffic
        Gulshan_Avenue          Moderate traffic
        Kazi_Nazrul_Islam_Ave   Moderate traffic
        Mirpur_Road             Light traffic
        Airport_Road_Uttara     Light traffic
        VIP_Road_Tejgaon        Light traffic

Suggested corridor: VIP_Road_Tejgaon (Light traffic)
```

The system repeats the prediction for each requested hour.

**Example Model Evaluation:**

```text
Model check on unseen data:
traffic-category accuracy = 54%
```

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Classification
- Random Forest Regression

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Prepare the Dataset

Place the dataset file in the same directory as the Python script.

```text
traffic_prediction/
│
├── traffic_prediction.py
├── dhaka_traffic_dataset.csv
└── README.md
```

### 5. Run the Program

```bash
python traffic_prediction.py
```

## 📝 Input Examples

The system supports several input formats.

**Date input:**

```text
16th January, 2027
16-11-2026
16/11/2026
2027-01-16
```

**Date range:**

```text
16-11-2026 to 16-02-2027
```

**Hour input:**

```text
7
7 am
7 pm
19:00
```

**Hour range:**

```text
7 to 11
7 am to 11 am
7-11
14 to 18
```

If no hour is provided, the system uses 12:00 PM by default.

## 📁 Project Structure

```text
Dhaka-Traffic-Prediction/
│
├── traffic_prediction.py
├── dhaka_traffic_dataset.csv
├── predicted_traffic.csv
├── README.md
└── requirements.txt
```

`predicted_traffic.csv` is automatically generated after running the program.

## ⚠️ Limitations

- Predictions are based on historical dataset patterns.
- The current model does not use live GPS, real-time traffic feeds, or live road conditions.
- Weather information is not included in the current prediction model.
- Prediction accuracy depends on the quality, quantity, and representativeness of historical data.
- The suggested corridor is based on predicted traffic severity, not actual travel distance or navigation routes.
- Future predictions should be treated as estimates rather than guaranteed traffic conditions.

## 🚀 Future Improvements

Potential improvements include:

- Integrating real-time traffic information through APIs.
- Adding weather conditions as prediction features.
- Exploring XGBoost and other advanced machine learning algorithms.
- Implementing LSTM for time-series traffic forecasting.
- Improving model accuracy through additional historical traffic data.
- Developing an interactive dashboard using Streamlit.
- Adding geographical road mapping and route optimization.
