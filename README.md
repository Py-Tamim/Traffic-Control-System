# Dhaka Traffic Prediction System 🚦

A Machine Learning-based traffic prediction system designed to analyze and forecast traffic conditions across major road corridors in Dhaka, Bangladesh.

The system uses **Random Forest Classification and Regression** to predict traffic congestion levels and estimate important traffic parameters based on historical traffic patterns, road corridors, dates, and time of day.

Users can enter a specific date, a date range, a single hour, or an hourly range to generate traffic predictions across all available road corridors.

## 📌 Project Overview

Traffic congestion is one of the major transportation challenges in Dhaka, Bangladesh. This project aims to explore how machine learning can be used to understand historical traffic patterns and estimate future traffic conditions.

The system analyzes historical traffic data to predict several important traffic indicators, including:

- Traffic congestion levels.
- Traffic flow (vehicles per hour).
- Traffic density (vehicles per kilometer).
- Average vehicle speed (km/h).
- Volume-to-capacity ratio.
- Vehicle composition percentages.
- CAV V2X penetration rate.
- Overall traffic severity.

In addition to predicting traffic conditions, the system compares multiple road corridors and identifies the corridor with the lowest predicted traffic severity for the selected date and time.

## ✨ Key Features

- **Traffic Classification:** Classifies traffic conditions into six categories, ranging from No Traffic to Extreme Traffic.
- **Traffic Parameter Prediction:** Estimates multiple numerical traffic indicators using Random Forest Regression.
- **Date-Based Prediction:** Supports individual dates and date ranges.
- **Hourly Prediction:** Accepts individual hours and hourly ranges in both 12-hour and 24-hour formats.
- **Multiple Corridor Analysis:** Generates predictions for all available road corridors.
- **Corridor Recommendation:** Identifies the road corridor with the lowest predicted traffic severity.
- **Model Evaluation:** Evaluates model performance using unseen test data.
- **Data Visualization:** Generates multiple line graphs to compare predicted traffic conditions across different corridors.
- **CSV Export:** Automatically saves detailed prediction results into `predicted_traffic.csv`.

## 🛣️ Supported Road Corridors

The current dataset includes five major road corridors in Dhaka:

1. Airport Road – Uttara
2. Gulshan Avenue
3. Kazi Nazrul Islam Avenue
4. Mirpur Road
5. VIP Road – Tejgaon

## 🤖 Machine Learning Models

This project uses two Random Forest models to handle traffic classification and numerical traffic prediction.

### 1. Random Forest Classifier

The classification model predicts the traffic congestion category based on the selected road corridor, date, and time-related features.

The system uses six traffic categories:

| Code | Traffic Level | Severity |
|---|---|---:|
| A | No Traffic | 0 |
| B | Light Traffic | 1 |
| C | Moderate Traffic | 2 |
| D | Heavy Traffic | 3 |
| E | Very Heavy Traffic | 4 |
| F | Extreme Traffic | 5 |

**Model Configuration:**

- Algorithm: `RandomForestClassifier`
- Number of estimators: 200
- Class weighting: Balanced
- Minimum samples per leaf: 10
- Random state: 42

### 2. Random Forest Regressor

The regression model predicts multiple numerical traffic parameters simultaneously.

These predictions provide a more detailed understanding of traffic conditions beyond the congestion category.

**Model Configuration:**

- Algorithm: `RandomForestRegressor`
- Number of estimators: 200
- Minimum samples per leaf: 10
- Random state: 42

### Feature Engineering

The models use the following input features:

| Feature | Description |
|---|---|
| corridor_name | Selected road corridor |
| hour_of_day | Hour of prediction |
| day_of_week | Day of the week |
| is_weekend | Weekend indicator |
| sin_time | Sine transformation of time |
| cos_time | Cosine transformation of time |

Sine and cosine transformations are used to represent time as a cyclical feature.

For example, 23:00 and 00:00 are consecutive hours. Using cyclical transformations helps the models understand this relationship instead of treating them as completely different numerical values.

### Predicted Traffic Parameters

The regression model estimates the following parameters:

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

The dataset is divided into training and testing sets to evaluate how well the models perform on unseen data.

**Dataset Split:**

- Training data: 80%
- Testing data: 20%

**Evaluation Metrics:**

Classification:
- Accuracy Score

Regression:
- R² Score

### Example Model Evaluation

| Evaluation Metric | Result |
|---|---:|
| Traffic Classification Accuracy | 54% |

*Note: The result above is an example from one execution. Actual performance may vary depending on the dataset, training configuration, and data quality.*

## 📈 Data Visualization

The project generates multiple line graphs to visualize and compare predicted traffic conditions across the supported road corridors.

The visualizations include:

- Traffic flow over time.
- Traffic density.
- Average vehicle speed.
- Volume-to-capacity ratio.
- Vehicle composition percentages.
- CAV V2X penetration rate.
- Overall traffic severity.

These graphs help identify traffic patterns and differences between road corridors at different times of the day.

## 🖥️ Sample Output

### Example Input

```text
Enter a date or date range:
16-01-2027

Enter prediction hour or hour range:
12 to 17
```

### Example Prediction

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

The system repeats the prediction for every requested hour.

### Example Model Evaluation

```text
Model check on unseen data:
traffic-category accuracy = 54%
```

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Classification Algorithm:** Random Forest Classifier
- **Regression Algorithm:** Random Forest Regressor

## 📦 Installation and Setup

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

Alternatively, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the dataset file in the same directory as the Python script.

```text
traffic_prediction/
│
├── traffic_prediction.py
├── dhaka_traffic_dataset.csv
├── README.md
└── requirements.txt
```

### 5. Run the Program

```bash
python traffic_prediction.py
```

## 📝 Supported Input Formats

The system supports different date and time input formats.

### Date Input

```text
16th January, 2027
16-11-2026
16/11/2026
2027-01-16
```

### Date Range

```text
16-11-2026 to 16-02-2027
```

### Hour Input

```text
7
7 am
7 pm
19:00
```

### Hour Range

```text
7 to 11
7 am to 11 am
7-11
14 to 18
```

If no hour is provided, the system uses 12:00 PM as the default prediction time.

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

**File Description:**

- `traffic_prediction.py` – Main Python script containing data processing, model training, prediction, evaluation, and visualization.
- `dhaka_traffic_dataset.csv` – Historical traffic dataset used for model training and testing.
- `predicted_traffic.csv` – Automatically generated file containing prediction results.
- `README.md` – Project documentation.
- `requirements.txt` – Required Python libraries.


## 🤝 AI Assistance and Acknowledgement

During the development of this project, I used AI tools as a supporting resource for research, planning, and technical guidance.

AI assistance was mainly used for:

- Exploring and refining the initial project idea.
- Getting guidance on finding, preparing, and structuring CSV datasets suitable for Dhaka traffic prediction.
- Understanding different machine learning algorithms and selecting suitable models for traffic classification and regression.
- Comparing potential approaches and getting technical explanations during development.

The AI tools were used as a reference and learning resource. I reviewed the suggestions and used them to guide my own implementation, experimentation, and understanding of the project.


## ⚠️ Limitations

Although the system provides a machine learning-based approach to traffic prediction, it has some limitations:

- Predictions depend on historical traffic patterns and the quality of the available dataset.
- The current system does not use live GPS information or real-time traffic feeds.
- Weather conditions are not included in the current prediction model.
- Prediction accuracy may vary depending on the amount and representativeness of historical data.
- The recommended corridor is selected based on predicted traffic severity, not actual travel distance or navigation routes.
- Predictions for future dates should be considered estimates rather than guaranteed traffic conditions.

## 🚀 Future Improvements

There are several opportunities to improve and expand this project in the future:

- Integrating real-time traffic information through external APIs.
- Including weather conditions as additional prediction features.
- Exploring XGBoost and other advanced machine learning algorithms.
- Implementing LSTM for time-series traffic forecasting.
- Improving prediction accuracy with more historical traffic data.
- Developing an interactive dashboard using Streamlit.
- Adding geographical road mapping and route optimization.
- Comparing different machine learning models to identify the most suitable approach.

The project was developed as a practical learning experience to explore how machine learning can be applied to real-world transportation challenges in Dhaka, Bangladesh.

## This project is intended for educational and research purposes. You may explore, modify, and extend it with appropriate attribution.
