# AQI Time-Series Forecasting using Machine Learning

An end-to-end **Air Quality Index (AQI) forecasting system** that predicts the **next day’s AQI** for multiple Indian cities using historical air quality data and machine learning techniques.  
This project demonstrates a complete machine learning pipeline, from data preprocessing and feature engineering to model deployment using Streamlit.

---

## Project Overview

Air quality monitoring plays a vital role in environmental safety and public health.  
This project focuses on forecasting AQI values by capturing temporal patterns in historical air quality data using machine learning.

The project was developed **under the guidance of Ms. Tripti Shrivastava** and involved collaborative teamwork throughout the development process.

---

## Key Features

- Time-series based AQI prediction
- City-wise AQI forecasting
- Feature engineering using historical AQI trends
- Machine learning model deployment via Streamlit
- User-friendly and interactive web interface

---

## Machine Learning Approach

### Feature Engineering

The following features were engineered to capture AQI trends over time:

- AQI lag of 1 day  
- AQI lag of 7 days  
- 7-day rolling mean of AQI values  
- City-wise categorical encoding using Label Encoding  

### Model Used

- **Algorithm:** Random Forest Regressor  
- **Rationale:** Handles non-linear relationships effectively and performs well on noisy real-world datasets  

### Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Machine Learning Model:** Random Forest Regressor  
- **Deployment Framework:** Streamlit  
- **Development Tools:** Google Colab, Visual Studio Code  

---

## Project Structure
```bash
AQI_TimeSeries_Forecasting_ML
│
├── app.py # Streamlit application
├── city_encoder.pkl # Saved LabelEncoder for city feature
├── aqi.csv # Dataset used for training and inference
└── README.md # Project documentation
```
---

## How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AQI-TimeSeries-Forecasting-ML.git
cd AQI-TimeSeries-Forecasting-ML 
```

## Step 2: Install Dependencies
pip install -r requirements.txt

## Step 3: Run the Streamlit Application
streamlit run app.py


The application will open in your default web browser.
