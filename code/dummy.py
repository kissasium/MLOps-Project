# import os
# from dotenv import load_dotenv, find_dotenv
# import requests
# import pandas as pd
# from datetime import datetime
# import time
# import logging

# # Load environment variables from .env file
# load_dotenv(find_dotenv())

# # Set up logging
# logging.basicConfig(filename='data_collection.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# # Check if the API key is set in the environment variables
# if "OPENWEATHER_APIKEY" not in os.environ:
#     logging.critical("API key is not set in environment variables")
#     raise ValueError("API key is not set in environment variables")

# # API key configuration
# API_KEY = os.getenv('OPENWEATHER_APIKEY')

# # API URLs
# URLS = {
#     "weather": "http://api.openweathermap.org/data/2.5/weather",
#     "air_quality": "http://api.openweathermap.org/data/2.5/air_pollution"
# }

# # Parameters for the API calls
# PARAMETERS = {
#     "weather": {"q": "London,uk", "appid": API_KEY, "units": "metric"},  # temperature in Celsius
#     "air_quality": {"lat": 51.5074, "lon": -0.1278, "appid": API_KEY}  # Latitude and Longitude of London
# }

# # Function to fetch data from the API
# def fetch_data(url, params):
#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
#         return response.json()
#     except requests.RequestException as e:
#         logging.error(f"Error fetching data from {url}: {str(e)}")
#         return None

# # Function to save the data into a CSV file
# def save_data(data, filename):
#     try:
#         df = pd.DataFrame([data])
#         df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
#         logging.info(f"Data saved successfully to {filename}")
#     except Exception as e:
#         logging.error(f"Error saving data: {str(e)}")

# # Main function to fetch weather and air quality data
# def main():
#     # Fetch weather data
#     weather_data = fetch_data(URLS["weather"], PARAMETERS["weather"])
#     if weather_data:
#         # Extract relevant weather information
#         weather_info = {
#             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             "temperature": weather_data['main']['temp'],
#             "humidity": weather_data['main']['humidity'],
#             "weather_condition": weather_data['weather'][0]['description']
#         }
#         save_data(weather_info, "data/weather_data.csv")
   
#     # Fetch air quality data
#     air_quality_data = fetch_data(URLS["air_quality"], PARAMETERS["air_quality"])
#     if air_quality_data:
#         # Extract relevant air quality information
#         air_quality_info = {
#             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             "aqi": air_quality_data['list'][0]['main']['aqi'],
#             "pm2_5": air_quality_data['list'][0]['components']['pm2_5'],
#             "pm10": air_quality_data['list'][0]['components']['pm10'],
#             "no2": air_quality_data['list'][0]['components']['no2'],
#             "o3": air_quality_data['list'][0]['components']['o3']
#         }
#         save_data(air_quality_info, "data/air_quality_data.csv")

# if __name__ == "__main__":
#     # Run the data collection at regular intervals (e.g., every 5 minutes)
#     while True:
#         main()
#         sleeping_time = 300  # 5 minutes in seconds
#         logging.info(f"Sleeping for {sleeping_time // 60} minutes")
#         time.sleep(sleeping_time)  # Wait for 5 minutes before fetching the data again


# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# import joblib

# # Example data loading - adjust to your dataset
# data = pd.read_csv('data/merged_output.csv')  # Load your time series data
# data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime
# data.set_index('Date', inplace=True)

# # Fit ARIMA model (2, 1, 1)
# model = ARIMA(data['value'], order=(2, 1, 1))  # Replace 'value' with your column name
# model_fit = model.fit()

# # Save the trained model
# joblib.dump(model_fit, 'arima_model.pkl')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
from statsmodels.tsa.arima.model import ARIMA
import logging
import os

# Set up a logging file
log_file = "custom_logs.txt"
if os.path.exists(log_file):
    os.remove(log_file)  # Remove existing log file
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting Pollution Prediction Experiment")

# Load the dataset
df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Convert AQI to the target variable for prediction
aqi = df['aqi']

# Split the data into training and testing sets (80:20 ratio)
train_size = int(len(aqi) * 0.8)
train, test = aqi[:train_size], aqi[train_size:]

# Define a function for calculating accuracy (percentage error)
def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / y_true
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# ---- ARIMA Model ----
mlflow.set_experiment('Pollution Prediction - ARIMA')

def arima_model(train, test, p, d, q):
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    accuracy = calculate_accuracy(test, forecast)
    return forecast, rmse, mae, accuracy, model_fit

# Hyperparameter tuning for ARIMA
best_rmse_arima = float('inf')
best_model_arima = None
best_forecast_arima = None

for p in range(1, 4):
    for d in range(1, 2):
        for q in range(1, 4):
            with mlflow.start_run():
                forecast, rmse, mae, accuracy, model_fit = arima_model(train, test, p, d, q)
                
                # Log parameters and metrics
                mlflow.log_param("p", p)
                mlflow.log_param("d", d)
                mlflow.log_param("q", q)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("Accuracy", accuracy)
                
                # Log to the custom log file
                msg = (f"ARIMA({p},{d},{q}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")
                logging.info(msg)
                print(msg)

                # Save ARIMA model summary as a text artifact
                with open("arima_summary.txt", "w") as f:
                    f.write(str(model_fit.summary()))
                mlflow.log_artifact("arima_summary.txt")
                
                if rmse < best_rmse_arima:
                    best_rmse_arima = rmse
                    best_model_arima = (p, d, q)
                    best_forecast_arima = forecast

logging.info(f"Best ARIMA Model: {best_model_arima} with RMSE: {best_rmse_arima:.2f}")
print("\nBest ARIMA Model:", best_model_arima)
print(f"Best RMSE: {best_rmse_arima:.2f}")

# ---- Visualization ----
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual AQI', color='blue')
plt.plot(test.index, best_forecast_arima, label='Forecasted AQI (ARIMA)', color='red')
plt.title('ARIMA Model Forecast vs Actual AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.savefig("arima_results.png")
mlflow.log_artifact("arima_results.png")
plt.show()
