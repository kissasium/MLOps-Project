import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(filename='live_test_predictions.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# API Key and URLs
API_KEY = os.getenv('OPENWEATHER_APIKEY')
URLS = {
    "weather": "http://api.openweathermap.org/data/2.5/weather",
    "air_quality": "http://api.openweathermap.org/data/2.5/air_pollution"
}

# Parameters for the API calls
PARAMETERS = {
    "weather": {"q": "London,uk", "appid": API_KEY, "units": "metric"},
    "air_quality": {"lat": 51.5074, "lon": -0.1278, "appid": API_KEY}
}

# Fetch data from API
def fetch_data(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {str(e)}")
        return None

# Save data to CSV
def save_data(data, filename):
    try:
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
        logging.info(f"Data saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")

# Merge the data (weather and air quality data)
def merge_data():
    weather_file = "data/weather_data.csv"
    air_quality_file = "data/air_quality_data.csv"
    output_file = "data/merged_output.csv"

    weather_df = pd.read_csv(weather_file)
    air_quality_df = pd.read_csv(air_quality_file)

    merged_df = pd.merge(weather_df, air_quality_df, on="timestamp", how="inner")
    merged_df.to_csv(output_file, index=False)

    logging.info(f"Merged data saved to {output_file}")

# ARIMA Model Training and Testing
def arima_model():
    data_file = "data/merged_output.csv"
    df = pd.read_csv(data_file)

    # Preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    aqi = df['aqi']
    train_size = int(len(aqi) * 0.8)
    train, test = aqi[:train_size], aqi[train_size:]

    # ARIMA Model
    best_aic = np.inf
    best_order = None
    best_model = None
    best_predictions = None

    for p in range(1, 3):
        for d in range(1, 2):
            for q in range(1, 3):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=len(test))

                    aic = model_fit.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model_fit
                        best_predictions = predictions
                except Exception as e:
                    logging.error(f"Error with ARIMA({p}, {d}, {q}): {e}")

    # Evaluate the model
    mse = mean_squared_error(test, best_predictions)
    mae = mean_absolute_error(test, best_predictions)
    accuracy = 100 - (np.sqrt(mse) / np.mean(test) * 100)

    logging.info(f"Best ARIMA model order: {best_order}")
    logging.info(f"Predictions: {best_predictions}")
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"Accuracy: {accuracy}%")

# Main function
def main():
    while True:
        # Fetch live data from the APIs
        weather_data = fetch_data(URLS["weather"], PARAMETERS["weather"])
        if weather_data:
            weather_info = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "temperature": weather_data['main']['temp'],
                "humidity": weather_data['main']['humidity'],
                "weather_condition": weather_data['weather'][0]['description']
            }
            save_data(weather_info, "data/weather_data.csv")

        air_quality_data = fetch_data(URLS["air_quality"], PARAMETERS["air_quality"])
        if air_quality_data:
            air_quality_info = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "aqi": air_quality_data['list'][0]['main']['aqi'],
                "pm2_5": air_quality_data['list'][0]['components']['pm2_5'],
                "pm10": air_quality_data['list'][0]['components']['pm10'],
                "no2": air_quality_data['list'][0]['components']['no2'],
                "o3": air_quality_data['list'][0]['components']['o3']
            }
            save_data(air_quality_info, "data/air_quality_data.csv")

        # Merge the new data and train/test the ARIMA model
        merge_data()
        arima_model()

        # Sleep for a specified period before the next cycle
        sleep_time = 60 * 5  # Sleep for 5 minutes (adjust as necessary)
        logging.info(f"Sleeping for {sleep_time // 60} minutes")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
