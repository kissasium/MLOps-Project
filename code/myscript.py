import os
from dotenv import load_dotenv, find_dotenv
import requests
import pandas as pd
from datetime import datetime
import time
import logging

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set up logging
logging.basicConfig(filename='C:/Users/kissa zahra/Desktop/semesters/Semester 7/MLOps/MLOPs Final Project/course-project-kissasium/data_collection.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Check if the API key is set in the environment variables
if "OPENWEATHER_APIKEY" not in os.environ:
    logging.critical("API key is not set in environment variables")
    raise ValueError("API key is not set in environment variables")

# API key configuration
API_KEY = os.getenv('OPENWEATHER_APIKEY')

# API URLs
URLS = {
    "weather": "http://api.openweathermap.org/data/2.5/weather",
    "air_quality": "http://api.openweathermap.org/data/2.5/air_pollution"
}

# Parameters for the API calls
PARAMETERS = {
    "weather": {"q": "London,uk", "appid": API_KEY, "units": "metric"},  # temperature in Celsius
    "air_quality": {"lat": 51.5074, "lon": -0.1278, "appid": API_KEY}  # Latitude and Longitude of London
}

# Function to fetch data from the API
def fetch_data(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {str(e)}")
        return None

# Function to save the data into a CSV file
def save_data(data, filename):
    try:
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
        logging.info(f"Data saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")

# Main function to fetch weather and air quality data
def main():
    # Fetch weather data
    weather_data = fetch_data(URLS["weather"], PARAMETERS["weather"])
    if weather_data:
        # Extract relevant weather information
        weather_info = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "temperature": weather_data['main']['temp'],
            "humidity": weather_data['main']['humidity'],
            "weather_condition": weather_data['weather'][0]['description']
        }
        save_data(weather_info, "C:/Users/kissa zahra/Desktop/semesters/Semester 7/MLOps/MLOPs Final Project/course-project-kissasium/data/weather_data.csv")              #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   
    # Fetch air quality data
    air_quality_data = fetch_data(URLS["air_quality"], PARAMETERS["air_quality"])
    if air_quality_data:
        # Extract relevant air quality information
        air_quality_info = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "aqi": air_quality_data['list'][0]['main']['aqi'],
            "pm2_5": air_quality_data['list'][0]['components']['pm2_5'],
            "pm10": air_quality_data['list'][0]['components']['pm10'],
            "no2": air_quality_data['list'][0]['components']['no2'],
            "o3": air_quality_data['list'][0]['components']['o3']
        }
        save_data(air_quality_info, "C:/Users/kissa zahra/Desktop/semesters/Semester 7/MLOps/MLOPs Final Project/course-project-kissasium/data/air_quality_data.csv")     #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == "__main__":
    # Run the data collection at regular intervals (e.g., every 5 minutes)
    while True:
        main()
        sleeping_time = 300  # 5 minutes in seconds
        logging.info(f"Sleeping for {sleeping_time // 60} minutes")
        time.sleep(sleeping_time)  # Wait for 5 minutes before fetching the data again
