from prometheus_client import start_http_server, Summary, Gauge
import time
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Metrics for API performance and ARIMA prediction
DATA_INGESTION_TIME = Summary('data_ingestion_duration_seconds', 'Time spent on data ingestion')
API_LATENCY = Summary('api_latency_seconds', 'Time taken for API requests')
PREDICTION_ERROR = Gauge('arima_prediction_error', 'Prediction error of ARIMA model')

# Simulated data ingestion function
@DATA_INGESTION_TIME.time()
def data_ingestion():
    # Load your data
    df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
    # Perform data processing or merging if required
    return dfs

# Simulated ARIMA prediction function
def arima_prediction(df):
    aqi = df['aqi']
    model = ARIMA(aqi, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)

    # Calculate prediction error
    error = abs(forecast - aqi[-5:]).mean()
    PREDICTION_ERROR.set(error)

# API call simulation (For monitoring)
@API_LATENCY.time()
def api_call():
    # Simulate API call (replace with actual API request logic)
    time.sleep(1)  # Simulate some delay

if __name__ == '__main__':
    # Start Prometheus HTTP server to expose metrics
    start_http_server(8000)
    print("Prometheus metrics server running on http://localhost:8000")

    # Main loop
    while True:
        df = data_ingestion()
        arima_prediction(df)
        api_call()
        time.sleep(10)  # Run this every 10 seconds
