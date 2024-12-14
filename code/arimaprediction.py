import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data\merged_output.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Convert AQI to the target variable for prediction
df.set_index('timestamp', inplace=True)
aqi = df['aqi']

# Split the data into training and testing sets (80:20 ratio)
train_size = int(len(aqi) * 0.8)
train, test = aqi[:train_size], aqi[train_size:]

# Define a function for calculating accuracy (percentage error)
def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / y_true
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# Train ARIMA model and make predictions
def arima_model(train, test, p, d, q):
    # Define the model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # Forecast the AQI for the test set
    forecast = model_fit.forecast(steps=len(test))

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    accuracy = calculate_accuracy(test, forecast)
    
    return forecast, rmse, mae, accuracy

# Hyperparameter tuning: Trying different values for p, d, q
best_rmse = float('inf')
best_model = None
best_forecast = None
best_mae = None
best_accuracy = None
best_params = None

for p in range(1, 4):  # Example: p = 1, 2, 3
    for d in range(1, 2):  
        for q in range(1, 4):  
            forecast, rmse, mae, accuracy = arima_model(train, test, p, d, q)
            print(f"ARIMA({p},{d},{q}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")

            # Track best model based on RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = (p, d, q)
                best_forecast = forecast
                best_mae = mae
                best_accuracy = accuracy
                best_params = (p, d, q)

# Output best model's performance
print("\nBest ARIMA Model:", best_model)
print(f"Best RMSE: {best_rmse:.2f}, Best MAE: {best_mae:.2f}, Best Accuracy: {best_accuracy:.2f}%")

# Plot the forecasted vs actual AQI
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual AQI', color='blue')
plt.plot(test.index, best_forecast, label='Forecasted AQI', color='red')
plt.title('ARIMA Model Forecast vs Actual AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()
