# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA

# # Load the dataset
# df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# # Convert AQI to the target variable for prediction
# df.set_index('timestamp', inplace=True)
# aqi = df['aqi']

# # Feature engineering for Random Forest
# df['prev_aqi'] = df['aqi'].shift(1)  # 1-day lag as feature
# df.dropna(inplace=True)  # Drop rows with NaN values due to shifting

# # Define features and target variable
# X = df[['prev_aqi']]  # You can add more features here if needed
# y = df['aqi']

# # Split the data into training and testing sets (80:20 ratio)
# train_size = int(len(aqi) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Define a function for calculating accuracy (percentage error)
# def calculate_accuracy(y_true, y_pred):
#     error = np.abs(y_true - y_pred) / y_true
#     accuracy = 100 - np.mean(error) * 100
#     return accuracy

# # Random Forest model function
# def random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth):
#     model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
    
#     # Metrics
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     accuracy = calculate_accuracy(y_test, y_pred)
    
#     return y_pred, rmse, mae, accuracy

# # ARIMA model function
# def arima_model(train, test, p, d, q):
#     model = ARIMA(train, order=(p, d, q))
#     model_fit = model.fit()

#     forecast = model_fit.forecast(steps=len(test))

#     rmse = np.sqrt(mean_squared_error(test, forecast))
#     mae = mean_absolute_error(test, forecast)
#     accuracy = calculate_accuracy(test, forecast)
    
#     return forecast, rmse, mae, accuracy

# # Hyperparameter tuning for Random Forest
# def random_forest_experiment(X_train, y_train, X_test, y_test):
#     best_rmse = float('inf')
#     best_model = None
#     best_y_pred = None
#     best_mae = None
#     best_accuracy = None

#     for n_estimators in [50, 100, 150]:
#         for max_depth in [5, 10, 15]:
#             with mlflow.start_run():
#                 y_pred, rmse, mae, accuracy = random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth)
                
#                 # Log metrics to MLflow
#                 mlflow.log_param("n_estimators", n_estimators)
#                 mlflow.log_param("max_depth", max_depth)
#                 mlflow.log_metric("RMSE", rmse)
#                 mlflow.log_metric("MAE", mae)
#                 mlflow.log_metric("Accuracy", accuracy)

#                 # Track best model based on RMSE
#                 if rmse < best_rmse:
#                     best_rmse = rmse
#                     best_model = (n_estimators, max_depth)
#                     best_y_pred = y_pred
#                     best_mae = mae
#                     best_accuracy = accuracy
    
#     return best_model, best_y_pred, best_rmse, best_mae, best_accuracy

# # Hyperparameter tuning for ARIMA
# def arima_experiment(train, test):
#     best_rmse = float('inf')
#     best_model = None
#     best_forecast = None
#     best_mae = None
#     best_accuracy = None

#     for p in range(1, 4):
#         for d in range(1, 2):  # d = 1 (differencing)
#             for q in range(1, 4):
#                 with mlflow.start_run():
#                     forecast, rmse, mae, accuracy = arima_model(train, test, p, d, q)

#                     # Log metrics to MLflow
#                     mlflow.log_param("p", p)
#                     mlflow.log_param("d", d)
#                     mlflow.log_param("q", q)
#                     mlflow.log_metric("RMSE", rmse)
#                     mlflow.log_metric("MAE", mae)
#                     mlflow.log_metric("Accuracy", accuracy)

#                     # Track best model based on RMSE
#                     if rmse < best_rmse:
#                         best_rmse = rmse
#                         best_model = (p, d, q)
#                         best_forecast = forecast
#                         best_mae = mae
#                         best_accuracy = accuracy
    
#     return best_model, best_forecast, best_rmse, best_mae, best_accuracy

# # Run Random Forest experiment
# best_rf_model, best_rf_y_pred, best_rf_rmse, best_rf_mae, best_rf_accuracy = random_forest_experiment(X_train, y_train, X_test, y_test)

# # Run ARIMA experiment
# best_arima_model, best_arima_forecast, best_arima_rmse, best_arima_mae, best_arima_accuracy = arima_experiment(aqi[:train_size], aqi[train_size:])

# # Output best models' performance
# print("\nBest Random Forest Model:", best_rf_model)
# print(f"Best Random Forest RMSE: {best_rf_rmse:.2f}, Best MAE: {best_rf_mae:.2f}, Best Accuracy: {best_rf_accuracy:.2f}%")

# print("\nBest ARIMA Model:", best_arima_model)
# print(f"Best ARIMA RMSE: {best_arima_rmse:.2f}, Best MAE: {best_arima_mae:.2f}, Best Accuracy: {best_arima_accuracy:.2f}%")

# # Plot the forecasted vs actual AQI for Random Forest
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.index, y_test, label='Actual AQI', color='blue')
# plt.plot(y_test.index, best_rf_y_pred, label='Predicted AQI (RF)', color='red')
# plt.title('Random Forest Model Prediction vs Actual AQI')
# plt.xlabel('Date')
# plt.ylabel('AQI')
# plt.legend()
# plt.show()

# # Plot the forecasted vs actual AQI for ARIMA
# plt.figure(figsize=(10, 6))
# plt.plot(aqi[train_size:].index, aqi[train_size:], label='Actual AQI', color='blue')
# plt.plot(aqi[train_size:].index, best_arima_forecast, label='Forecasted AQI (ARIMA)', color='red')
# plt.title('ARIMA Model Forecast vs Actual AQI')
# plt.xlabel('Date')
# plt.ylabel('AQI')
# plt.legend()
# plt.show()




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Convert AQI to the target variable for prediction
aqi = df['aqi']

# Feature engineering for Random Forest: Use previous AQI values as features
df['prev_aqi'] = df['aqi'].shift(1)  # 1-day lag as feature
df.dropna(inplace=True)  # Drop rows with NaN values due to shifting
X = df[['prev_aqi']]  # You can add more features here if needed
y = df['aqi']

# Split the data into training and testing sets (80:20 ratio)
train_size = int(len(aqi) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a function for calculating accuracy (percentage error)
def calculate_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred) / y_true
    accuracy = 100 - np.mean(error) * 100
    return accuracy

# ---- Random Forest Model ----
# Set experiment name for Random Forest
mlflow.set_experiment('Pollution Prediction - Random Forest')

def random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth):
    # Define and train the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = calculate_accuracy(y_test, y_pred)
    
    return y_pred, rmse, mae, accuracy

# Hyperparameter tuning for Random Forest
best_rmse_rf = float('inf')
best_model_rf = None
best_y_pred_rf = None

for n_estimators in [50, 100, 150]:  # Example values for n_estimators
    for max_depth in [5, 10, 15]:  # Example values for max_depth
        with mlflow.start_run():  # Start logging for this run
            y_pred, rmse, mae, accuracy = random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth)
            
            # Log parameters and metrics
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("Accuracy", accuracy)

            print(f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")

            # Track best model based on RMSE
            if rmse < best_rmse_rf:
                best_rmse_rf = rmse
                best_model_rf = (n_estimators, max_depth)
                best_y_pred_rf = y_pred

print("\nBest Random Forest Model:", best_model_rf)
print(f"Best RMSE: {best_rmse_rf:.2f}")

# ---- ARIMA Model ----
# Set experiment name for ARIMA
mlflow.set_experiment('Pollution Prediction - ARIMA')

# ARIMA model training
train, test = aqi[:train_size], aqi[train_size:]

def arima_model(train, test, p, d, q):
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    accuracy = calculate_accuracy(test, forecast)
    
    return forecast, rmse, mae, accuracy

# Hyperparameter tuning for ARIMA
best_rmse_arima = float('inf')
best_model_arima = None
best_forecast_arima = None

for p in range(1, 4):  # Example: p = 1, 2, 3
    for d in range(1, 2):  # d = 1 (differencing)
        for q in range(1, 4):  # Example: q = 1, 2, 3
            with mlflow.start_run():  # Start logging for this run
                forecast, rmse, mae, accuracy = arima_model(train, test, p, d, q)
                
                # Log parameters and metrics
                mlflow.log_param("p", p)
                mlflow.log_param("d", d)
                mlflow.log_param("q", q)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("Accuracy", accuracy)

                print(f"ARIMA({p},{d},{q}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")

                # Track best model based on RMSE
                if rmse < best_rmse_arima:
                    best_rmse_arima = rmse
                    best_model_arima = (p, d, q)
                    best_forecast_arima = forecast

print("\nBest ARIMA Model:", best_model_arima)
print(f"Best RMSE: {best_rmse_arima:.2f}")

# ---- Visualization ----
# Plot the forecasted vs actual AQI for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual AQI', color='blue')
plt.plot(y_test.index, best_y_pred_rf, label='Predicted AQI (Random Forest)', color='red')
plt.title('Random Forest Model Prediction vs Actual AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

# Plot the forecasted vs actual AQI for ARIMA
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual AQI', color='blue')
plt.plot(test.index, best_forecast_arima, label='Forecasted AQI (ARIMA)', color='red')
plt.title('ARIMA Model Forecast vs Actual AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

