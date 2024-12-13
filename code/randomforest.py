# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt



# # Calculate accuracy as 100 - Mean Absolute Percentage Error (MAPE)
# def calculate_accuracy(y_true, y_pred):
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     accuracy = 100 - mape
#     return accuracy


# # Load the dataset
# df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# # Set the timestamp as index
# df.set_index('timestamp', inplace=True)

# # Select AQI as the target variable
# aqi = df['aqi']

# # Prepare features - Use past AQI values or any other relevant columns as features
# # You can use other columns like weather data, but for simplicity, we are using the AQI values
# # We will create a lag of 1 day for this example, i.e., the previous day's AQI will be used to predict today's AQI.
# df['AQI_lag'] = df['aqi'].shift(1)

# # Drop the missing values created by lagging
# df = df.dropna()

# # Split the data into features (X) and target (y)
# X = df[['AQI_lag']]  # Features (previous day's AQI)
# y = df['aqi']  # Target (current day's AQI)

# # Split the data into training and testing sets (80:20 ratio)
# train_size = int(len(df) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Initialize and train the Random Forest model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_model.predict(X_test)

# # Calculate evaluation metrics
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# # Calculate accuracy for the test set
# accuracy = calculate_accuracy(y_test, y_pred)
# # Print evaluation metrics
# print(f"Random Forest - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# print(f"Random Forest Model Accuracy: {accuracy:.2f}%")


# # Plot the actual vs predicted AQI for the test period
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.index, y_test, label='Actual AQI', color='blue')
# plt.plot(y_test.index, y_pred, label='Predicted AQI', color='red')
# plt.title('Random Forest Model: Actual vs Predicted AQI')
# plt.xlabel('Date')
# plt.ylabel('AQI')
# plt.legend()
# plt.show()

# # Plotting for a specific date range (e.g., 7th Dec to 13th Dec)
# start_date = '2023-12-07'
# end_date = '2023-12-13'

# # Filter the data for the specified date range
# date_filtered_actual = y_test[start_date:end_date]
# date_filtered_predicted = y_pred[(y_test.index >= start_date) & (y_test.index <= end_date)]

# # Plot the actual vs predicted AQI for the specified date range
# plt.figure(figsize=(10, 6))
# plt.plot(date_filtered_actual.index, date_filtered_actual, label='Actual AQI', color='blue')
# plt.plot(date_filtered_actual.index, date_filtered_predicted, label='Predicted AQI', color='red')
# plt.title(f'Random Forest Model: Actual vs Predicted AQI ({start_date} to {end_date})')
# plt.xlabel('Date')
# plt.ylabel('AQI')
# plt.legend()
# plt.show()




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Convert AQI to the target variable for prediction
df.set_index('timestamp', inplace=True)
aqi = df['aqi']

# Feature engineering: Use previous AQI values as features
df['prev_aqi'] = df['aqi'].shift(1)  # 1-day lag as feature
df.dropna(inplace=True)  # Drop rows with NaN values due to shifting

# Define features and target variable
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

# Train Random Forest model and make predictions
def random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth):
    # Define the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = calculate_accuracy(y_test, y_pred)
    
    return y_pred, rmse, mae, accuracy

# Hyperparameter tuning: Trying different values for n_estimators and max_depth
best_rmse = float('inf')
best_model = None
best_y_pred = None
best_mae = None
best_accuracy = None
best_params = None

for n_estimators in [50, 100, 150]:  # Example values for n_estimators
    for max_depth in [5, 10, 15]:  # Example values for max_depth
        y_pred, rmse, mae, accuracy = random_forest_model(X_train, y_train, X_test, y_test, n_estimators, max_depth)
        print(f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.2f}%")

        # Track best model based on RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = (n_estimators, max_depth)
            best_y_pred = y_pred
            best_mae = mae
            best_accuracy = accuracy
            best_params = (n_estimators, max_depth)

# Output best model's performance
print("\nBest Random Forest Model:", best_model)
print(f"Best RMSE: {best_rmse:.2f}, Best MAE: {best_mae:.2f}, Best Accuracy: {best_accuracy:.2f}%")

# Plot the forecasted vs actual AQI
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual AQI', color='blue')
plt.plot(y_test.index, best_y_pred, label='Predicted AQI', color='red')
plt.title('Random Forest Model Prediction vs Actual AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

