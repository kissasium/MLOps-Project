import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/merged_output.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Prepare data for ARIMA model
aqi = df['aqi']
train_size = int(len(aqi) * 0.8)
train, test = aqi[:train_size], aqi[train_size:]

def predict_aqi(days=5):
    """
    Predict AQI for the next 5 days using the best ARIMA model (2,1,1)
    """
    # Fit the ARIMA model with best parameters
    model = ARIMA(train, order=(2, 1, 1))
    model_fit = model.fit()
    
    # Forecast the next 5 days
    forecast = model_fit.forecast(steps=days)
    
    return forecast.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Predict AQI for next 5 days
        predictions = predict_aqi()
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
