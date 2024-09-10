from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from datetime import datetime, timedelta

# create our Flask app
app = Flask(__name__)

def get_binance_klines_url(symbol, interval='1d', limit=30):
    base_url = "https://api.binance.com/api/v3/klines"
    url = f"{base_url}?symbol={symbol}&interval={interval}&limit={limit}"
    return url

# define our endpoint
@app.route("/inference/<string:symbol>")
def get_inference(symbol):
    """Generate inference for given symbol."""
    try:
        # Get the data from Binance
        url = get_binance_klines_url(symbol)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        df = df[['Open time', 'Close']]
        df.columns = ['ds', 'y']
        print(df.tail(5))
    else:
        return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}),
                        status=response.status_code,
                        mimetype='application/json')

    # Prepare data for ARIMA
    y = df['y'].values

    # Use auto_arima to find the best parameters
    model = auto_arima(y, start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                       start_P=0, seasonal=False, d=1, D=1, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)

    # Fit the ARIMA model
    arima_model = ARIMA(y, order=model.order)
    results = arima_model.fit()

    # Make a forecast for the next day
    forecast = results.forecast(steps=1)
    forecasted_value = forecast[0]
    print(forecasted_value)  # Print the forecasted value

    return Response(str(forecasted_value), status=200)

# run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
