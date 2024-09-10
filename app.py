from flask import Flask, Response
import requests
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# create our Flask app
app = Flask(__name__)

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

# define our endpoint
@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for given token."""
    try:
        # Get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your API key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"])
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"], unit='ms')
        df = df[:-1]  # Removing today's price
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
