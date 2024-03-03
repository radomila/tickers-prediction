import yfinance as yf
from sklearn.linear_model import LinearRegression

# Data
data_btc = yf.download("BTC-USD", start="2020-02-28", end="2024-03-03")
data_aapl = yf.download("AAPL", start="2020-02-28", end="2024-03-03")
data_xom = yf.download("XOM", start="2020-02-28", end="2024-03-03")

data = [data_btc, data_aapl, data_xom]

future_predictions = []

for dataset in data:
    X = dataset.loc[:, ~dataset.columns.isin(["Close", "Date"])]
    y = dataset["Close"].values

    # Train data
    X_train = X[:-5]
    y_train = y[:-5]

    # Test data
    X_test = X[-5:]
    y_test = y[-5:]

    reg = LinearRegression().fit(X_train, y_train)

    future_values = reg.predict(X_test)
    future_predictions.append(future_values)

predictions_dict = {}

for i, ticker in enumerate(["BTC-USD", "AAPL", "XOM"]):
    predictions_dict[ticker] = future_predictions[i]

for ticker, predictions in predictions_dict.items():
    print(f"{ticker} = {predictions.tolist()}")

