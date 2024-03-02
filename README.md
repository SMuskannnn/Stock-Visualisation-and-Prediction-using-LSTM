# Stock-Visualisation-and-Prediction-using-LSTM


# Stock Movement Visualization and Prediction

This project is aimed at visualizing stock movement and predicting future stock prices using historical data. It provides a web interface built with Streamlit where users can input a stock ticker symbol, select a date range, and view various visualizations and predictions related to the chosen stock.

## Features

- **Summary Tab:** Displays an overview of the selected stock, including the latest price, price chart, stock information, top news related to the stock, annual return, standard deviation, and risk-adjusted return.

- **Dashboard Tab:** Provides interactive visualizations such as candlestick charts for price and volume, moving averages (simple and exponential), MACD (Moving Average Convergence Divergence), and RSI (Relative Strength Index). Users can customize the parameters for each visualization.

- **Prediction Tab:** Utilizes a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. It includes training the model, validating the model, and displaying predicted vs. actual stock prices for both training and testing data.

## Access the site on

```bash
https://stock-visualisation-and-prediction-using-lstm-kfqyijysz3pmbnou.streamlit.app/
```
## How to Use

1. Clone the repository to your local machine:

    ```bash
    https://github.com/SMuskannnn/Stock-Visualisation-and-Prediction-using-LSTM.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. Access the web interface by opening the provided URL in your web browser.

5. Enter a stock ticker symbol, select a date range, and explore the various tabs to view visualizations and predictions.

## Technologies Used

- Python
- Streamlit
- Plotly
- PyTorch
- yfinance
