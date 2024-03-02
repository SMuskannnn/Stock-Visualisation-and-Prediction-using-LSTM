import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import torch
import torch.nn as nn
import plotly.graph_objs as go
import plotly.io as pio

st.title('Stock Movement Visualization and Prediction')
start_date = st.date_input('Start Date', min_value=datetime.datetime(2000, 1, 1), max_value=datetime.datetime(2025, 1, 1))
ticker = st.text_input('Ticker')
end_date = st.date_input('End Date', min_value=datetime.datetime(2000, 1, 1), max_value=datetime.datetime(2025, 1, 1))


#functions

def SMA(data, period, column = 'Close'):
    return data[column].rolling(window = period).mean()


def EMA(data, period, column='Close'):
    return data[column].ewm(span=period, adjust = False).mean()

def MACD(data, period_long=26, period_short=12,period_signal=9,column='Close'):
    #calculate the short term EMA
    shortEMA = EMA(data, period_short, column = column)
    #calculate  long term EMA
    longEMA = EMA(data, period_long, column = column)
    #calcultae teh moving averagge convergence/divergence
    data['MACD'] = shortEMA-longEMA
    #calculate signal_line
    data['Signal_Line'] = EMA(data, period_signal,column='MACD')
    #histogram
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    return data

def RSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    data['RSI'] = RSI
    return data


from copy import deepcopy as dc
def prepared_data(df,n_steps):
    df = dc(df)
    #df.set_index('Date',inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace = True)
    return df


#object to make dataset time series
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        return self.x[i],self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        batch_size = x.size(0)
        h0= torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _= self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output= model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 99: # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss
    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('*****************')
    print()


def inverse_scaling(predicted, x_train, lookback):
    train_predictions = predicted.flatten()
    dummies = np.zeros((x_train.shape[0], lookback+1))
    dummies[:, 0]= train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])
    return train_predictions

def inverse_data(y_train, x_train, lookback):
    dummies = np.zeros((x_train.shape[0], lookback+1))
    dummies[:, 0] =y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])
    return new_y_train


# Function to get the latest stock data
def get_latest_stock_data(ticker):
    stock = yf.Ticker(ticker)
    latest_data = stock.history(period='1d')
    return latest_data


if ticker:
    stck = yf.download(ticker, start_date, end_date)
    tick = yf.Ticker(ticker)

    summary, dashboard = st.tabs(["Summary","Dashboard"])
    
    with summary:
        st.header(ticker)
        # Get the latest price and open price
        latest_data = get_latest_stock_data(ticker)
	latest_price = latest_data['Close'].iloc[-1]
	# try:
	#     latest_price = latest_data['Close'].iloc[-1]
	# except IndexError:
	#     st.warning("No data available for the selected ticker. Please enter a valid ticker.")
	#     st.stop()
        open_price = latest_data['Open'].iloc[0]
        # Calculate the delta
        delta = latest_price - open_price
        # Define color based on delta
        color = "green" if delta >= 0 else "red"

        # Display the metric with appropriate styling
        st.write(
            f"""
            <div style="padding: 10px; border-radius: 10px; background-color: #f0f0f0;">
                <span style="font-size: 24px; color: {color};">${latest_price:.2f}</span>
                <span style="font-size: 16px; color: {color};">({delta:+.2f})</span>
            </div>""",
            unsafe_allow_html=True)
        #area chart to display price movement
        fig = go.Figure(go.Scatter(x=stck.index, y=stck['Close'], fill='tozeroy', mode='lines', name='Stock Price'))
        fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
        st.plotly_chart(fig)
        
        #stock information
        st.subheader("stock information")
        stck1 = stck
        stck1['%change'] = stck['Adj Close']/stck['Adj Close'].shift(1)-1
        stck1.dropna(inplace = True)
        st.write(stck1)
        annual_return = stck1['%change'].mean()*252*100
        st.write('Annual Return is',annual_return,'%')
        stdev = np.std(stck1['%change'])*np.sqrt(252)
        st.write('Standard Deviation is',stdev*100,'%')
        st.write('Risk Adj Return is',annual_return/(stdev*100))

        #Top 10 news
        st.header(f'News of {ticker}')
        #tick = yf.Ticker(ticker)
        stock_news = tick.news

        # Display the headlines and links as clickable links
        for news_item in stock_news:
                st.write(f"- [{news_item['title']}]({news_item['link']})")
                #st.write(news_item['summary'])
        with dashboard:
            st.subheader('Price and Volume movement')
            
            #Price and volume, using candlestick,lines and bar chart

            # Create figure
            fig = go.Figure()

            # Add candlestick chart for price data
            fig.add_trace(go.Candlestick(x=stck.index,
                            open=stck['Open'],
                            high=stck['High'],
                            low=stck['Low'],
                            close=stck['Close'], name='Candlestick'))

            # Add line chart for price data
            fig.add_trace(go.Scatter(x=stck.index, y=stck['Close'], mode='lines', name='Price (Line)'))

            # Add bar chart for volume data
            fig.add_trace(go.Bar(x=stck.index, y=stck['Volume'], name='Volume', yaxis='y2'))

            # Update layout
            fig.update_layout(title='Price and Volume Over Time',
                            yaxis=dict(title='Price'),
                            yaxis2=dict(title='Volume', overlaying='y', side='right'))

            # Add selector for price chart type
            price_chart_type = st.selectbox('Select Price Chart Type', ['Candlestick', 'Line'])

            # Display the selected price chart type along with volume
            if price_chart_type == 'Candlestick':
                fig.data = [trace for trace in fig.data if 'Candlestick' in trace.name or 'Volume' in trace.name]  # Keep candlestick and volume traces
                st.plotly_chart(fig)
            elif price_chart_type == 'Line':
                fig.data = [trace for trace in fig.data if 'Price (Line)' in trace.name or 'Volume' in trace.name]  # Keep line and volume traces
                st.plotly_chart(fig)
            

            #Moving averages
            st.subheader('Simple Moving Averages')
            ma_days = st.slider('Moving Average Days', min_value=1, max_value=200, value=50, step=1)

            ma1 = SMA(stck, ma_days)
            ma2 = SMA(stck, ma_days*2)

            # Create traces
            trace_close = go.Scatter(x=stck.index, y=stck['Close'], mode='lines', name='Close')
            trace_ma1 = go.Scatter(x=stck.index, y=ma1, mode='lines', name=f'MA{ma_days}', line=dict(color='red'))
            trace_ma2 = go.Scatter(x=stck.index, y=ma2, mode='lines', name=f'MA{ma_days*2}', line=dict(color='green'))

            # Create figure
            fig = go.Figure()

            # Add traces to the figure
            fig.add_trace(trace_close)
            fig.add_trace(trace_ma1)
            fig.add_trace(trace_ma2)


            # Update layout
            fig.update_layout(title=f'Stock Close Price with {ma_days} Days  and {ma_days*2} Days Moving Average',
                            xaxis=dict(title='Date', rangeslider=dict(visible=True), type="date"),
                            yaxis=dict(title='Price'),
                            autosize=True)

            # Show plot using Streamlit
            st.plotly_chart(fig)


            
            st.subheader('Exponential Moving Averages')
            moving_avg_days = 200
            ema_type = st.radio("EMA Type", ["Single EMA", "Double EMA"])

            if ema_type == "Single EMA":
                ema_period = st.slider('Select Period for EMA', min_value=1, max_value=200, value=moving_avg_days)
                ema = EMA(stck, ema_period)
                trace_ema = go.Scatter(x=stck.index, y=ema, mode='lines', name=f'EMA{ema_period}', line=dict(color='yellow'))

            elif ema_type == "Double EMA":
                ema_period1 = st.slider('Select Period for First EMA', min_value=1, max_value=200, value=20)
                ema_period2 = st.slider('Select Period for Second EMA', min_value=1, max_value=200, value=50)
                ema1 = EMA(stck, ema_period1)
                ema2 = EMA(stck, ema_period2)
                trace_ema1 = go.Scatter(x=stck.index, y=ema1, mode='lines', name=f'EMA{ema_period1}', line=dict(color='yellow'))
                trace_ema2 = go.Scatter(x=stck.index, y=ema2, mode='lines', name=f'EMA{ema_period2}', line=dict(color='orange'))

            trace_close = go.Scatter(x=stck.index, y=stck['Close'], mode='lines', name='Close')

            fig = go.Figure()
            fig.add_trace(trace_close)

            if ema_type == "Single EMA":
                fig.add_trace(trace_ema)
            elif ema_type == "Double EMA":
                fig.add_trace(trace_ema1)
                fig.add_trace(trace_ema2)

            fig.update_layout(title='Stock Close Price with Moving Averages',
                            xaxis=dict(title='Date', rangeslider=dict(visible=True), type="date"),
                            yaxis=dict(title='Price'),
                            autosize=True)

            st.plotly_chart(fig)
            
            #Moving Average Convergence Divergence(MACD)
            st.subheader('MACD')
            macd = MACD(stck)
            fig = go.Figure()

            # Plot MACD
            fig.add_trace(go.Scatter(x=macd.index, y=macd['MACD'], mode='lines', name='MACD',line=dict(color='blue')))

            # Plot Signal Line
            fig.add_trace(go.Scatter(x=macd.index, y=macd['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red')))
            # Plot MACD Histogram
            fig.add_trace(go.Bar(x=macd.index, y=macd['MACD_Histogram'], marker_color='rgba(0, 0, 255, 0.5)', name='MACD Histogram'))
            fig.update_layout(title='MACD for {}'.format(ticker),
                            xaxis_title='Date',
                            yaxis_title='MACD',
                            autosize=True)

            st.plotly_chart(fig)
            
            #Relative Strength Index
            st.subheader('Relative Strength Index')
            # Calculate RSI
            stck = RSI(stck)
            trace_rsi = go.Scatter(x=stck.index, y=stck['RSI'], mode='lines', name='RSI', line=dict(color='orange'))
            layout = go.Layout(title=f'RSI for {ticker}',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='RSI'),
                            autosize=True)
            fig = go.Figure(data=[trace_rsi], layout=layout)
            st.plotly_chart(fig)

            #lstm model preparation
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            df = stck[['Close']]
            lookback = 50
            shifted_df = prepared_data(df, lookback)
            matrix_data = shifted_df.to_numpy()
            #preprocessing(Normalisation)
            from sklearn.preprocessing import MinMaxScaler
            try:
		scaler = MinMaxScaler(feature_range=(-1,1))
		matrix_data = scaler.fit_transform(matrix_data)
	    except ValueError:
		st.warning("An error occured,please check your input start date.Make sure that it is in past.")
		st.stop()
            # scaler = MinMaxScaler(feature_range=(-1,1))
            # matrix_data = scaler.fit_transform(matrix_data)
            #getting the required column alone for training
            x = matrix_data[:,1:]
            y = matrix_data[:,0]
            x = dc(np.flip(x,axis = 1))
            #splitting training and test data
            split_index = int(len(x)*0.95)
            x_train = x[:split_index]
            x_test = x[split_index:]

            y_train = y[:split_index]
            y_test = y[split_index:]
            #reshaping for NN
            x_train = x_train.reshape((-1,lookback,1))
            x_test = x_test.reshape((-1,lookback,1))

            y_train = y_train.reshape((-1,1))
            y_test = y_test.reshape((-1,1))
            #changing to tensor
            x_train = torch.tensor(x_train).float()
            x_test = torch.tensor(x_test).float()

            y_train = torch.tensor(y_train).float()
            y_test = torch.tensor(y_test).float()

            #timeseries dataset
            train_dataset = TimeSeriesDataset(x_train,y_train)
            test_dataset = TimeSeriesDataset(x_test,y_test)

            from torch.utils.data import DataLoader

            batch_size = 20

            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

            for _, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to(device),batch[1].to(device)
                print(x_batch.shape, y_batch.shape)
                break
            #loadinng model
            model = LSTM(1, 4, 1)
            model.to(device)
         #   model
            #training
            learning_rate = 0.01
            num_epochs = 10
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

            for epoch in range(num_epochs):
                train_one_epoch()
                validate_one_epoch()

            #predicted values
            with torch.no_grad():
                predicted_train = model(x_train.to(device)).to('cpu').numpy()
                predicted_test = model(x_test.to(device)).to('cpu').numpy()


            st.subheader('Prediction using historical data')
            #inverse scaling to original data scale
            prediction_train = inverse_scaling(predicted_train, x_train, lookback)
            prediction_test = inverse_scaling(predicted_test, x_test, lookback)
            new_y_train = inverse_data(y_train, x_train, lookback)
            new_y_test = inverse_data(y_test,x_test, lookback)
            # Plotting Predicted vs Actual for Training Data
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(x=np.arange(len(prediction_train)), y=prediction_train, mode='lines', name='Predicted'))
            fig_train.add_trace(go.Scatter(x=np.arange(len(new_y_train)), y=new_y_train, mode='lines', name='Actual'))
            fig_train.update_layout(title='Predicted vs Actual for Training Data', xaxis_title='Index', yaxis_title='Value')
            st.plotly_chart(fig_train)

            # Plotting Predicted vs Actual for Testing Data
            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=np.arange(len(prediction_test)), y=prediction_test, mode='lines', name='Predicted'))
            fig_test.add_trace(go.Scatter(x=np.arange(len(new_y_test)), y=new_y_test, mode='lines', name='Actual'))
            fig_test.update_layout(title='Predicted vs Actual for Testing Data', xaxis_title='Index', yaxis_title='Value')
            st.plotly_chart(fig_test)

        

else:
    pass


