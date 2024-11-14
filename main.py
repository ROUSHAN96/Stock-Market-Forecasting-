import yfinance as yf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go 
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import streamlit as st
from statsmodels.tsa.stattools import adfuller

app_name = 'Stock Market Forcasting App'
st.title(app_name)
st.subheader('This app is created to forcasting the stock market price of selected companys.')
st.image("https://akm-img-a-in.tosshub.com/businesstoday/images/story/202409/66ed09ce60b96-stock-market-fpi-inflows-to-india-are-likely-to-accelerate-if-the-fed-easing-cycle-triggers-a-risk--20361334-16x9.png?size=948:533")

st.sidebar.header('Select the parameters from below')
start_date = st.sidebar.date_input('Start date', date(2020,1,1))
end_date = st.sidebar.date_input('End date', date(2020,12,31))

ticker_list = ["AAPL","MSFT","META","GOOG","TSLA","NVDA","NFLX","ADBE","PYPL"]
ticker = st.sidebar.selectbox('Select the company',ticker_list)

data = yf.download(ticker,start=start_date,end=end_date)
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from',start_date,'to',end_date)
st.write(data)

st.header('Data visualisation')
st.subheader('Plot of the data')
st.write('NOTE : Select your specific date range on the slidebar,or zoom in on the plot and select your specific column')
fig = px.line(data, x='Date', y=data.columns, title='Closing prices of the stock',width=1000,height=600)
st.plotly_chart(fig)

column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

st.header('Is data Stationary?')
st.write('NOTE : If p-value is less than 0.05, then data is Stationary')
st.write(adfuller(data[column])[1]<0.05)

st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())

st.write("## Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

p = st.slider('Select the value of p',0,5,2)
d = st.slider('Select the value of d',0,5,2)
q = st.slider('Select the value of q',0,5,2)
seasonal_order = st.number_input('Select the value of seasonal p',0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

st.header('Model Summary')
st.write(model.summary())
st.write("---")

st.write("<p style='color:green; font-size:50px; font-weight:bold;'>Forecasting the data</p>",unsafe_allow_html=True)
forecast_period = st.number_input('Select the number of days to forecast',1,365,10)

predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean
# st.write(predictions)

predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace=True)
st.write("Predictions",predictions)
st.write("Actual Date", data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
st.plotly_chart(fig)

show_plots = False
if st.button('Show Sepetrate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width=1200,height=400,labels={'x': 'Date', 'y': 'Prices'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], title='predicted', width=1200,height=400,labels={'x': 'Date', 'y': 'Prices'}).update_traces(line_color='Red'))
        show_plots = True
    else:
        show_plots = False    

hide_plots = False
if st.button("Hide Seperate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plots = False
st.write("---")      
# sarema model  or  arima model  