import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import math

st.set_page_config(page_title="Stock LSTM Predictor", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #3e4e88;'>📈 Stock Market Prediction with LSTM</h1>
    <p style='text-align: center; font-size:18px;'>Predict stock prices using LSTM and visualize insights easily.</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
    future_days = st.slider("Days to Predict into Future", 1, 30, 7)
    run_btn = st.button("🚀 Run Prediction")

if run_btn:
    st.info(f"Fetching data for `{ticker}` from Yahoo Finance...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the given ticker and date range.")
        st.stop()

    data = df[['Close']].values
    st.success("Data loaded successfully!")

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    def create_sequences(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.info("Training the model...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    st.success("Model training complete!")

    # Predictions on test data
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))

    # Future prediction
    st.info(f"Predicting next {future_days} days...")
    last_60_days = scaled_data[-60:]
    future_preds = []
    current_input = last_60_days.reshape(1, -1, 1)

    for _ in range(future_days):
        next_pred = model.predict(current_input)[0, 0]
        future_preds.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame(future_preds, index=future_dates, columns=["Predicted Price"])

    # Display stock information
    st.subheader("📊 Stock Info")
    st.write(df.tail())

    latest_close = float(df['Close'].values[-1])
    st.metric(label="Latest Close Price", value=f"${latest_close:.2f}")
    st.metric(label="Model RMSE", value=f"{rmse:.4f}")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Closing Price", "🟢 Moving Average", "📊 Actual vs Predicted", "🔮 Future Prediction"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df.index, df['Close'], label='Closing Price')
        ax1.set_title(f"{ticker} Closing Price Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        st.pyplot(fig1)

    with tab2:
        ma_50 = df['Close'].rolling(window=50).mean()
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df['Close'], label='Closing Price')
        ax2.plot(ma_50, label='50-day MA', color='orange')
        ax2.set_title(f"{ticker} Moving Average vs Closing Price")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price ($)")
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(actual_prices, label="Actual Price")
        ax3.plot(predicted_prices, label="Predicted Price")
        ax3.set_title(f"{ticker} Actual vs Predicted Price")
        ax3.legend()
        st.pyplot(fig3)

    with tab4:
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(df.index, df['Close'], label="Historical Price")
        ax4.plot(future_df.index, future_df['Predicted Price'], label="Future Prediction", color='red')
        ax4.set_title(f"{ticker} Future {future_days} Days Prediction")
        ax4.legend()
        st.pyplot(fig4)
        st.dataframe(future_df.style.format({"Predicted Price": "${:.2f}"}))
        csv = future_df.to_csv().encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name=f"{ticker}_future_predictions.csv")

