import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Stock Prediction (LSTM)", layout="wide")
st.title("📈 Stock Price Prediction App (LSTM)")
st.write("Upload your Investing.com AAPL CSV and run prediction.")

uploaded = st.file_uploader("Upload CSV (Investing.com format)", type=["csv"])

SEQ_LEN = st.slider("Sequence Length (days)", 30, 120, 60)
epochs = st.slider("Epochs", 1, 50, 15)
batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

def clean_investing_csv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Investing.com
    if "Price" in df.columns and "Close" not in df.columns:
        df.rename(columns={"Price": "Close"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Close"] = df["Close"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df[["Date", "Close"]].dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def build_lstm(seq_len: int) -> Sequential:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if uploaded:
    raw = pd.read_csv(uploaded)
    df = clean_investing_csv(raw)

    st.success(f"Dataset loaded successfully. Rows after cleaning: {len(df)}")
    st.dataframe(df.head())

    st.subheader("Closing Price Trend")
    fig1 = plt.figure(figsize=(12,4))
    plt.plot(df["Date"], df["Close"])
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    st.pyplot(fig1)

    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    if len(scaled) <= SEQ_LEN:
        st.error("Dataset is too small for the selected sequence length. Reduce SEQ_LEN.")
        st.stop()

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, SEQ_LEN, 1)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = df["Date"].iloc[SEQ_LEN + split:].reset_index(drop=True)

    if st.button("Train & Predict"):
        tf.random.set_seed(42)
        model = build_lstm(SEQ_LEN)

        with st.spinner("Training model..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

        st.subheader("Training vs Validation Loss (MSE)")
        fig2 = plt.figure(figsize=(10,4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig2)

        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.subheader("Actual vs Predicted (Test)")
        fig3 = plt.figure(figsize=(12,4))
        plt.plot(test_dates, actual, label="Actual")
        plt.plot(test_dates, pred, label="Predicted (Test)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig3)

               # Next-day prediction
        last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        next_scaled = model.predict(last_seq, verbose=0)
        next_close = float(scaler.inverse_transform(next_scaled)[0, 0])

        last_date = pd.to_datetime(df["Date"].iloc[-1])
        next_date = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=1)[0]

        
        last_pred = float(np.ravel(pred)[-1])

        st.subheader("Next-Day Forecast")
        fig4 = plt.figure(figsize=(12,4))
        plt.plot(test_dates, np.ravel(actual), label="Actual")
        plt.plot(test_dates, np.ravel(pred), label="Predicted (Test)")

        plt.plot(
            [pd.to_datetime(test_dates.iloc[-1]), next_date],
            [last_pred, next_close],
            color="green",
            linewidth=2,
            label="Next Day Prediction"
        )

        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig4)

        st.info(f"Next day predicted close: {next_close:.2f} | Prediction date: {next_date.date()}")

else:
    st.warning("Please upload your CSV file to continue.")
