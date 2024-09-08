import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math
from warnings import simplefilter
import plotly.graph_objects as go
import time

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

def main():
    st.title("Sistem Prediksi Harga Emas")
    st.write("Halaman ini mencari parameter terbaik dari 'Epoch' dan 'Batch Size'. Proses prediksi akan memakan waktu yang cukup lama, Silahkan tunggu 5-10 Menit...")

    st.header("Data Download")
    stock_symbol = st.selectbox("Kode Emas:", ["GC=F"])
    start_date = st.date_input("Start Date", pd.to_datetime("2022-08-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-08-01"))
    price_type = st.selectbox("Select Price Type:", ["Close", "Open", "High", "Low"])

    data = yf.download(stock_symbol, start=start_date, end=end_date)

    close_prices = data[price_type].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    n_steps = 30
    X, y = prepare_data(scaled_data, n_steps)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    epoch_options = [50, 100, 150, 200]
    batch_size_options = [16, 32, 64, 128]

    if "best_params_lstm" not in st.session_state:
        st.session_state["best_params_lstm"] = None
        st.session_state["best_model"] = None
        st.session_state["best_y_pred_lstm"] = None
        st.session_state["best_y_test_orig_lstm"] = None
        st.session_state["best_history_lstm"] = None

    if st.button("Apply Hyperparameters", key='apply_button'):
        st.sidebar.text("Applying hyperparameters...")
        results = []
        best_rmse = float('inf')

        for e in epoch_options:
            for b in batch_size_options:
                st.write(f"Training with Epochs: {e}, Batch Size: {b}")
                start_time = time.time()

                params = {'units': 100, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'epochs': e, 'batch_size': b}
                _, history_lstm, y_pred_lstm, y_test_orig_lstm, model_summary = run_optimization(params, 'lstm', X_train, y_train, X_test, y_test, scaler)

                end_time = time.time()
                duration = end_time - start_time

                mse, rmse, mape = evaluate_results(y_test_orig_lstm, y_pred_lstm)

                results.append((e, b, mse, rmse, mape, history_lstm))

                if rmse < best_rmse:
                    best_rmse = rmse
                    st.session_state["best_params_lstm"] = params
                    st.session_state["best_y_pred_lstm"] = y_pred_lstm
                    st.session_state["best_y_test_orig_lstm"] = y_test_orig_lstm
                    st.session_state["best_history_lstm"] = history_lstm
                    st.session_state["best_model"] = build_model(params, X_train)
                    st.session_state["best_model"].fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

        st.header("Summary of All Results")
        results_df = pd.DataFrame(results, columns=["Epochs", "Batch Size", "MSE", "RMSE", "MAPE", "History"])
        st.write(results_df.drop(columns=["History"]))

        best_params = results_df.loc[results_df['RMSE'].idxmin()]
        st.header("Best Parameters and Results Based on RMSE")
        st.write(f"Epochs: {best_params['Epochs']}")
        st.write(f"Batch Size: {best_params['Batch Size']}")
        st.write(f"Best RMSE: {best_params['RMSE']}")

        # Visualize the best prediction and plot the loss
        if st.session_state["best_y_pred_lstm"] is not None:
            st.header("Loss Plot for Best Model")
            st.line_chart(pd.DataFrame({'Train Loss': st.session_state["best_history_lstm"].history['loss'], 'Validation Loss': st.session_state["best_history_lstm"].history['val_loss']}))

            visualize_predictions(data, train_size, n_steps, st.session_state["best_y_test_orig_lstm"], st.session_state["best_y_pred_lstm"])

    st.header("Future Predictions")
    forecast_days = st.slider("Select Number of Days to Forecast:", 1, 60, 7)
    if st.button("Predict Future Prices"):
        if st.session_state["best_model"] is not None:
            future_predictions = forecast_future_prices(st.session_state["best_model"], scaler, scaled_data, n_steps, forecast_days)
            st.header(f"Forecasting for Next {forecast_days} Days")
            visualize_future_predictions(data, future_predictions, forecast_days)
        else:
            st.error("No best model found. Please run hyperparameter tuning first.")

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def run_optimization(space, model_type, X_train, y_train, X_test, y_test, scaler):
    final_model = build_model(space, X_train)

    history = final_model.fit(X_train, y_train,
                              epochs=space['epochs'],
                              batch_size=space['batch_size'],
                              verbose=2,
                              validation_split=0.1,)

    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the model summary as a string
    stringlist = []
    final_model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    return space, history, y_pred, y_test_orig, model_summary

def build_model(params, X_train):
    model = Sequential()
    model.add(LSTM(units=params['units'], return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=params['units'], activation='tanh'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    return model

def evaluate_results(y_test_orig, y_pred):
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    return mse, rmse, mape

def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index[:train_size + n_steps],
                             y=data['Close'].values[:train_size + n_steps],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig.update_layout(title="Gold Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template='plotly_dark')

    st.plotly_chart(fig)

def forecast_future_prices(model, scaler, scaled_data, n_steps, forecast_days):
    last_sequence = scaled_data[-n_steps:]
    future_predictions = []

    for _ in range(forecast_days):
        last_sequence = last_sequence.reshape((1, n_steps, 1))
        predicted_price = model.predict(last_sequence)
        future_predictions.append(predicted_price[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], predicted_price.reshape((1, 1, 1)), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

def visualize_future_predictions(data, future_predictions, forecast_days):
    last_date = data.index[-1]
    last_close_price = data['Close'].values[-1]
    future_predictions = np.insert(future_predictions, 0, last_close_price)
    future_dates = pd.date_range(last_date, periods=forecast_days + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index,
                             y=data['Close'].values,
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=future_dates,
                             y=future_predictions,
                             mode='lines',
                             name="Future Predictions",
                             line=dict(color='green')))

    fig.update_layout(title="Prediksi Masa Depan",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
