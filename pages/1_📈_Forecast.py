import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from warnings import simplefilter
import plotly.graph_objects as go
import time

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Streamlit app
def main():
    st.title("Sistem Prediksi Harga Emas")
    st.write("Web ini adalah sistem prediksi yang menggunakan algoritma LSTM. Silahkan input parameter dan tanggal prediksi. Keputusan yang bijak dalam menentukan nilai-nilai tersebut dapat sangat mempengaruhi performa dan kemampuan generalisasi model, yang bertujuan untuk mencapai performa yang optimal dalam memprediksi harga Emas.")

    stock_symbol = st.selectbox("Kode Emas:", ["GC=F"])
    start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2022-08-01"))
    end_date = st.date_input("Tanggal Selesai", pd.to_datetime("2024-08-01"))
    price_type = st.selectbox("Pilih Field:", ["Close", "Open", "High", "Low"])

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

    st.header("Setting Parameter")
    epochs = st.selectbox("Pilih Nilai Epochs:", [50, 100, 150, 200], index=0)
    batch_size = st.selectbox("Pilih Nilai Batch Size:", [16, 32, 64, 128], index=0)

    forecast_future = st.checkbox("Prediksi Masa Depan")
    if forecast_future:
        forecast_days = st.slider("Atur Hari Prediksi Masa Depan:", 1, 60, 7)

    if st.button("Apply Forecasti", key='apply_button'):
        st.text("Applying...")
        space = {
            'units': 100,
            'dropout_rate': 0.2,  # Fixed value
            'learning_rate': 0.001,  # Fixed value
            'epochs': epochs,
            'batch_size': batch_size
        }

        start_time = time.time()

        # Correctly get the final model
        best_params_lstm, history_lstm, y_pred_lstm, y_test_orig_lstm, final_model = run_optimization(space, 'lstm', X_train, y_train, X_test, y_test, scaler)

        end_time = time.time()
        duration = end_time - start_time
        
        st.write(f"Total time taken for prediction: {duration:.2f} seconds")

        # 1. Result Evaluasi Model
        st.header("Results for LSTM Model")
        display_results(history_lstm, y_test_orig_lstm, y_pred_lstm, data, train_size, n_steps)

        # 2. Grafik Visualisasi Prediksi Emas (LSTM)
        st.header("Visualisasi Prediksi Emas (LSTM)")
        visualize_predictions(data, train_size, n_steps, y_test_orig_lstm, y_pred_lstm)

        # 3. Tabel Hasil Prediksi Emas (LSTM)
        st.write("### Tabel Hasil Prediksi Emas (LSTM)")
        display_results_table(data, train_size, n_steps, y_test_orig_lstm, y_pred_lstm)

        # 4. Grafik Prediksi Masa Depan
        if forecast_future:
            future_predictions = forecast_future_prices(final_model, scaler, scaled_data, n_steps, forecast_days)
            st.header(f"Forecasting for Next {forecast_days} Days")
            visualize_future_predictions(data, future_predictions, forecast_days)
            
        # 5. Grafik Evaluation Loss
        st.write("### Grafik Evaluation Loss")
        plot_loss_chart(history_lstm)

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def run_optimization(space, model_type, X_train, y_train, X_test, y_test, scaler):
    best_params = space  # Directly use space as best_params since we are not using optimization
    final_model = build_model(best_params, X_train)

    history = final_model.fit(X_train, y_train,
                              epochs=best_params['epochs'],
                              batch_size=best_params['batch_size'],
                              verbose=2,
                              validation_split=0.1,)

    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    return best_params, history, y_pred, y_test_orig, final_model

def build_model(params, X_train):
    model = Sequential()
    model.add(LSTM(units=params['units'], return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=params['units'], activation='tanh'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    return model

#Result Evaluasi Prediksi
def display_results(history, y_test_orig, y_pred, data, train_size, n_steps):
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    st.write("Mean Absolute Error (MAE):", f"{mae:.2f}")
    st.write("Root Mean Squared Error (RMSE):", f"{rmse:.2f}")
    st.write("Mean Absolute Percentage Error (MAPE):", f"{mape:.2f}%")

def display_results_table(data, train_size, n_steps, y_test_orig, y_pred):
    dates = data.index[train_size + n_steps:]
    dates = dates.strftime('%Y-%m-%d')

    if len(dates) != len(y_test_orig) or len(dates) != len(y_pred):
        st.error("Mismatch in lengths of dates, actual values, and predictions.")
        return

    table_data = pd.DataFrame({
        "Tanggal": dates,
        "Data Aktual": [f"{value:.2f}" for value in y_test_orig.flatten()],
        "Prediksi": [f"{value:.2f}" for value in y_pred.flatten()],
        "MAE": [f"{value:.2f}" for value in np.abs(y_test_orig.flatten() - y_pred.flatten())],
        "MAPE": [f"{value:.2f}" for value in np.abs((y_test_orig.flatten() - y_pred.flatten()) / y_test_orig.flatten()) * 100]
    })
    
    st.table(table_data)

#Visual Grafik Prediksi Data Uji
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

    fig.update_layout(title="Prediksi Harga Emas",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template='plotly_dark')

    st.plotly_chart(fig)

#Fungsi Prediksi Masa Depan
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


#Visual Grafik Prediksi Masa Depan
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
    
    # Buat tabel data prediksi masa depan
    future_table = pd.DataFrame({
        "Tanggal": future_dates.strftime('%Y-%m-%d'),
        "Harga Prediksi": [f"{value:.2f}" for value in future_predictions]
    })

    st.write("### Tabel Prediksi Harga Masa Depan")
    st.table(future_table)

def plot_loss_chart(history):
    loss_data = pd.DataFrame({'Train Loss': history.history['loss'], 'Validation Loss': history.history['val_loss']})
    st.line_chart(loss_data)

if __name__ == "__main__":
    main()
