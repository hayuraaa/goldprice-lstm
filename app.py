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
import time
import matplotlib.pyplot as plt

def main():
    print("Forex Currency Price Prediction App")
    print("This is a prediction system using the LSTM algorithm with parameter input.")
    print("Hyperparameters cannot be directly obtained from the data and need to be initialized before the training process begins.")
    print("Wise decisions in determining these values can greatly impact the performance and generalization capability of the model.")
    
    stock_symbol = input("Enter Currency Pair (e.g., JPY=X, EURUSD=X, etc.): ")
    start_date = input("Enter Start Date (YYYY-MM-DD): ")
    end_date = input("Enter End Date (YYYY-MM-DD): ")
    price_type = input("Select Price Type (Close, Open, High, Low): ")

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

    epoch_options = [50, 100]
    batch_size_options = [16, 32]

    results = []
    for e in epoch_options:
        for b in batch_size_options:
            print(f"Training with Epochs: {e}, Batch Size: {b}")
            start_time = time.time()

            best_params_lstm, history_lstm, y_pred_lstm, y_test_orig_lstm, model_summary = run_optimization({'units': 100, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'epochs': e, 'batch_size': b}, 'lstm', X_train, y_train, X_test, y_test, scaler)

            end_time = time.time()
            duration = end_time - start_time

            print(f"Total time taken for prediction: {duration:.2f} seconds")

            mse, rmse, mape = calculate_metrics(y_test_orig_lstm, y_pred_lstm)
            print(f"MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")

            results.append((e, b, mse, rmse, mape, history_lstm, y_test_orig_lstm, y_pred_lstm))

    print("Summary of All Results")
    results_df = pd.DataFrame(results, columns=["Epochs", "Batch Size", "MSE", "RMSE", "MAPE", "History", "Y_Test_Orig", "Y_Pred"])
    print(results_df[["Epochs", "Batch Size", "MSE", "RMSE", "MAPE"]])

    best_params = results_df.loc[results_df['RMSE'].idxmin()]
    print("Best Parameters Based on RMSE")
    print(f"Epochs: {best_params['Epochs']}")
    print(f"Batch Size: {best_params['Batch Size']}")
    print(f"RMSE: {best_params['RMSE']}")

    # Visualize best results
    visualize_predictions(data, train_size, n_steps, best_params['Y_Test_Orig'], best_params['Y_Pred'])

    # Display loss graph for the best model
    display_loss_graph(best_params['History'])

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
                              verbose=0,  # No verbose
                              validation_split=0.1)

    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the model summary as a string
    stringlist = []
    final_model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    return best_params, history, y_pred, y_test_orig, model_summary

def build_model(params, X_train):
    model = Sequential()
    model.add(LSTM(units=params['units'], return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=params['units'], activation='tanh'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    return model

def calculate_metrics(y_test_orig, y_pred):
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100
    return mse, rmse, mape

def display_loss_graph(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
    plt.figure(figsize=(14, 7))

    plt.plot(data.index[:train_size + n_steps],
             data['Close'].values[:train_size + n_steps],
             color='gray', label='Training Data')

    plt.plot(data.index[train_size + n_steps:],
             y_test_orig.flatten(),
             color='blue', label='Actual Stock Prices')

    plt.plot(data.index[train_size + n_steps:],
             y_pred.flatten(),
             color='red', label='Predicted Stock Prices')

    plt.title('Forex Currency Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
