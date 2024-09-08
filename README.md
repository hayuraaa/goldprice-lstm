# Prediksi Emas Menggunakan LSTM

Aplikasi ini memprediksi harga emas menggunakan model Long Short-Term Memory (LSTM) yang diimplementasikan dengan TensorFlow. Aplikasi ini dibuat menggunakan framework **Streamlit** untuk memudahkan pengguna dalam berinteraksi dengan model.

## Fitur
- Prediksi harga emas berdasarkan data historis dari Yahoo Finance.
- Visualisasi data historis dan hasil prediksi menggunakan **Plotly**.
- Model LSTM untuk prediksi harga emas.

## Pustaka yang Digunakan
- **Streamlit**: Untuk membangun aplikasi web.
- **yfinance**: Untuk mengambil data harga emas historis dari Yahoo Finance.
- **NumPy**: Untuk operasi numerik.
- **Pandas**: Untuk manipulasi data.
- **Scikit-Learn**: Untuk preprocessing data.
- **TensorFlow==2.15.0**: Untuk membangun dan melatih model LSTM.
- **Plotly**: Untuk visualisasi data.
- **Datetime**: Untuk manipulasi tanggal.
- **Math**: Untuk operasi matematika dasar.
- **Time**: Untuk pengukuran waktu.

## Persyaratan Sistem
- Python 3.7 atau lebih baru
- Koneksi internet untuk mengunduh data dari Yahoo Finance

## Instalasi

1. Clone repositori ini ke dalam direktori lokal:

   ```bash
   git clone https://github.com/hayuraaa/goldprice-lstm.git
   cd goldprice-lstm

2. python -m venv env
source env/bin/activate  # Untuk Mac/Linux
.\env\Scripts\activate   # Untuk Windows

3. Instal semua dependensi yang diperlukan:

```bash
pip install streamlit yfinance numpy pandas scikit-learn tensorflow==2.15.0 plotly
streamlit run 1_Home.py

