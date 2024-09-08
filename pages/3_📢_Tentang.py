import streamlit as st
import pandas as pd
import yfinance as yf

def get_gold_data():
    # Mendapatkan data harga emas
    gold_info = yf.Ticker('GC=F')  # Kode untuk futures emas
    if gold_info:
        gold_name = gold_info.info.get('shortName', 'Gold')
        gold_symbol = 'GC=F'
        
        # Mendapatkan harga emas saat ini
        gold_price = get_gold_price(gold_info)

        return pd.DataFrame([{
            'Nama': gold_name,
            'Kode': gold_symbol,
            'Harga': gold_price,
        }])

    return pd.DataFrame()

def get_gold_price(gold_info):
    try:
        # Mendapatkan harga emas dari yfinance
        gold_data = gold_info.history(period="1d")
        if not gold_data.empty:
            return gold_data['Close'][0]
    except Exception as e:
        print(f"Error getting price for {gold_info}: {e}")
    return None

def main():
    st.title('Prediksi Harga Emas Menggunakan LSTM 📈')
    
    st.write("""
    **Apa Itu Emas?**
    Emas adalah logam mulia yang telah lama dianggap sebagai simbol kekayaan dan kemakmuran. Selain digunakan sebagai perhiasan, emas juga digunakan sebagai alat investasi, cadangan devisa, dan pelindung nilai. Nilainya yang stabil menjadikannya aset yang sangat dihargai, terutama di saat ketidakpastian ekonomi. Banyak negara menyimpan emas dalam jumlah besar sebagai bagian dari cadangan nasional mereka.
    """)

    st.write("""
    **Mengapa Harga Emas Penting?**
    Harga emas memainkan peran penting dalam ekonomi global. Sebagai "safe haven", emas sering kali digunakan oleh investor untuk melindungi kekayaan mereka dari inflasi dan gejolak pasar lainnya. Fluktuasi harga emas dapat dipengaruhi oleh berbagai faktor seperti tingkat suku bunga, nilai tukar mata uang, dan kebijakan moneter. Oleh karena itu, memprediksi harga emas menjadi krusial bagi investor, bank sentral, dan pemerintah.
    """)

    st.write("""
    **Prediksi Harga Emas Menggunakan LSTM**
    Dalam konteks prediksi harga emas, model Long Short Term Memory (LSTM) menjadi salah satu pilihan yang kuat. LSTM adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk mengenali pola jangka panjang dalam data berurutan, seperti data harga historis emas. Dengan menggunakan LSTM, kita dapat memprediksi pergerakan harga emas di masa depan berdasarkan tren masa lalu, yang memberikan wawasan yang lebih baik untuk pengambilan keputusan investasi.
    """)

    # Mendapatkan data harga emas
    gold_df = get_gold_data()

    # Menampilkan tabel harga emas
    st.table(gold_df)
    
    st.write('Data harga emas diambil dari Yahoo Finance: https://finance.yahoo.com/')
    
if __name__ == '__main__':
    main()
