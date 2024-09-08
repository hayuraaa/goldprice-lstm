import streamlit as st

# Fungsi untuk menambahkan baris baru
def add_new_lines(n=1):
    for _ in range(n):
        st.write("")

def main():
    # Header aplikasi
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>Sistem Prediksi Harga Emas</h1>", unsafe_allow_html=True)
    add_new_lines(2)
    
    st.markdown("""
    <p style='text-align: justify;'>
    Selamat datang di aplikasi prediksi harga emas yang canggih! Aplikasi ini memanfaatkan <span style="color:#FF6347;"><b>Algoritma Long Short Term Memory (LSTM)</b></span> untuk membantu Anda membuat keputusan yang lebih cerdas dalam perdagangan emas. Sistem ini dirancang untuk memberikan prediksi yang <span style="color:#32CD32;"><b>akurat</b></span> dan <span style="color:#32CD32;"><b>dapat diandalkan</b></span>, sehingga dapat membantu Anda dalam mengambil <span style="color:#FF4500;"><b>keputusan investasi yang lebih baik</b></span>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Bagian Gambaran Umum
    add_new_lines()
    st.markdown("<h2 style='text-align: center;'>Gambaran Umum Proses</h2>", unsafe_allow_html=True)
    add_new_lines()
    
    st.markdown("""
    <p style='text-align: justify;'>
    Dalam membangun model prediksi, terdapat beberapa langkah penting yang harus dilakukan. Berikut adalah tahapan utama dalam proses <span style="color:#1E90FF;"><b>Machine Learning</b></span>:
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **<span style="color:#FF8C00;">Pengumpulan Data</span>**: Mengambil data dari berbagai sumber seperti pustaka yfinance, file CSV, API, dan lainnya.
    - **<span style="color:#FF8C00;">Pembersihan Data</span>**: Menghapus data duplikat, menangani nilai yang hilang, dan memastikan data dalam kondisi bersih.
    - **<span style="color:#FF8C00;">Praproses Data</span>**: Mengubah data menjadi format yang sesuai untuk analisis, termasuk penanganan fitur kategorikal dan numerik.
    - **<span style="color:#FF8C00;">Rekayasa Fitur</span>**: Memanipulasi fitur untuk meningkatkan akurasi model, termasuk seleksi dan transformasi fitur.
    - **<span style="color:#FF8C00;">Pemecahan Data</span>**: Membagi data menjadi set pelatihan, validasi, dan pengujian.
    - **<span style="color:#FF8C00;">Pembangunan Model</span>**: Aplikasi ini menggunakan model <span style="color:#FF6347;"><b>LSTM</b></span> untuk prediksi.
    - **<span style="color:#FF8C00;">Evaluasi Model</span>**: Mengukur performa model menggunakan metrik seperti <span style="color:#1E90FF;"><b>MAPE, MSE,</b></span> dan <span style="color:#1E90FF;"><b>RMSE</b></span>.
    """, unsafe_allow_html=True)

    add_new_lines()
    
    st.markdown("""
    Saat membangun model, pengguna dapat memasukkan nilai untuk masing-masing hiperparameter. Berikut adalah beberapa hiperparameter penting yang dapat diatur:
    
    - **<span style="color:#FFD700;">Time Steps</span>**: Menentukan jumlah data latih yang digunakan untuk memprediksi harga emas di masa depan.
    - **<span style="color:#FFD700;">Units</span>**: Menentukan jumlah lapisan LSTM yang digunakan dalam jaringan saraf.
    - **<span style="color:#FFD700;">Dropout</span>**: Teknik untuk mengurangi overfitting dengan menonaktifkan beberapa neuron selama pelatihan.
    - **<span style="color:#FFD700;">Learning Rate</span>**: Mengatur kecepatan perubahan bobot selama proses pelatihan.
    - **<span style="color:#FFD700;">Epochs</span>**: Menentukan jumlah iterasi program dalam mengolah dataset latih.
    - **<span style="color:#FFD700;">Batch Size</span>**: Menentukan jumlah sampel yang diproses dalam satu batch pada jaringan saraf.
    """, unsafe_allow_html=True)
    
    add_new_lines(2)
    
    # Pesan Akhir
    st.markdown("""
    <p style='text-align: center;'>
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
