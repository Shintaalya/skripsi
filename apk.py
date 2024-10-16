import streamlit as st #import modul Streamlit yang digunakan untuk membangun antarmuka pengguna
import pandas as pd #import modul pandas yang digunakan untuk analisis data
import numpy as np #import modul numpy
import pickle #import modul pickle yang digunakan untuk serialisasi dan deserialisasi objek Python
from PIL import Image  #import kelas Image dari modul PIL (Python Imaging Library) yang digunakan untuk memanipulasi gambar
from streamlit_option_menu import option_menu  #pustaka yang memberikan fungsi tambahan untuk membuat menu pilihan dengan Streamlit
from linear_regression_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error #untuk menghitung dan mengukur tingkat kesalahan (eror) prediksi Anda.

with st.sidebar: #Fungsi tersebut menghasilkan objek pilihan menu
    choose = option_menu("Linear Regression (Polynomial)", ["Home", "Dataset", "Prepocessing", "Predict", "Help"],
                             icons=['house', 'table', 'cloud-upload', 'boxes','check2-circle'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding":"5!important", "background-color": "10A19D"}, #Mengatur tampilan kontainer (wadah) dari menu pilihan
            "icon": {"color": "blue", "font-size": "25px"},  #Mengatur tampilan ikon dalam menu pilihan. Properti "color" mengatur warna ikon. Properti "font-size" mengatur ukuran font ikon.
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}, #Mengatur tampilan tautan dalam menu pilihan
            "nav-link-selected": {"background-color": "#00FFFF"}, #Mengatur tampilan tautan yang dipilih dalam menu pilihan
        }
        )
if choose=='Home':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    logo = Image.open('makam1.jpg')

    st.image(logo, use_column_width=True, caption='Rumah di Jaksel') #mengatur lebar gambar agar sesuai dengan lebar kolom
    st.write('<p style = "text-align: justify;">Rumah merupakan salah satu kebutuhan pokok manusia, selain sandang dan pangan, rumah juga berfungsi sebagai tempat tinggal dan berfungsi untuk melindungi dari gangguan iklim dan makhluk hidup lainnya. Tak kalah buruknya dengan emas, rumah pun bisa dijadikan sebagai sarana investasi masa depan karena pergerakan harga yang berubah dari waktu ke waktu, dan semakin banyak orang yang membutuhkan hunian selain kedekatan dengan tempat kerja, pusat perkantoran dan pusat bisnis, transportasi. dll tentunya akan cepat mempengaruhi harga rumah tersebut.</p>', unsafe_allow_html = True)
    st.write('<p style = "text-align: justify;">Dalam proyek ini, kami mengembangkan sebuah sistem untuk memprediksi harga rumah berdasarkan parameter luas tanah dan luas bangunan, dan output yang dihasilkan adalah prediksi harga rumah. Kami menggunakan metode regresi linear dengan fitur ekspansi (expand feature) dan melatih model menggunakan metode Stochastic Gradient Descent. Untuk mengevaluasi model, kami menggunakan metrik MSE, RMSE, dan R (Square).Diharapkan dengan adanya sistem ini, dapat membantu dalam memprediksi harga rumah sesuai dengan luas tanah dan luas bangunan yang diinginkan.</p>', unsafe_allow_html = True)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("Dr. Indah Agustien Siradjuddin,S.Kom.,M.Kom")

elif choose=='Dataset':
    st.markdown('<h1 style = "text-align: center;"> Data Harga Rumah </h1>', unsafe_allow_html = True) #untuk menentukan apakah Streamlit harus mengizinkan HTML dalam teks Markdown
    df = pd.read_csv('https://raw.githubusercontent.com/Shintaalya/repo/main/HARGA%20RUMAH%20JAKSEL.csv')
    df
    st.markdown('<h1 style = "text-align: center;"> Fitur Dataset: </h1><ol type = "1" style = "text-align: justify; background-color: #00FFFF; padding: 30px; border-radius: 20px;"><p>Dataset ini diambil dari kaggle.com</p><li><i><b>HARGA</b></i> = harga dari rumah</li><li><i><b>LT</b></i> = Jumlah Luas Tanah</li><li><i><b>LB</b></i> = Jumlah Luas Bangunan</li><li><i><b>JKT</b></i> = Jumlah Kamar Tidur</li><li><i><b>JKM</b></i> = Jumlah Kamar Mandi</li><li><i><b>GRS</b></i> = Ada / Tidak Ada</li></ol>', unsafe_allow_html = True)

elif choose=='Prepocessing':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    st.write("Dari 7 Fitur")
    logo = Image.open('dataset.png')
    st.image(logo, caption='')
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("Diseleksi menjadi 2 Fitur")
    logo = Image.open('dataset2.png')
    st.image(logo, caption='')
    st.write("Berdasarkan garis lurus atau linearnya")
    logo = Image.open('dataset3.png')
    st.image(logo, caption='')
elif choose=='Predict':
    st.markdown('<h1 style = "text-align: center;"> Prediksi Harga Rumah</h1>', unsafe_allow_html = True)
    logo = Image.open('eror.png')
    st.image(logo, caption='')
    import urllib.request

    # Mendownload file model.pkl
    url = 'https://raw.githubusercontent.com/Shintaalya/repo/main/model.pkl'
    filename = 'model.pkl'  # Nama file yang akan disimpan secara sementara
    urllib.request.urlretrieve(url, filename)
    
    # Load the model
    with open('model.pkl','rb') as file:
        model_data = pickle.load(file) #Menggunakan modul pickle, data yang ada di dalam file 'model.pkl' dibaca dan dimuat ke dalam variabel model_data.
        model = model_data['model'] #Variabel model diisi dengan nilai dari kunci 'model' yang ada di dalam model_data
        X_train_expanded = model_data['X_train_expanded'] #Variabel X_train_expanded diisi dengan nilai dari kunci 'X_train_expanded' yang ada di dalam model_data.mendapatkan data latihan yang telah diperluas (expanded)
        y_train_mean = model_data['y_train_mean'] #untuk mendapatkan nilai rata-rata dari data latihan
        y_train_std = model_data['y_train_std'] #untuk mendapatkan standar deviasi dari data latih
        best_X_train = model_data['best_X_train'] #untuk mendapatkan data latih terbaik
        best_y_train = model_data['best_y_train'] #untuk mendapatkan target data latih terbaik
        coef = model_data['coef']
        intercept = model_data['intercept']

    # Function to normalize input data
    def normalize_input_data(data): #memiliki satu parameter data.akan menerima data input yang ingin dinormalisasi
        normalized_data = (data - np.mean(best_X_train, axis=0)) / np.std(best_X_train, axis=0) 
        #Normalisasi dengan mengurangi rata-rata dari best_X_train dari setiap nilai dalam data, dan kemudian membaginya dengan standar deviasi dari best_X_train
        return normalized_data #mengembalikan normalized_data sebagai hasil normalisasi.
    
    # Function to expand input features
    def expand_input_features(data): #data input yang ingin diperluas fiturnya.
        normalized_data = normalize_input_data(data) #Fungsi ini melakukan normalisasi terhadap data input dengan mengurangi rata-rata dari best_X_train dan membaginya dengan standar deviasi dari best_X_train
        expanded_data = model.expand_features(normalized_data, degree=2) #Fungsi ini mengembangkan fitur input dengan menggunakan ekspansi polinomial dengan derajat 2.
        return expanded_data #Hasil ekspansi fitur disimpan dalam variabel expanded_data kemudian expanded_data dikembalikan sebagai hasil dari fungsi
    
    # Function to denormalize predicted data
    def denormalize_data(data): #mengalikan data dengan y_train_std/standar deviasi dari data latih yang digunakan dalam normalisasi
        denormalized_data = (data * y_train_std) + y_train_mean # hasil perkalian tersebut ditambahkan dengan y_train_mean/nilai rata-rata dari data latih yang digunakan dalam normalisasi.
        return denormalized_data #mengembalikan data yang telah dinormalisasi ke bentuk semula sebelum normalisasi dilakukan

    def linear_regression_polynomial_formula(coefficients):
        n = len(coefficients)
        polynomial = ""
    
        for i in range(n):
            power = n - i - 1
            coefficient = coefficients[i]
    
            if power > 1:
                term = f"{coefficient} * X^{power}"
            elif power == 1:
                term = f"{coefficient} * X"
            else:
                term = f"{coefficient}"
    
            if coefficient >= 0 and i > 0:
                polynomial += " + " + term
            else:
                polynomial += term

        return polynomial

    def main():
        st.title("Rumus Linear Regression dengan Polynomial")
        st.write("y = w0 + w1X + w2X^2 + ... + wn*X^n")
        st.write(coef)
        st.write(intercept)
    
        # Definisikan koefisien-koefisien polynomial dari model Linear Regression
        coefficients = [2, -1, 3]  # Ganti dengan koefisien-koefisien yang diinginkan
    
        result = linear_regression_polynomial_formula(coefficients)
    
        # Menampilkan rumus Linear Regression dengan Polynomial
        st.write(f"y = {result}")
    
        # Tambahkan fitur untuk menampilkan rumus/model Linear Regression dengan Polynomial di sini
    
    if __name__ == "__main__":
        main()

    # Streamlit app code
    def main():
        st.title('Prediksi Harga Rumah')
    
        # Input form #digunakan untuk membuat field input teks di mana pengguna dapat memasukkan nilai
        input_data_1 = st.text_input('Luas Tanah', '100')
        input_data_2 = st.text_input('Luas Bangunan', '200')
    
        # Check if input values are numeric
        if not input_data_1.isnumeric() or not input_data_2.isnumeric():
            st.error('Please enter numeric values for the input features.')
            return
        
        # Convert input values to float
        input_feature_1 = float(input_data_1)
        input_feature_2 = float(input_data_2)
    
        # Normalize and expand input features
        input_features = np.array([[input_feature_1, input_feature_2]])
        expanded_input = expand_input_features(input_features)
    
        # Perform prediction
        normalized_prediction = model.predict(expanded_input)
        prediction = denormalize_data(normalized_prediction)
    
        # Display the prediction
        st.subheader('Hasil Prediksi')
        st.write(prediction[0])
    
elif choose == 'Help':
    st.markdown('<h1 style="text-align: center;"> Panduan : </h1><ol type="1" style="text-align: justify; background-color: #00FFFF; padding: 30px; border-radius: 20px;"><li><i><b>Cara View Dataset</b></i> <ol type="a"><li>Masuk ke sistem</li><li>Pilih menu dataset</li></ol></li><li><i><b>Cara Prediksi Harga</b></i> <ol type="a"><li>Pilih menu predict</li><li>Pilih LT dan LB</li><li>Klik tombol prediksi</li></ol></li></ol>', unsafe_allow_html=True)

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
