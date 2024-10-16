import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import time
import math
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import itertools
from sklearn.feature_selection import mutual_info_classif




st.set_page_config(
    page_title="Klasifikasi Stroke dengan SVM",
    page_icon='https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/stroke.jpg',
    
    initial_sidebar_state="expanded",
    layout="centered"
   
)

st.write("""<h1 style="text-align: center;">KLASIFIKASI PENYAKIT STROKE DENGAN SELEKSI FITUR INFORMATION GAIN MENGGUNAKAN METODE SUPPORT VECTOR MACHINE </h1><br>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h3 style="text-align: center;"> SAN SAYIDUS SOLATAL A`LA <p>200411100032</p></h3>""", unsafe_allow_html=True),
            ["Home", "Description", "Dataset", "Prepocessing" ,"Modeling", "Implementation"],
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#6495ED"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#00008B"}
            }
        )
        # st.write("""
        # <div style = "position: fixed; left:40px; bottom: 10px;">
        #     <center><a href=""><span><img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" width="40px" height="40px"></span></a><a style = "margin-left: 20px;" href="http://hanifsantoso05.github.io/datamining/intro.html"><span><img src="https://friconix.com/png/fi-stluxx-jupyter-notebook.png" width="40px" height="40px"></span></a> <a style = "margin-left: 20px;" href="mailto: mellnur2901@gmail.com"><span><img src="https://cdn-icons-png.flaticon.com/512/60/60543.png" width="40px" height="40px"></span></a></center>
        # </div> 
        # """, unsafe_allow_html=True)

    if selected == "Home":
        st.write("""<div style="text-align: justify;">
        Klasifikasi merupakan salah satu teknik data mining yang dapat digunakan untuk prediksi kelas dari suatu kelompok data. Klasifikasi penyakit stroke dengan menggunakan fitur Information Gain. Support Vector Machine (SVM) merupakan metode supervised learning yang sering digunakan untuk klasifikasi dan regresi. 
        SVM juga dapat mengatasi masalah klasifikasi dan regresi untuk data linier maupun non-linier Penelitian ini bertujuan untuk meningkatkan akurasi klasifikasi metode SVM pada data penyakit stroke dengan menggunakan seleksi fitur information gain. Penggunaan metode SVM dengan waktu pengujiannya yang singkat 
        diperlukan juga untuk memperkecil beban komputasinya. Salah satu cara untuk memperkecil beban komputasi data sebelum dilakukan uji dengan menggunakan SVM adalah dengan melakukan seleksi fitur Information Gain. Oleh karena itu, peneliti tertarik untuk meneliti terkait klasifikasi penyakit stroke dengan menggunakan 
        metode SVM menggunakan fitur Information Gain. Penelitian ini merujuk pada penelitian sebelumnya yang telah membandingkan metode SVM dengan metode lainnya dapat menghasilkan akurasi yang kurang tinggi sehingga diperlukan ekstraksi fitur agar dapat meningkatkan nilai akurasi akurasi terbaik daripada peneliti yang menggunakan 
        metode lainnya dengan metode Information Gain dapat mengurangi dimensi fitur pada klasifikasi dengan menerapkan teknik scoring dalam melakukan pembobotan menggunakan maksimal entropy.<br>
      </div>
      """, unsafe_allow_html=True)
        
    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""<div style="text-align: justify;">
        Dataset yang digunakan pada penelitian ini merupakan data Prediksi Stroke. Dataset ini berjumlah 5110 data dengan 11 fitur dan 1 label berisi 2 kelas yakni 1 (stroke) dan 0 (tidak stroke). Dataset ini berisi jumlah diagnosis stroke 249 orang dan tidak stroke sebanyak 4861 orang. Dan di dalam dataset terdapat nilai kosong (missing value) sebanyak 201 data.
        Dataset stroke ini berasal dari penelitian thesis Oluwafemi Emmanuel Zachariah tentang “Prediksi Penyakit Stroke dengan Data Demografis Dan Perilaku Menggunakan Algoritma SVM” dari Sheffield Hallam University yang diambil dari catatan kesehatan dari berbagai rumah sakit di Bangladesh yang dapat diakses melalui link berikut  ini https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/.<br>
        </div>
        """, unsafe_allow_html=True)
        

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Data ini akan digunakan untuk melakukan prediksi atau klasifikasi penderita Stroke.
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            -   ID	Merupakan id unik tiap pasien bertipe Numerik
            -   Gender	Merupakan fitur jenis kelamin pasien. berisi dari kategori yaitu female (perempuan), male (laki-laki)  dan other (lainnya) bertipe	Kategorik
            -   Age	Merupakan fitur umur pasien	Numerik
            -   Hypertension,	Merupakan fitur yang berisi apakah pasien tersebut memiliki penyakit jantung atau tidak. Berisi kategori 0 dan kategori 1 bertipe Kategorik
            -   heart_disease	Fitur  apakah pasien tersebut memiliki penyakit hipertensi (1) atau tidak (0) bertipe Kategorik
            -   Ever_married	Fitur berisi apakah pasien sudah pernah menikah atau tidak, terdiri dari 2 kategori yaitu kategori Yes dan No bertipe Kategorik
            -   Work_type	Fitur tipe pekerjaan berisi 5 fitur yakni Private (pribadi) , self- employed (bekerja sendiri), children (anak-anak) , Govt_job (pekerjaan pemerintahan) , Never_worked (tidak pernah bekerja) bertipeKategorik
            -   Residence_type	Fitur jenis tempat tinggal berisi berisi 2 kategori Urban (Perkotaan) dan Rural (Pedesaan) bertipe	Kategorik
            -   avg_glucose_level	Fitur berisi rata rata tingkat kadar glukosa dalam darah pada pasien bertipe Numerik 
            -   bmi	Fitur bmi (body mass index) pasien bertipe Numerik
            -   smoking_status	fitur 	berisi status merokok pasien, Never smoke (tidak pernah merokok), Unknown (tidak diketahui), formery smoked (sebelumnya merokok), smoked (merokok) bertipe Kategorik
            -   stroke	Fitur berisi label diagnosis stroke berisi 2 kategori 1 (stroke) orang dan 0 (tidak) bertipe Kategorik

            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <br>
        <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/"> Kunjungi Sumber Data di Kaggle</a>
        <br>
        <a href="https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv"> Lihat Dataset Stroke</a>""", unsafe_allow_html=True)

        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset ini adalah numerik dan kategorikal.
        """)

    elif selected == "Dataset":
        # Memuat data
        df = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv')

        # Menampilkan header dan dataset
        st.header("Analisa Data Stroke")
        st.subheader("Dataset Stroke")
        st.dataframe(df, width=1000)

        # Membuat kategori
        df['age_group'] = pd.cut(df['age'], bins=[0, 17, 65, 79, float('inf')], labels=['0-17', '18-65', '66-79', '>79'])
        df['avg_group'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 200, float('inf')], labels=['0-100', '101-200', '>200'])
        df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 24.9, 29.9, float('inf')], labels=['0-24.9', '25-29.9', '>29.9'])

        def plot_stroke_cases(column_name, plot_title, colors=['skyblue', 'red']):
            # Mengelompokkan data berdasarkan kolom tertentu dan stroke
            stroke_counts = df.groupby([column_name, 'stroke']).size().unstack()

            # Mengatur plot
            fig, ax = plt.subplots(figsize=(10, 6))
            stroke_counts.plot(kind='bar', stacked=False, color=colors, ax=ax)

            # Menambahkan judul dan label
            ax.set_title(plot_title)
            ax.set_xlabel(column_name.capitalize())
            ax.set_ylabel('Jumlah Kasus')
            ax.legend(['Tidak Stroke','Stroke'], title='Status Stroke')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Menambahkan nilai pada batang
            for p in ax.patches:
                height = p.get_height()
                if height > 0:  # Hanya menambahkan anotasi untuk batang dengan tinggi lebih besar dari 0
                    ax.annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center',
                                va='center',
                                xytext=(0, 10),
                                textcoords='offset points')

            # Menampilkan plot
            st.pyplot(fig)

        # List kolom yang akan digunakan untuk membuat plot
        columns = ['gender', 'age_group', 'avg_group', 'bmi_group', 'hypertension', 
                'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

        # Menampilkan dropdown untuk memilih kolom
        selected_column = st.selectbox('Pilih kolom untuk melihat jumlah kasus stroke', columns)

        # Memanggil fungsi plot_stroke_cases untuk membuat plot
        plot_stroke_cases(selected_column, f'Jumlah Kasus Stroke Berdasarkan {selected_column.capitalize()}')
       
    elif selected == "Prepocessing":
        df = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/healthcare-dataset-stroke-data.csv')
        st.subheader('Hasil Penghapusan Kolom ID ')
        df = df.drop('id', axis=1)
        st.dataframe(df, width=800)

        st.subheader("""Cek missing value:""")
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)

        # Imputasi Missing Values dengan mean
        st.subheader("""Penanganan data kosong:""")
        st.write("""Imputasi dengan metode Mean mengisi missing data dalam suatu variable dengan rata-rata dari semua nilai yang diketahui pada suatu variabel.""")
        df['bmi'].fillna(math.floor(df['bmi'].mean()),inplace=True)
        mis = df.isnull().sum().reset_index()
        mis.columns = ['Fitur', 'Jumlah Missing Values']
        st.dataframe(mis, width=400)

        st.subheader("""Label Encoding""")
        st.write("""Dilakukan perubahan data dari kategorikal menjadi numerik berdasarkan penomoran kategori secara berurutan.""")
        encode = LabelEncoder()
        # transformasi data
        df['gender'] = encode.fit_transform(df['gender'].values)
        df['ever_married'] = encode.fit_transform(df['ever_married'].values)
        df['work_type'] = encode.fit_transform(df['work_type'].values)
        df['Residence_type'] = encode.fit_transform(df['Residence_type'].values)
        df['smoking_status'] = encode.fit_transform(df['smoking_status'].values)
        st.dataframe(df)

        # Normalisasi
        st.subheader('Normalisasi data')
        st.write("""Min-Max Scaler mengubah nilai data menjadi rentang nilai 0 hingga 1. Dengan Min-Max Scaler, semua nilai dalam atribut akan berada dalam rentang 0 hingga 1.""")
        columns_to_normalize = ['age', 'bmi', 'avg_glucose_level']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[columns_to_normalize])
        df[columns_to_normalize] = scaled
        st.dataframe(df)

        # Simpan DataFrame setelah semua operasi pemrosesan
        df.to_csv('data_preprocess.csv', index=False)
        # # Menampilkan hasil imputasi
        # st.dataframe(df_imputed, width=600)

        #SMOTE    
        st.subheader('Synthetic Minority Over-sampling Technique (SMOTE)')
        st.write("""Teknik SMOTE bekerja dengan menambahkan data pada kelas minor untuk menyeimbangkan jumlah data yang sama dengan kelas mayor dengan cara membangkitkan data sintesis.""")
        # x adalah atribut yang mempengaruhi
        # y adalah label itu sendiri
        X = df.drop(['stroke'],axis=1)
        y = df['stroke']

        # Visualisasi data sebelum SMOTE
        st.subheader('Distribusi Data Sebelum SMOTE')
        fig_before, ax_before = plt.subplots(figsize=(8, 6))  # Perkecil ukuran diagram
        sns.countplot(x=y, ax=ax_before)
        ax_before.set_title('Distribusi Label Sebelum SMOTE')
        for p in ax_before.patches:
            ax_before.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        st.pyplot(fig_before)

#SMOTE
        smote = SMOTE(k_neighbors=3,random_state=42)
        # sampling smote
        X_resampled, y_resampled = smote.fit_resample(X,y)
        
       # Visualisasi data setelah SMOTE
        st.subheader('Distribusi Data Setelah SMOTE')
        fig_after, ax_after = plt.subplots(figsize=(8, 6))  # Perkecil ukuran diagram
        sns.countplot(x=y_resampled, ax=ax_after)
        ax_after.set_title('Distribusi Label Setelah SMOTE')
        for p in ax_after.patches:
            ax_after.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
        st.pyplot(fig_after)

        # Simpan DataFrame setelah SMOTE
        df.to_csv('SMOTE.csv', index=False)

    elif selected == "Modeling":
        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            SVM = st.checkbox('SVM')
            SVM_IG = st.checkbox('SVM IG')
            SVM_SMOTE = st.checkbox('SVM SMOTE')
            SVM_SMOTE_IG = st.checkbox('SVM SMOTE IG')
            submitted = st.form_submit_button("Submit")
            if submitted:
                if SVM:
                    st.subheader('SVM kernel RBF')
                    @st.cache_data
                    def load_data():
                        data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/nonSMOTE.csv')
                        data = data.drop(['Unnamed: 0'], axis=1)
                        return data

                    @st.cache_data
                    def perform_cross_validation(x_train, y_train, params, kernel_type):
                        k_fold = KFold(n_splits=5)
                        param_combinations = [(C, param) for C in params['C'] for param in params[kernel_type]]

                        results = []
                        for i, (C, param) in enumerate(param_combinations):
                            if kernel_type == 'gamma':
                                model = SVC(kernel='rbf', C=C, gamma=param)
                            elif kernel_type == 'degree':
                                model = SVC(kernel='poly', C=C, degree=param)
                            
                            scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
                            results.append({
                                'params': {'C': C, kernel_type: param},
                                'scores': scores,
                                'iteration': i + 1
                            })
                        return results

                    data = load_data()
                    jumlah_total_kelas = data['stroke'].value_counts().to_dict()
                    st.write(jumlah_total_kelas)
                    X1 = data.drop('stroke', axis=1)
                    y1 = data['stroke']

                    x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

                    # SVM-SMOTE kernel RBF
                    st.subheader('SVM kernel RBF')
                    params_rbf = {
                        'C': [0.1, 1, 10, 100],
                        'gamma': [0.01, 0.1, 1, 10, 100]
                    }
                    results_rbf = perform_cross_validation(x_train, y_train, params_rbf, 'gamma')

                    for result in results_rbf:
                        st.write(f"Iterasi ke-{result['iteration']}:")
                        st.write(f"Hyperparameter: C={result['params']['C']}, gamma={result['params']['gamma']}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    best_result_rbf = max(results_rbf, key=lambda x: max(x['scores']))
                    best_params_rbf = best_result_rbf['params']
                    best_iteration_rbf = best_result_rbf['iteration']
                    best_fold_index_rbf = best_result_rbf['scores'].tolist().index(max(best_result_rbf['scores'])) + 1

                    st.subheader(f"Model terbaik pada iterasi ke-{best_iteration_rbf}: C={best_params_rbf['C']}, gamma={best_params_rbf['gamma']}")
                    st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index_rbf}: {max(best_result_rbf['scores'])}")

                    # SVM-SMOTE kernel Polynomial
                    st.subheader('SVM-SMOTE kernel Polynomial')
                    params_poly = {
                        'C': [0.1, 1, 10, 100],
                        'degree': [1, 2, 3, 4, 5]
                    }
                    results_poly = perform_cross_validation(x_train, y_train, params_poly, 'degree')

                    for result in results_poly:
                        st.write(f"Iterasi ke-{result['iteration']}:")
                        st.write(f"Hyperparameter: C={result['params']['C']}, degree={result['params']['degree']}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    best_result_poly = max(results_poly, key=lambda x: max(x['scores']))
                    best_params_poly = best_result_poly['params']
                    best_iteration_poly = best_result_poly['iteration']
                    best_fold_index_poly = best_result_poly['scores'].tolist().index(max(best_result_poly['scores'])) + 1

                    st.subheader(f"Model terbaik pada iterasi ke-{best_iteration_poly}: C={best_params_poly['C']}, degree={best_params_poly['degree']}")
                    st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index_poly}: {max(best_result_poly['scores'])}")

                    # # Fungsi untuk memuat dan memproses data
                    # data= pd.read_csv('data_preprocess.csv')
                    #  # Menghitung jumlah total tiap kelas
                    # jumlah_total_kelas = data['stroke'].value_counts().to_dict()
                    # st.write(jumlah_total_kelas)
                    # X = data.drop('stroke',axis=1)
                    # y = data['stroke']

                    # # Memisahkan dataset menjadi data pelatihan dan pengujian
                    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    # # Membuat objek k_fold
                    # k_fold = KFold(n_splits=5)

                    # # Menentukan parameter yang akan diuji
                    # params = {
                    #     'C': [0.1, 1, 10, 100],
                    #     'gamma': [0.01, 0.1, 1, 10, 100]
                    # }

                    # # Variasi parameter
                    # param_combinations = [(C, gamma) for C in params['C'] for gamma in params['gamma']]

                    # # Untuk menyimpan hasil cross-validation
                    # results = []

                    # # Melakukan cross-validation untuk setiap kombinasi parameter
                    # for i, (C, gamma) in enumerate(param_combinations):
                    #     svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
                    #     scores = cross_val_score(svm_rbf, x_train, y_train, cv=k_fold, scoring='accuracy')
                    #     results.append({
                    #         'params': {'C': C, 'gamma': gamma},
                    #         'scores': scores,
                    #         'iteration': i+1  # Menyimpan nomor iterasi
                    #     })

                    #     # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    #     st.write(f"Iterasi ke-{i+1}:")
                    #     st.write(f"Hyperparameter: C={C}, gamma={gamma}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                    # best_result = max(results, key=lambda x: max(x['scores']))
                    # best_params = best_result['params']
                    # best_iteration = best_result['iteration']
                    # best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1

                    # st.subheader(f"Model terbaik pada iterasi ke-{best_iteration}: C={best_params['C']}, gamma={best_params['gamma']}")
                    # st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index}: {max(best_result['scores'])}")
                    

                    # st.subheader('SVM kernel Polynomial')
                    # # Menentukan parameter yang akan diuji
                    # params = {
                    #     'C': [0.1, 1, 10, 100],
                    #     'degree': [1, 2, 3, 4, 5]
                    # }

                    # # Variasi parameter
                    # param_combinations = [(C, degree) for C in params['C'] for degree in params['degree']]

                    # # Untuk menyimpan hasil cross-validation
                    # results = []

                    # # Melakukan cross-validation untuk setiap kombinasi parameter
                    # for i, (C, degree) in enumerate(param_combinations):
                    #     svm_poly = SVC(kernel='poly', C=C, degree=degree)
                    #     scores = cross_val_score(svm_poly, x_train, y_train, cv=k_fold, scoring='accuracy')
                    #     results.append({
                    #         'params': {'C': C, 'degree': degree},
                    #         'scores': scores,
                    #         'iteration': i+1  # Menyimpan nomor iterasi
                    #     })

                    #     # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    #     st.write(f"Iterasi ke-{i+1}:")
                    #     st.write(f"Hyperparameter: C={C}, degree={degree}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                    # best_result2 = max(results, key=lambda x: max(x['scores']))
                    # best_params2 = best_result2['params']
                    # best_iteration2 = best_result2['iteration']
                    # best_fold_index2 = best_result2['scores'].tolist().index(max(best_result2['scores'])) + 1

                    # st.subheader(f"Model terbaik pada iterasi ke-{best_iteration2}: C={best_params2['C']}, degree={best_params2['degree']}")
                    # st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index}: {max(best_result2['scores'])}")
                    
                    # # Membuat model dengan parameter terbaik
                    # best_model = SVC(kernel='poly', C=best_params2['C'], degree=best_params2['degree'])

                    # # Looping melalui setiap fold untuk mendapatkan data fold terbaik
                    # for fold_index, (train_index, test_index) in enumerate(k_fold.split(x_train)):
                    #     if fold_index == best_fold_index:
                    #         x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
                    #         y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
                    #         break

                    # # Melatih model dengan data latih pada fold terbaik
                    # best_model.fit(x_train_fold, y_train_fold)
                    # # Melakukan prediksi pada data uji
                    # y_pred = best_model.predict(x_test)
                    # # Membuat confusion matrix
                    # conf_matrix = confusion_matrix(y_test, y_pred)

                    # # Plot confusion matrix
                    # plt.figure(figsize=(4, 3))
                    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                    # plt.xlabel('Predicted Labels')
                    # plt.ylabel('True Labels')
                    # plt.title('Confusion Matrix')
                    # st.pyplot(plt)
                    # # Ekstraksi nilai True Positives (TP), False Negatives (FN), True Negatives (TN), dan False Positives (FP)
                    # TP = conf_matrix[1, 1]
                    # FN = conf_matrix[1, 0]
                    # TN = conf_matrix[0, 0]
                    # FP = conf_matrix[0, 1]
                    # # Menampilkan nilai True Positives, False Negatives, True Negatives, dan False Positives
                    # st.write(f"True Positives: {TP}")
                    # st.write(f"False Negatives: {FN}")
                    # st.write(f"True Negatives: {TN}")
                    # st.write(f"False Positives: {FP}")
                    # # Menghitung akurasi uji
                    # test_accuracy = accuracy_score(y_test, y_pred)
                    # # Menampilkan akurasi uji
                    # st.write(f"Akurasi pada data uji: {test_accuracy}")

                    
                elif SVM_IG:
                    st.subheader('SVM IG kernel Polynomial')
                    @st.cache_data
                    def load_data():
                        data = pd.read_csv('data_preprocess.csv')
                        return data

                    @st.cache_data
                    def calculate_info_gain(X, y):
                        info_gain = mutual_info_classif(X, y, random_state=13)
                        info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                        info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                        return info_gain_df

                    @st.cache_data
                    def perform_cross_validation(x_train, y_train, info_gain_df, C, degree):
                        k_fold = KFold(n_splits=5)
                        results = []
                        
                        for K in range(5, len(info_gain_df) + 1):
                            selected_features = info_gain_df['Fitur'][:K]
                            svm_poly = SVC(kernel='poly', C=C, degree=degree)
                            scores = cross_val_score(svm_poly, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')
                            best_score = max(scores)
                            best_fold = scores.tolist().index(best_score) + 1
                            results.append({
                                'num_features': K,
                                'features': selected_features,
                                'params': {'C': C, 'degree': degree},
                                'scores': scores,
                                'best_score': best_score,
                                'best_fold': best_fold,
                            })
                        return results

                    data = load_data()
                    X = data.drop('stroke', axis=1)
                    y = data['stroke']

                    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    info_gain_df = calculate_info_gain(X, y)
                    st.dataframe(info_gain_df)

                    C = 100
                    degree = 4

                    results = perform_cross_validation(x_train, y_train, info_gain_df, C, degree)

                    for result in results:
                        K = result['num_features']
                        st.write(f"Jumlah fitur: {K}")
                        st.write(f"Hyperparameter: C={C}, degree={degree}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    st.subheader("Ringkasan Best Fold untuk Tiap Jumlah Fitur:")
                    for result in results:
                        st.write(f"Jumlah fitur: {result['num_features']}, Best Fold: {result['best_score']} (Fold ke-{result['best_fold']})")
                    # data = pd.read_csv('data_preprocess.csv')
                    # X = data.drop('stroke', axis=1)
                    # y = data['stroke']
                    # # Memisahkan dataset menjadi data pelatihan dan pengujian
                    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    # # Menghitung Information Gain untuk setiap fitur
                    # info_gain = mutual_info_classif(X, y,random_state=13)
                    # # Membuat DataFrame untuk menampilkan hasil
                    # info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                    # info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                    # st.dataframe(info_gain_df)
                    
                    # # Membuat objek k_fold
                    # k_fold = KFold(n_splits=5)
                    # # Mendefinisikan parameter yang akan diuji
                    # C = 100
                    # degree = 4

                    # # Untuk menyimpan hasil cross-validation
                    # results = []
                    # # Loop untuk mencoba berbagai kombinasi fitur mulai dari 5 hingga semua fitur
                    # for K in range(5, len(info_gain_df) + 1):
                    #     selected_features = info_gain_df['Fitur'][:K]
                    #     # Filter matriks fitur berdasarkan fitur yang dipilih
                    #     X_selected = X[selected_features]
                    #     # Melakukan cross-validation dengan parameter yang ditentukan
                    #     svm_poly = SVC(kernel='poly', C=C, degree=degree)
                    #     scores = cross_val_score(svm_poly, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')
                    #     best_score = max(scores)
                    #     best_fold = scores.tolist().index(best_score) + 1  # +1 karena indeks dimulai dari 0
                    #     results.append({
                    #         'num_features': K,
                    #         'features': selected_features,
                    #         'params': {'C': C, 'degree': degree},
                    #         'scores': scores,
                    #         'best_score': best_score,
                    #         'best_fold': best_fold,
                    #     })
                    #     # Menampilkan hasil akurasi dengan 5-fold untuk jumlah fitur saat ini
                    #     st.write(f"Jumlah fitur: {K}")
                    #     st.write(f"Hyperparameter: C={C}, degree={degree}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menampilkan ringkasan best fold untuk tiap iterasi
                    # st.subheader("Ringkasan Best Fold untuk Tiap Jumlah Fitur:")
                    # for result in results:
                    #     st.write(f"Jumlah fitur: {result['num_features']}, Best Fold: {result['best_score']} (Fold ke-{result['best_fold']})")
     
                elif SVM_SMOTE:
                    st.subheader('SVM-SMOTE kernel RBF')

                    @st.cache_data
                    def load_data():
                        data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                        data = data.drop(['Unnamed: 0'], axis=1)
                        return data

                    @st.cache_data
                    def perform_cross_validation(x_train, y_train, params, kernel_type):
                        k_fold = KFold(n_splits=5)
                        param_combinations = [(C, param) for C in params['C'] for param in params[kernel_type]]

                        results = []
                        for i, (C, param) in enumerate(param_combinations):
                            if kernel_type == 'gamma':
                                model = SVC(kernel='rbf', C=C, gamma=param)
                            elif kernel_type == 'degree':
                                model = SVC(kernel='poly', C=C, degree=param)
                            
                            scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
                            results.append({
                                'params': {'C': C, kernel_type: param},
                                'scores': scores,
                                'iteration': i + 1
                            })
                        return results

                    data = load_data()
                    jumlah_total_kelas = data['stroke'].value_counts().to_dict()
                    st.write(jumlah_total_kelas)
                    X1 = data.drop('stroke', axis=1)
                    y1 = data['stroke']

                    x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

                    # SVM-SMOTE kernel RBF
                    st.subheader('SVM-SMOTE kernel RBF')
                    params_rbf = {
                        'C': [0.1, 1, 10, 100],
                        'gamma': [0.01, 0.1, 1, 10, 100]
                    }
                    results_rbf = perform_cross_validation(x_train, y_train, params_rbf, 'gamma')

                    for result in results_rbf:
                        st.write(f"Iterasi ke-{result['iteration']}:")
                        st.write(f"Hyperparameter: C={result['params']['C']}, gamma={result['params']['gamma']}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    best_result_rbf = max(results_rbf, key=lambda x: max(x['scores']))
                    best_params_rbf = best_result_rbf['params']
                    best_iteration_rbf = best_result_rbf['iteration']
                    best_fold_index_rbf = best_result_rbf['scores'].tolist().index(max(best_result_rbf['scores'])) + 1

                    st.subheader(f"Model terbaik pada iterasi ke-{best_iteration_rbf}: C={best_params_rbf['C']}, gamma={best_params_rbf['gamma']}")
                    st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index_rbf}: {max(best_result_rbf['scores'])}")

                    # SVM-SMOTE kernel Polynomial
                    st.subheader('SVM-SMOTE kernel Polynomial')
                    params_poly = {
                        'C': [0.1, 1, 10, 100],
                        'degree': [1, 2, 3, 4, 5]
                    }
                    results_poly = perform_cross_validation(x_train, y_train, params_poly, 'degree')

                    for result in results_poly:
                        st.write(f"Iterasi ke-{result['iteration']}:")
                        st.write(f"Hyperparameter: C={result['params']['C']}, degree={result['params']['degree']}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    best_result_poly = max(results_poly, key=lambda x: max(x['scores']))
                    best_params_poly = best_result_poly['params']
                    best_iteration_poly = best_result_poly['iteration']
                    best_fold_index_poly = best_result_poly['scores'].tolist().index(max(best_result_poly['scores'])) + 1

                    st.subheader(f"Model terbaik pada iterasi ke-{best_iteration_poly}: C={best_params_poly['C']}, degree={best_params_poly['degree']}")
                    st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index_poly}: {max(best_result_poly['scores'])}")

                    # data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                    # data = data.drop(['Unnamed: 0'],axis=1)
                    # # Menghitung jumlah total tiap kelas
                    # jumlah_total_kelas = data['stroke'].value_counts().to_dict()
                    # st.write(jumlah_total_kelas)
                    # X1 = data.drop('stroke',axis=1)
                    # y1 = data['stroke']

                    # # Memisahkan dataset menjadi data pelatihan dan pengujian
                    # x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
                    # # Membuat objek k_fold
                    # k_fold = KFold(n_splits=5)

                    # # Menentukan parameter yang akan diuji
                    # params = {
                    #     'C': [0.1, 1, 10, 100],
                    #     'gamma': [0.01, 0.1, 1, 10, 100]
                    # }

                    # # Variasi parameter
                    # param_combinations = [(C, gamma) for C in params['C'] for gamma in params['gamma']]

                    # # Untuk menyimpan hasil cross-validation
                    # results = []

                    # # Melakukan cross-validation untuk setiap kombinasi parameter
                    # for i, (C, gamma) in enumerate(param_combinations):
                    #     svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
                    #     scores = cross_val_score(svm_rbf, x_train, y_train, cv=k_fold, scoring='accuracy')
                    #     results.append({
                    #         'params': {'C': C, 'gamma': gamma},
                    #         'scores': scores,
                    #         'iteration': i+1  # Menyimpan nomor iterasi
                    #     })

                    #     # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    #     st.write(f"Iterasi ke-{i+1}:")
                    #     st.write(f"Hyperparameter: C={C}, gamma={gamma}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                    # best_result = max(results, key=lambda x: max(x['scores']))
                    # best_params = best_result['params']
                    # best_iteration = best_result['iteration']
                    # best_fold_index = best_result['scores'].tolist().index(max(best_result['scores'])) + 1

                    # st.subheader(f"Model terbaik pada iterasi ke-{best_iteration}: C={best_params['C']}, gamma={best_params['gamma']}")
                    # st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index}: {max(best_result['scores'])}")
                    

                    # st.subheader('SVM-SMOTE kernel Polynomial')
                    # # Menentukan parameter yang akan diuji
                    # params = {
                    #     'C': [0.1, 1, 10, 100],
                    #     'degree': [1, 2, 3, 4, 5]
                    # }

                    # # Variasi parameter
                    # param_combinations = [(C, degree) for C in params['C'] for degree in params['degree']]

                    # # Untuk menyimpan hasil cross-validation
                    # results = []

                    # # Melakukan cross-validation untuk setiap kombinasi parameter
                    # for i, (C, degree) in enumerate(param_combinations):
                    #     svm_poly = SVC(kernel='poly', C=C, degree=degree)
                    #     scores = cross_val_score(svm_poly, x_train, y_train, cv=k_fold, scoring='accuracy')
                    #     results.append({
                    #         'params': {'C': C, 'degree': degree},
                    #         'scores': scores,
                    #         'iteration': i+1  # Menyimpan nomor iterasi
                    #     })

                    #     # Menampilkan hasil akurasi dengan 5-fold pada tiap iterasi
                    #     st.write(f"Iterasi ke-{i+1}:")
                    #     st.write(f"Hyperparameter: C={C}, degree={degree}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menentukan model terbaik berdasarkan akurasi fold tertinggi
                    # best_result2 = max(results, key=lambda x: max(x['scores']))
                    # best_params2 = best_result2['params']
                    # best_iteration2 = best_result2['iteration']
                    # best_fold_index2 = best_result2['scores'].tolist().index(max(best_result2['scores'])) + 1

                    # st.subheader(f"Model terbaik pada iterasi ke-{best_iteration2}: C={best_params2['C']}, degree={best_params2['degree']}")
                    # st.subheader(f"Akurasi fold tertinggi pada fold ke-{best_fold_index}: {max(best_result2['scores'])}")
                    
                    # # Membuat model dengan parameter terbaik
                    # best_model = SVC(kernel='poly', C=best_params2['C'], degree=best_params2['degree'])

                    # # Looping melalui setiap fold untuk mendapatkan data fold terbaik
                    # for fold_index, (train_index, test_index) in enumerate(k_fold.split(x_train)):
                    #     if fold_index == best_fold_index:
                    #         x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[test_index]
                    #         y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
                    #         break

                    # # Melatih model dengan data latih pada fold terbaik
                    # best_model.fit(x_train_fold, y_train_fold)
                    # # Melakukan prediksi pada data uji
                    # y_pred = best_model.predict(x_test)
                    # # Membuat confusion matrix
                    # conf_matrix = confusion_matrix(y_test, y_pred)

                    # # Plot confusion matrix
                    # plt.figure(figsize=(4, 3))
                    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                    # plt.xlabel('Predicted Labels')
                    # plt.ylabel('True Labels')
                    # plt.title('Confusion Matrix')
                    # st.pyplot(plt)
                    # # Ekstraksi nilai True Positives (TP), False Negatives (FN), True Negatives (TN), dan False Positives (FP)
                    # TP = conf_matrix[1, 1]
                    # FN = conf_matrix[1, 0]
                    # TN = conf_matrix[0, 0]
                    # FP = conf_matrix[0, 1]
                    # # Menampilkan nilai True Positives, False Negatives, True Negatives, dan False Positives
                    # st.write(f"True Positives: {TP}")
                    # st.write(f"False Negatives: {FN}")
                    # st.write(f"True Negatives: {TN}")
                    # st.write(f"False Positives: {FP}")
                    # # Menghitung akurasi uji
                    # test_accuracy = accuracy_score(y_test, y_pred)
                    # # Menampilkan akurasi uji
                    # st.write(f"Akurasi pada data uji: {test_accuracy}")

                    
                elif SVM_SMOTE_IG:
                    st.subheader('SVM-SMOTE-IG kernel RBF')
                    @st.cache_data
                    def load_data():
                        data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                        data = data.drop(['Unnamed: 0'], axis=1)
                        return data

                    @st.cache_data
                    def calculate_info_gain(X, y):
                        info_gain = mutual_info_classif(X, y, random_state=13)
                        info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                        info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                        return info_gain_df

                    @st.cache_data
                    def perform_cross_validation(x_train, y_train, info_gain_df, C, degree):
                        k_fold = KFold(n_splits=5)
                        results = []
                        
                        for K in range(5, len(info_gain_df) + 1):
                            selected_features = info_gain_df['Fitur'][:K]
                            svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
                            scores = cross_val_score(svm_rbf, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')
                            best_score = max(scores)
                            best_fold = scores.tolist().index(best_score) + 1
                            results.append({
                                'num_features': K,
                                'features': selected_features,
                                'params': {'C': C, 'gamma': gamma},
                                'scores': scores,
                                'best_score': best_score,
                                'best_fold': best_fold,
                            })
                        return results

                    data = load_data()
                    X = data.drop('stroke', axis=1)
                    y = data['stroke']

                    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    info_gain_df = calculate_info_gain(X, y)
                    st.dataframe(info_gain_df)

                    C = 10
                    gamma = 100

                    results = perform_cross_validation(x_train, y_train, info_gain_df, C, gamma)

                    for result in results:
                        K = result['num_features']
                        st.write(f"Jumlah fitur: {K}")
                        st.write(f"Hyperparameter: C={C}, gamma={gamma}")
                        st.write("Detail Akurasi Tiap Fold:")
                        for j, score in enumerate(result['scores']):
                            st.write(f"Fold ke-{j + 1}: {score}")
                        st.write()

                    st.subheader("Ringkasan Best Fold untuk Tiap Jumlah Fitur:")
                    for result in results:
                        st.write(f"Jumlah fitur: {result['num_features']}, Best Fold: {result['best_score']} (Fold ke-{result['best_fold']})")
                    # data = pd.read_csv('https://raw.githubusercontent.com/SanS2A/dataset/main/SMOTE.csv')
                    # data = data.drop(['Unnamed: 0'],axis=1)

                    # X = data.drop('stroke', axis=1)
                    # y = data['stroke']
                    # # Memisahkan dataset menjadi data pelatihan dan pengujian
                    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    # # Menghitung Information Gain untuk setiap fitur
                    # info_gain = mutual_info_classif(X, y,random_state=13)
                    # # Membuat DataFrame untuk menampilkan hasil
                    # info_gain_df = pd.DataFrame({'Fitur': X.columns, 'Information Gain': info_gain})
                    # info_gain_df = info_gain_df.sort_values(by='Information Gain', ascending=False)
                    # st.dataframe(info_gain_df)
                    
                    # # Membuat objek k_fold
                    # k_fold = KFold(n_splits=5)
                    # # Mendefinisikan parameter yang akan diuji
                    # C = 10
                    # gamma = 100

                    # # Untuk menyimpan hasil cross-validation
                    # results = []
                    # # Loop untuk mencoba berbagai kombinasi fitur mulai dari 5 hingga semua fitur
                    # for K in range(5, len(info_gain_df) + 1):
                    #     selected_features = info_gain_df['Fitur'][:K]
                    #     # Filter matriks fitur berdasarkan fitur yang dipilih
                    #     X_selected = X[selected_features]
                    #     # Melakukan cross-validation dengan parameter yang ditentukan
                    #     svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma)
                    #     scores = cross_val_score(svm_rbf, x_train[selected_features], y_train, cv=k_fold, scoring='accuracy')
                    #     best_score = max(scores)
                    #     best_fold = scores.tolist().index(best_score) + 1  # +1 karena indeks dimulai dari 0
                    #     results.append({
                    #         'num_features': K,
                    #         'features': selected_features,
                    #         'params': {'C': C, 'gamma': gamma},
                    #         'scores': scores,
                    #         'best_score': best_score,
                    #         'best_fold': best_fold,
                    #     })
                    #     # Menampilkan hasil akurasi dengan 5-fold untuk jumlah fitur saat ini
                    #     st.write(f"Jumlah fitur: {K}")
                    #     st.write(f"Hyperparameter: C={C}, gamma={gamma}")
                    #     st.write("Detail Akurasi Tiap Fold:")
                    #     for j, score in enumerate(scores):
                    #         st.write(f"Fold ke-{j+1}: {score}")
                    #     st.write()

                    # # Menampilkan ringkasan best fold untuk tiap iterasi
                    # st.subheader("Ringkasan Best Fold untuk Tiap Jumlah Fitur:")
                    # for result in results:
                    #     st.write(f"Jumlah fitur: {result['num_features']}, Best Fold: {result['best_score']} (Fold ke-{result['best_fold']})")
                    
        
    elif selected == "Implementation":
            st.write("Pilihlah model untuk melakukan klasifikasi:")
            option = st.radio("Pilih Metode:", ['SVM', 'SVM + IG','SVM + SMOTE','SVM + SMOTE + IG'])
            
            if option == 'SVM':
                with st.form(key='random_forest_form'):
                    st.subheader('Masukkan Data Anda')
                    age = st.number_input('Masukkan Umur Pasien')
                    gender = st.radio("Gender",('Male', 'Female'))
                    if gender == "Male":
                        gen_Female = 0
                        gen_Male = 1
                        
                    elif gender == "Female" :
                        gen_Female = 1
                        gen_Male = 0

                    # HYPERTENSION
                    hypertension = st.radio("Hypertency",('No', 'Yes'))
                    if hypertension == "Yes":
                        
                        hypertension_0 = 0
                        hypertension_1 = 1
                    elif hypertension == "No":
                        hypertension_1 = 0
                        hypertension_0 = 1
                    
                    # HEART
                    heart_disease = st.radio("Heart Disease",('No', 'Yes'))
                    if heart_disease == "Yes":
                        heart_disease_1 = 1
                        heart_disease_0 = 0
                    elif heart_disease == "No":
                        heart_disease_1 = 0
                        heart_disease_0 = 1

                    # MARRIED
                    ever_married = st.radio("Ever Married",('No', 'Yes'))
                    if ever_married == "Yes":
                        ever_married_Y = 1
                        ever_married_N = 0
                    elif ever_married == "No":
                        ever_married_Y = 0
                        ever_married_N = 1

                    # WORK
                    work_type = st.selectbox(
                    'Select a Work Type',
                    options=['Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'])
                    if work_type == "Govt_job":
                        work_type_G = 1
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Never_worked":
                        work_type_G = 0
                        work_type_Never = 1
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Private":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 1
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Self_employed":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 1
                        work_type_C = 0
                    elif work_type == "childern":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 1

                    # RESIDENCE
                    residence_type = st.radio("Residence Type",('Rural', 'Urban'))
                    if residence_type == "Rural":
                        residence_type_R = 1
                        residence_type_U = 0
                    elif residence_type == "Urban":
                        residence_type_R = 0
                        residence_type_U = 1

                    # GLUCOSE
                    avg_glucose_level = st.number_input('Average Glucose Level')
                    
                    # SMOKE
                    smoking_status = st.selectbox(
                    'Select a smoking status',
                    options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])

                    if smoking_status == "Unknown":
                        smoking_status_U = 1
                        smoking_status_F = 0
                        smoking_status_N = 0
                        smoking_status_S = 0
                    elif smoking_status == "Formerly smoked":
                        smoking_status_U = 0
                        smoking_status_F = 1
                        smoking_status_N = 0
                        smoking_status_S = 0
                    elif smoking_status == "never smoked":
                        smoking_status_U = 0
                        smoking_status_F = 0
                        smoking_status_N = 1
                        smoking_status_S = 0
                    elif smoking_status == "smokes":
                        smoking_status_U = 0
                        smoking_status_F = 0
                        smoking_status_N = 0
                        smoking_status_S = 1
                        
                    bmi = st.number_input('BMI')
                  
                    inputs = np.array([[
                            age,avg_glucose_level,bmi, gen_Female, gen_Male,
                            hypertension_0,hypertension_1,
                            heart_disease_0,heart_disease_1,
                            ever_married_N, ever_married_Y,
                            work_type_G, work_type_Never, work_type_P, work_type_S, work_type_C,
                            residence_type_R, residence_type_U,
                            smoking_status_U, smoking_status_F, smoking_status_N, smoking_status_S]])
                
                    st.subheader('Hasil Klasifikasi dengan SVM')
                    cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                    if cek_hasil :
                        st.write(inputs)
                        # Memeriksa apakah ada nilai yang hilang dalam DataFrame
                        
                        use_model = joblib.load("random_forest_model_n_estimator=20_max_depth=20_criterion=entropy.joblib")
                        
                        # Now you can pass inputs_reshaped to the predict() method
                        input_pred = use_model.predict(inputs)
                        # input_pred = use_model.predict([inputs])[0]
                        st.subheader('Hasil Prediksi')
                        st.write(input_pred)
                        if input_pred == 1:
                            st.error('Anda  Terkena Stroke')
                        else:
                            st.success('Anda tidak terkena Stroke')
                                        
            else:
                with st.form(key='SVMSE_form'):
                           
                            st.subheader('Masukkan Data Anda')
                            age = st.number_input('Masukkan Umur Pasien')

            # GENDER
                            gender = st.radio("Gender",('Male', 'Female', 'Other'))
                            if gender == "Male":
                                gen_Female = 0
                                gen_Male = 1
                                
                            elif gender == "Female" :
                                gen_Female = 1
                                gen_Male = 0
                                gen_Other = 0
                            

                            # HYPERTENSION
                            hypertension = st.radio("Hypertency",('No', 'Yes'))
                            if hypertension == "Yes":
                                
                                hypertension_0 = 0
                                hypertension_1 = 1
                            elif hypertension == "No":
                                hypertension_1 = 0
                                hypertension_0 = 1
                            
                            # HEART
                            heart_disease = st.radio("Heart Disease",('No', 'Yes'))
                            if heart_disease == "Yes":
                                heart_disease_1 = 1
                                heart_disease_0 = 0
                            elif heart_disease == "No":
                                heart_disease_1 = 0
                                heart_disease_0 = 1

                            # MARRIED
                            ever_married = st.radio("Ever Married",('No', 'Yes'))
                            if ever_married == "Yes":
                                ever_married_Y = 1
                                ever_married_N = 0
                            elif ever_married == "No":
                                ever_married_Y = 0
                                ever_married_N = 1

                            # WORK
                            work_type = st.selectbox(
                            'Select a Work Type',
                            options=['Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'])
                            if work_type == "Govt_job":
                                work_type_G = 1
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Never_worked":
                                work_type_G = 0
                                work_type_Never = 1
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Private":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 1
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Self_employed":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 1
                                work_type_C = 0
                            elif work_type == "childern":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 1

                            # RESIDENCE
                            residence_type = st.radio("Residence Type",('Rural', 'Urban'))
                            if residence_type == "Rural":
                                residence_type_R = 1
                                residence_type_U = 0
                            elif residence_type == "Urban":
                                residence_type_R = 0
                                residence_type_U = 1

                            # GLUCOSE
                            avg_glucose_level = st.number_input('Average Glucose Level')
                            
                            
                             
                            # if avg_glucose_level <=100 :
                            #     avg_glucose_level=0
                            # elif 101 <= avg_glucose_level <= 200:
                            #     avg_glucose_level=1
                            # else:
                            #     avg_glucose_level=2
                            # SMOKE
                            smoking_status = st.selectbox(
                            'Select a smoking status',
                            options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])

                            if smoking_status == "Unknown":
                                smoking_status_U = 1
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "Formerly smoked":
                                smoking_status_U = 0
                                smoking_status_F = 1
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "never smoked":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 1
                                smoking_status_S = 0
                            elif smoking_status == "smokes":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 1
                                
                            bmi = st.number_input('BMI')

                            df = pd.read_csv('data_clean.csv')
                            st.dataframe(df)
                            x = df.drop(columns=['stroke'])
                            x_normalized = x.copy() 
                            
                            #Normalisasi data input
                            df_min_bmi = x_normalized['bmi'].min().reshape(-1, 1)
                            df_max_bmi =  x_normalized['bmi'].max().reshape(-1, 1)
                            
                            df_min_age = x_normalized['age'].min().reshape(-1, 1)
                            df_max_age = x_normalized['age'].max().reshape(-1, 1)
                                    
                            df_min_avg = x_normalized['avg_glucose_level'].min().reshape(-1, 1)
                            df_max_avg = x_normalized['avg_glucose_level'] .max().reshape(-1, 1)
                            
                            # Make a copy of x to keep the original data
                            age_norm = float((age - df_min_age) / (df_max_age - df_min_age))
                            avg_norm = float((avg_glucose_level - df_min_avg) / (df_max_avg - df_min_avg))
                            bmi_norm = float((bmi - df_min_bmi) / (df_max_bmi - df_min_bmi))
                            
                            inputs = np.array([[bmi_norm,age_norm,avg_norm,work_type_G,gen_Male,smoking_status_N, gen_Female,work_type_P,hypertension_0,heart_disease_0,smoking_status_U,residence_type_R,heart_disease_1,ever_married_Y]])

                            st.subheader('Hasil Klasifikasi dengan SVM SMOTE ENN')
                            cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                            if cek_hasil :
                                st.dataframe(inputs)
                                
                                
                                use_model = joblib.load("SVMSE_k_3_n_estimators=40_max_depth=40_criterion=entropy.joblib")
                                # Normalisasi data input
                                # Now you can pass inputs_reshaped to the predict() method
                                input_pred = use_model.predict(inputs)
                                # input_pred = use_model.predict([inputs])[0]
                            
                            
                                st.subheader('Hasil Prediksi')
                                st.dataframe(input_pred)
                                if input_pred == 1:
                                    st.error('Anda  Terkena Stroke')
                                else:
                                    st.success('Anda tidak terkena Stroke')
                            
                    
