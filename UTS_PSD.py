
#Library yang diperlukan
import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from scipy.stats import skew, kurtosis, mode
from sklearn.neighbors import KNeighborsClassifier
import pickle

selected=option_menu(
    menu_title=None,
    options=['Data', 'Implementasi','Me'],
    default_index=0,
    orientation='horizontal',
    menu_icon=None,
    styles={
    "nav-link":{
        "font-size":"12px",
        "text-align":"center",
        "margin":"5px",
        "--hover-color":"pink",},
    "nav-link-selected":{
        "background-color":"purple"},
    })
if selected=='Data':
    st.title('DESI (Deteksi Emosi)')
    st.write('dataset diambil dari : https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download Dataset audio ini akan di ektraksi menjadi beberapa fitur numeric ZCR dan RMSE,Dataset ini berisi serangkaian 200 kata target yang diucapkan dalam frasa pembawa Katakan kata _ oleh dua aktris (berusia 26 dan 64 tahun), dan rekaman dibuat untuk setiap kata dengan menggambarkan tujuh emosi berbeda (marah, jijik, takut, bahagia, kaget menyenangkan, sedih, dan netral). Totalnya terdapat 2800 titik data (file audio) dalam dataset ini. Penyusunan dataset ini dilakukan dengan cara bahwa setiap dari dua aktris perempuan dan emosi yang mereka tunjukkan terkandung dalam folder tersendiri. Di dalamnya, terdapat file audio untuk semua 200 kata target. Format file audio ini adalah format WAV.Dengan kata lain, dataset ini memberikan akses ke rekaman suara dari dua aktris perempuan yang berbeda usia, masing-masing mengucapkan 200 kata target dalam tujuh berbagai emosi. Setiap kombinasi kata dan emosi memiliki rekaman suara sendiri-sendiri, dan semuanya tersusun dengan baik dalam struktur folder yang jelas. Format file audio yang digunakan adalah WAV.')

if selected=='Implementasi':
    st.title('DESI (Deteksi Emosi)')
    audio=st.file_uploader("Upload AUDIO disini ", type=['mp3','wav'])
    if audio :
        st.audio(audio)
        y, sr = librosa.load(audio)
        # UNTUK MENGHITUNG NILAI ZCR
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
        zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
        zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
        zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
        zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])
        # UNTUK MENGHITUNG NILAI RMSE
        rmse = np.sum(y**2) / len(y)
        rmse_median = np.median(y**2)
        rmse_std_dev = np.std(y**2)
        rmse_kurtosis = kurtosis(y**2)
        rmse_skew = skew(y**2)
        fiturnya={'ZCR Mean':zcr_mean,
            'ZCR Median':zcr_median,
            'ZCR Std Dev':zcr_std_dev,
            'ZCR Kurtosis':zcr_kurtosis,
            'ZCR Skew':zcr_skew,
            'RMSE': rmse,
            'RMSE Median' :rmse_median,
            'RMSE Std Dev':rmse_std_dev,
            'RMSE Kurtosis':rmse_kurtosis,
            'RMSE Skew':rmse_skew}
        fitur=pd.DataFrame(fiturnya,index=[0])
        st.write("=========> EKSTRAKSI AUDIO KEDALAM BEBERAPA CIRI <=========")
        st.write(fitur)
        with open('scaler.pkl', 'rb') as standarisasi:
            loadscal= pickle.load(standarisasi)
        datanormal=loadscal.transform(fitur)
        st.write("=========> CIRI YANG SUDAH DI NORMALISASI <=========")
        st.write(datanormal)
        with open('PCAknn9.pkl', 'rb') as pca:
            loadpca= pickle.load(pca)
        untukpca=loadpca.transform(datanormal)
        st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 9 FITUR <=========")
        st.write(untukpca)
        with open('knnpca9fix.pkl', 'rb') as modelpca:
            knnpca= pickle.load(modelpca)
        predik=knnpca.predict(untukpca)
        for predi in predik:
            st.write('SUARA DENGAN EMOSI : ', predi)

if selected=='Me':
    st.title('ABOUT ME')
    st.write("My Name is LU'LUATUL MAKNUNAH")
    st.write("Just Call Me LUNA")
    st.write("ID Number 210411100048")
