
#Library yang diperlukan
import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from scipy.stats import skew, kurtosis, mode
from sklearn.neighbors import KNeighborsClassifier

st.title('DETEKSI EMOSI')
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
    import pickle
    with open('scaler.pkl', 'rb') as standarisasi:
        loadscal= pickle.load(standarisasi)
    datanormal=loadscal.transform(fitur)
    st.write("=========> CIRI YANG SUDAH DI NORMALISASI <=========")
    st.write(datanormal)
    import pickle
    with open('mod.pkl', 'rb') as modelknn:
        loadknn= pickle.load(modelknn)
    prediksi=loadknn.predict(datanormal)
    for pred in prediksi:
        st.write('SUARA DENGAN EMOSI : ', pred)
    