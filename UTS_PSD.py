
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
    options=['KNN', 'PCA','Me'],
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
if selected=='KNN':
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
        with open('scaler.pkl', 'rb') as standarisasi:
            loadscal= pickle.load(standarisasi)
        datanormal=loadscal.transform(fitur)
        st.write("=========> CIRI YANG SUDAH DI NORMALISASI <=========")
        st.write(datanormal)
        with open('mod.pkl', 'rb') as modelknn:
            loadknn= pickle.load(modelknn)
        prediksi=loadknn.predict(datanormal)
        for pred in prediksi:
            st.write('SUARA DENGAN EMOSI : ', pred)

if selected=='PCA':
    st.title('DETEKSI EMOSI')
    audio=st.file_uploader("Upload AUDIO disini ", type=['mp3','wav'])
    if audio :
        st.audio(audio)
        genre = st.radio("PILIH MODEL : ",('PCA9', 'PCA8','PCA6','PCA5','PCA4','PCA3','PCA2','PCA1'))
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
        if genre=='PCA9':
            with open('PCA9.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 9 FITUR <=========")
            st.write(untukpca)
            with open('knnpca.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
        
        if genre=='PCA8':
            with open('PCA8.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 8 FITUR <=========")
            st.write(untukpca)
            with open('knnpca8.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
        
                
        if genre=='PCA6':
            with open('PCA6.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 6 FITUR <=========")
            st.write(untukpca)
            with open('knnpca6.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
                
        if genre=='PCA5':
            with open('PCA5.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 5 FITUR <=========")
            st.write(untukpca)
            with open('knnpca5.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
                
        if genre=='PCA4':
            with open('PCA4.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 4 FITUR <=========")
            st.write(untukpca)
            with open('knnpca4.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
                
        if genre=='PCA3':
            with open('PCA3.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 3 FITUR <=========")
            st.write(untukpca)
            with open('knnpca3.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
                
        if genre=='PCA2':
            with open('PCA2.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 2 FITUR <=========")
            st.write(untukpca)
            with open('knnpca2.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)
        
        if genre=='PCA1':
            with open('PCA1.pkl', 'rb') as pca:
                loadpca= pickle.load(pca)
            untukpca=loadpca.transform(datanormal)
            st.write("=========> CIRI YANG SUDAH DI REDUKSI DIMENSI MENJADI 1 FITUR <=========")
            st.write(untukpca)
            with open('knnpca1.pkl', 'rb') as modelpca:
                knnpca= pickle.load(modelpca)
            predik=knnpca.predict(untukpca)
            for predi in predik:
                st.write('SUARA DENGAN EMOSI : ', predi)

if selected=='Me':
    st.title('ABOUT ME')
    st.write("My Name is LU'LUATUL MAKNUNAH")
    st.write("Just Call Me LUNA")
    st.write("ID Number 210411100048")
