import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from crawl import *
from preprocessing import *
from training import *
from visualisasi import *
from aspek_base import *
from tfidf import *


with st.sidebar:
    selected = option_menu("Menu",['Crawling',"Preprocessing","TF-IDF","Training","Prediction"],
                           icons=['table','stars','bi bi-subtract', 'gear', 'bi bi-search'], menu_icon="cast",
                           default_index=0, styles={
        "container": {"padding": "5!important", "padding-top":"0px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
    })

if selected =="Crawling":
    st.title("Crawling Ulasan Kraton: Pada Google Maps")
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            jumlah_load = st.number_input('Masukan Jumlah Load', min_value=2, step=2)
            nama_file = st.text_input('Masukan Nama File Penyimpanan Hasil Crawling')
            crawling = st.button('Crawling Ulasan')
    if crawling:
        with st.spinner('Sedang Mengambil ulasan...'):
            ulasan  = crawling_ulasan(jumlah_load)
            directory = 'D:/Analisis Sentimen/Data/Crawling'
            file_path = os.path.join(directory, nama_file)
            ulasan.to_csv(file_path)
            st.write('#### **Ulasan**')
            st.dataframe(ulasan)
            st.info(f'Data {nama_file} Berhasil Disimpan')

if selected =='Preprocessing':   
    st.title("Preprocessing")
    folder_path = 'D:/Analisis Sentimen/Data/Crawling'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    selected_file = st.selectbox('Select a CSV file', csv_files)
    file_path = os.path.join(folder_path, selected_file)
    ulasan = pd.read_csv(file_path, index_col=0)
    st.write('#### **Ulasan Kraton:**')
    st.dataframe(ulasan)
    nama_file = st.text_input('Masukan Nama File Penyimpanan Hasil Preprocessing')
    preprocessing = st.button("Preprocessing")
    if preprocessing:
        with st.spinner('Sedang Melakukan Terjemah...'):
            ulasan['review'] = ulasan['review'].apply(translate_text)
            st.write('**Translate Text:**')
            st.write(ulasan[['rating', 'review']])
        ulasan['review'] = ulasan['review'].apply(clean)
        st.write('**Cleaning:**')
        st.write(ulasan[['rating', 'review']])
        ulasan['review'] = ulasan['review'].apply(normalisasi)
        st.write(f'**Normalize:**')
        st.write(ulasan[['rating', 'review']])
        ulasan['review'] = ulasan['review'].apply(tokenize)
        st.write(f'**Tokenize:**')
        st.write(ulasan[['rating', 'review']])
        ulasan['review'] = ulasan['review'].apply(stopword)
        st.write(f'**Stopword Remover:**')
        st.write(ulasan[['rating', 'review']])
        with st.spinner('Sedang Melakukan Stemming...'):
            ulasan['review'] = ulasan['review'].apply(stemming)
            st.write(f'**Stemming:**')
            st.write(ulasan[['rating', 'review']])
        ulasan['review'] = ulasan['review'].apply(join)
        ulasan['sentimen'] = ulasan['rating'].apply(label)
        ulasan = ulasan.dropna() 
        ulasan = fix_label(ulasan)
        
        st.write(f'**Hasil Preprocessing Text:**')
        st.write(ulasan[['sentimen', 'review']])
        directory = 'D:/Analisis Sentimen/Data/Preprocessing'
        file_path = os.path.join(directory, nama_file)
        ulasan.to_csv(file_path)
        st.info(f'Data {nama_file} Berhasil Disimpan')

if selected=='TF-IDF':
    st.title("TF-IDF")
    folder_path = 'D:/Analisis Sentimen/Data/Preprocessing'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    selected_file = st.selectbox('Select a CSV file', csv_files)
    file_path = os.path.join(folder_path, selected_file)
    ulasan = pd.read_csv(file_path, index_col=0)
    ulasan = ulasan[['review', 'sentimen']]
    st.write('#### **Ulasan Preprocessing:**')
    st.dataframe(ulasan)
    nama_file = st.text_input('Masukan Nama File Penyimpanan Hasil TFIDF')
    nama_model = st.text_input('Masukan Nama File Penyimpanan Model TFIDF')
    TFIDF = st.button("TF-IDF Vektorizer")
    if TFIDF:
        vec = SentenceTFIDFVectorizer()
        X = ulasan['review']
        y = ulasan['sentimen']
        tfidf_matrix, df_vocab = vec.fit_transform(X)
        st.write('#### **Nilai TF-IDF Setiap Kata:**')
        st.dataframe(df_vocab)
        st.write('#### **Hasil Matrix TF-IDF:**')
        tfidf_matrix = pd.DataFrame(tfidf_matrix)
        tfidf_matrix['sentimen'] = y
        st.dataframe(tfidf_matrix)
        directory = 'D:/Analisis Sentimen/Data/TF-IDF'
        file_path = os.path.join(directory, nama_file)
        tfidf_matrix.to_csv(file_path)
        directory2 = 'D:/Analisis Sentimen/Model/Model_TFIDF'
        file_path2 = os.path.join(directory2, nama_model)
        file = open(file_path2, 'wb')
        pickle.dump(vec, file)
        st.success(f'Data {nama_file} dan {nama_model} Berhasil Disimpan')
        

if selected =="Training":   
    st.title("Training")
    folder_path = 'D:/Analisis Sentimen/Data/TF-IDF'
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    selected_file = st.selectbox('Select a CSV file', csv_files)
    file_path = os.path.join(folder_path, selected_file)
    ulasan = pd.read_csv(file_path, index_col=0)
    st.write('#### **A. Ulasan Kraton:**')
    st.write(ulasan)
    st.write('#### **B. Pebadingan Label**')
    distribusi(ulasan)
    st.write('#### **C. Tunning Parameter SVM**')
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            training_size = st.slider('Jumlah Data Training (%):', min_value=1, max_value=100)
            kernel = st.selectbox('Pilih Jenis Kernel:',('rbf', 'poly', 'linear', 'sigmoid'))
        with col_2:
            c = st.selectbox('Pilih Nilai C:', (1, 10, 100))
            gamma = st.selectbox('Pilih Nilai Gamma:', ('scale', 'auto', 0.1, 1, 10, 100))
            balance = st.checkbox('Terapkan Class Weight:', )
    nama_model = st.text_input('Masukan Nama File Penyimpanan Model SVM')
    train = st.button("Training")
    if train:
        train_accuracy, test_accuracy, precision, recall, f1, y_true, y_pred, fpr, tpr, model = training(ulasan, training_size, kernel, c, gamma, balance)
        with st.container():
            col_1, col_2 = st.columns(2)
            with col_1:
                st.info(f'Akurasi Train: {train_accuracy} %' )
                st.info(f'Akurasi Test: {test_accuracy} %' )
                st.info(f'Precision: {precision} %' )
            with col_2:
                st.info(f'Recall: {recall} %' )
                st.info(f'F1-Score: {f1} %')
                st.success('Training Selesai!')
        st.write('#### **D. Confusion Matrix & ROC Curve**')
        with st.container():
            col_1, col_2 = st.columns(2)
            with col_1:
                conf_matrix(y_true, y_pred)
            with col_2:
                roc(fpr, tpr)

        directory = 'D:/Analisis Sentimen/Model/Model_SVM'
        file_path = os.path.join(directory, nama_model)
        file = open(file_path, 'wb')
        pickle.dump(model, file)
        st.success(f'Model {nama_model} Berhasil Disimpan')
            

if selected=='Prediction':
    st.title('Prediksi')
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            folder_path = 'D:/Analisis Sentimen/Model/Model_TFIDF'
            folder_tfidf = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]
            selected_tfidf = st.selectbox('Pilih Model TFIDF', folder_tfidf)
            file_path_tfidf = os.path.join(folder_path, selected_tfidf)
        with col_2:
            folder_path2 = 'D:/Analisis Sentimen/Model/Model_SVM'
            folder_svm = [file for file in os.listdir(folder_path2) if file.endswith('.pkl')]
            selected_svm = st.selectbox('Pilih Model SVM', folder_svm)
            file_path_svm = os.path.join(folder_path2, selected_svm)

    text = st.text_input('Masukan Text Ulasan:')
    prediksi = st.button('Predict')
    if prediksi:     
        text_prep = prep_all(text)
        file = open(file_path_tfidf, 'rb')
        vec = pickle.load(file)
        text_prep = vec.transform(text_prep['review'])
        file = open(file_path_svm, 'rb')
        model = pickle.load(file)
        pred = model.predict(text_prep)
        pred_prob = model.predict_proba(text_prep)
        rep = np.where(pred == 0, 'Negatif', 'Positif')
        st.info(f'Sentimen {rep[0]}')
        st.warning(f'Probabilitas Label Negatif: {round((pred_prob[0][0])*100, 2)}%')
        st.success(f'Probabilitas Label Positif: {round((pred_prob[0][1])*100, 2)}%')
        # list_aspek = aspek(prep_aspek(text))
        # st.info(f'Aspek: {list_aspek}') 
