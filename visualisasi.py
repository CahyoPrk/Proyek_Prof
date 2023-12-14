import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def distribusi(sentimen):
    label_counts = sentimen['sentimen'].value_counts()
    fig, ax = plt.subplots()
    fig.set_figheight(6)  # Mengatur tinggi gambar menjadi 6 inci
    fig.set_figwidth(8)  # Mengatur lebar gambar menjadi 8 inci
    label_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Label')
    ax.set_ylabel('Jumlah')
    ax.set_title('Perbandingan Jumlah Label')

    # Menambahkan nilai aktual pada setiap batang
    for i, count in enumerate(label_counts):
        ax.annotate(str(count), xy=(i, count), ha='center', va='bottom')

    plt.show()
    return st.pyplot(fig)

def conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    return st.pyplot(plt)

def roc(fpr, tpr):
    plt.clf()
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return st.pyplot(plt)