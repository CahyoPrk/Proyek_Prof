from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from tfidf import *
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics


def training(df, training_size, kernel, c, gamma, balance):
    df = df.dropna()
    X = df.drop(columns=['sentimen'])
    y = df['sentimen']

    test_size = (100 - training_size)/100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123, stratify=y)

    if balance == True:
        balance = 'balanced'
    else:
        balance = None
    svm = SVC(probability=True, kernel=kernel, C=c, gamma=gamma, class_weight=balance   )
    model = svm.fit(X_train, y_train)
    pred = model.predict(X_test)

    # akurasi data Training
    train_score = round(model.score(X_train, y_train)*100,2)
    accuracy = round(model.score(X_test, y_test)*100, 2)
    precision = round(precision_score(y_test, pred)*100, 2)
    recall = round(recall_score(y_test, pred)*100, 2)
    f1 = round(f1_score(y_test, pred)*100, 2)

    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    return train_score, accuracy, precision, recall, f1, y_test, pred, fpr, tpr, model