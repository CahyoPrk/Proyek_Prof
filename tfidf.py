import numpy as np
import math
import pandas as pd

class SentenceTFIDFVectorizer:
    def __init__(self):
        self.vocab = None
        self.idf_vector = None

    def fit_transform(self, sentences):
        # Step 1: Tokenize kalimat
        self.vocab = list(set().union(*[sentence.split() for sentence in sentences]))

        # Step 2: Hitung term frequency (TF)
        tf_matrix = np.zeros((len(sentences), len(self.vocab)))

        for i, sentence in enumerate(sentences):
            for word in sentence.split():
                if word in self.vocab:
                    tf_matrix[i, self.vocab.index(word)] += 1

        # Step 3: Hitung inverse document frequency (IDF)
        self.idf_vector = np.zeros(len(self.vocab))

        for i, word in enumerate(self.vocab):
            for sentence in sentences:
                if word in sentence.split():
                    self.idf_vector[i] += 1

        self.idf_vector = np.log10(len(sentences) / (1 + self.idf_vector))

        # Step 4: Hitung TF-IDF
        tf_idf_matrix = tf_matrix * self.idf_vector
        
        df_data = {'vocab': self.vocab, 'tf': np.sum(tf_matrix, axis=0),
                   'idf': self.idf_vector, 'tf_idf': np.sum(tf_idf_matrix, axis=0)}
        df = pd.DataFrame(df_data)
        return tf_idf_matrix, df

    def transform(self, sentences):
        # Step 1: Hitung term frequency (TF)
        tf_matrix = np.zeros((len(sentences), len(self.vocab)))

        for i, sentence in enumerate(sentences):
            for word in sentence.split():
                if word in self.vocab:
                    tf_matrix[i, self.vocab.index(word)] += 1

        # Step 2: Hitung TF-IDF
        tf_idf_matrix = tf_matrix * self.idf_vector

        return tf_idf_matrix