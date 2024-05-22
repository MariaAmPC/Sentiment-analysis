import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

url_test =("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Testing_meme_dataset.csv")
df_test= pd.read_csv(url_test,index_col=0)

# Beispieltextkorpus (ersetze dies durch deinen Datensatz)
corpus = df_test['sentence']

# Initialisierung des TF-IDF-Vektorisierers
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Laden des vortrainierten Word2Vec-Modells
word2vec_model = KeyedVectors.load_word2vec_format(r"C:\Users\49170\Documents\FAU\ML4B\GoogleNews-vectors-negative300.bin", binary=True)

# Funktion zur Berechnung des Satzvektors
def get_sentence_vector(sentence, tfidf_vectorizer, tfidf_feature_names, word2vec_model):
    tokens = word_tokenize(sentence.lower())
    tfidf_vector = tfidf_vectorizer.transform([" ".join(tokens)])
    tfidf_scores = {word: tfidf_vector[0, tfidf_feature_names.tolist().index(word)] for word in tokens if word in tfidf_feature_names}
    
    word_vectors = []
    for word in tokens:
        if word in word2vec_model.key_to_index and word in tfidf_scores:
            word_vector = word2vec_model.get_vector(word) * tfidf_scores[word]
            word_vectors.append(word_vector)
    
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(word2vec_model.vector_size)
    
    return sentence_vector

# Berechnen der Vektoren für jeden Satz im Korpus
#sentence_vectors = [get_sentence_vector(sentence, tfidf_vectorizer, tfidf_feature_names, word2vec_model) for sentence in corpus]
print(get_sentence_vector(df_test["sentence"][1], tfidf_vectorizer, tfidf_feature_names, word2vec_model).size)
print(get_sentence_vector(df_test["sentence"][12], tfidf_vectorizer, tfidf_feature_names, word2vec_model).size)
print(get_sentence_vector(df_test["sentence"][69], tfidf_vectorizer, tfidf_feature_names, word2vec_model).size)
