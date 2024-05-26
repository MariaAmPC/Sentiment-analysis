import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

# Laden des trainierten Modells und der benötigten Daten
loaded_model = np.load('neuronal_network.npz')
size = int(len(loaded_model)/2)+1
weight = [loaded_model[f'w{i}'] for i in range(size-1)]
bias = [loaded_model[f'b{i}'] for i in range(size-1)]
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

emotions = ["neutral", "worry", "happiness", "sadness", "love", "hate"]

# Laden des Datensatzes
#df = pd.read_csv("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/tweet_emotions.csv")
#df = df[df.sentiment.isin(emotions)]

# Trainingsdaten und Testdaten aufteilen
#df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)

# Funktionen für Vorhersage und Textkodierung
def sigmoid(value): 
    return 1 / (1 + np.exp(-value))

def forward(bias, weight, x): 
    pre = bias + np.dot(weight, x)
    return sigmoid(pre)

def encode_text(text):
    return bert_model.encode([text])

def predict_hate_speech(sentence):
    encoded_sentence = encode_text(sentence)
    l = [0] * size
    for i in range(1, size):
        if i == 1:
            l[1] = forward(bias[0], weight[0], encoded_sentence.T)
        else:
            l[i] = forward(bias[i-1], weight[i-1], l[i-1])

    prediction = np.argmax(l[size-1])
    return emotions[prediction]

# Streamlit App
st.title('Tweet Sentiment Analysis')

# Auswahl eines Textes aus dem Datensatz
#st.sidebar.subheader('Wähle einen Text aus dem Datensatz')
#selected_text = st.sidebar.selectbox('Text auswählen', df_train['content'])

#if st.sidebar.button('Predict für ausgewählten Text'):
    #prediction = predict_hate_speech(selected_text)
    #st.write('Prediction:', prediction)

# Texteingabe für eigene Vorhersage
st.subheader('Enter your own Text in English')
user_text = st.text_area('enter Text here we classify between ["neutral", "worry", "happiness", "sadness", "love", "hate"]')
if st.button('Predicting the Sentiment'):
    prediction = predict_hate_speech(user_text)
    st.write('Prediction:', prediction)
