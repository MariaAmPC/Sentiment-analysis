import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
transformer = 'nreimers/albert-small-v2'

# Load custom CSS
def load_css(css_file_path):
    with open(css_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('style.css')

# Funktion zur Textbereinigung
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # URLs entfernen
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Nicht-Alphabetische Zeichen entfernen
    text = text.lower()  # Kleinbuchstaben
    text = text.split()  # In Wörter teilen
    text = [word for word in text if word not in stop_words]  # Stopwörter entfernen
    return text

# Funktion zur Wortfrequenzanalyse
def get_top_n_words(corpus, n=None):
    counter = Counter()
    for text in corpus:
        counter.update(text)
    return counter.most_common(n)

# Balkendiagramm für die häufigsten Wörter pro Sentiment-Klasse
def plot_top_words(top_words_per_sentiment):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)

    for ax, (sentiment, top_words) in zip(axes.flatten(), top_words_per_sentiment.items()):
        words = list(top_words.keys())
        counts = list(top_words.values())
        ax.barh(words, counts)
        ax.set_title(sentiment)
        ax.set_xlabel('Häufigkeit')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

# Tortendiagramm für Sentiment-Verteilung
def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Balkendiagramm für die Häufigkeit der Sentiment-Klassen
def plot_sentiment_class_distribution(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Häufigkeit')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Laden des trainierten Modells und der benötigten Daten
loaded_model = np.load('neuronal_network.npz')
size = int(len(loaded_model)/2)+1
weight = [loaded_model[f'w{i}'] for i in range(size-1)]
bias = [loaded_model[f'b{i}'] for i in range(size-1)]
bert_model = SentenceTransformer(transformer)

emotions = ["neutral", "worry", "happiness", "sadness", "love", "hate"]

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

# Sidebar
st.sidebar.header('Navigation')
selected_option = st.sidebar.radio('Choose an action', ['Predict from Dataset', 'Enter Custom Text', 'Dataset'])

if selected_option == 'Predict from Dataset':
    df = pd.read_csv("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/tweet_emotions.csv")
    df = df[df.sentiment.isin(emotions)]
    st.subheader('Select a Text from the Dataset')
    selected_text = st.selectbox('Select text: here we classify between ["neutral", "worry", "happiness", "sadness", "love", "hate"] ', df['content'])
    if st.button('Predict sentiment for selected text'):
        prediction = predict_hate_speech(selected_text)
        st.write('Prediction:', prediction)
        
if selected_option == 'Enter Custom Text':
    st.subheader('Enter your own Text in English')
    user_text = st.text_area('Enter text: here we classify between ["neutral", "worry", "happiness", "sadness", "love", "hate"]')
    
    if user_text.strip() == '':
        if st.button('Predicting the Sentiment'):
            st.error('Error: Please enter some text before predicting the sentiment.')
    elif st.button('Predicting the Sentiment'):
        prediction = predict_hate_speech(user_text)
        st.write('Prediction:', prediction)

elif selected_option == 'Dataset':
    df = pd.read_csv("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/tweet_emotions.csv")
    df = df[df.sentiment.isin(emotions)]

    # Sentiment-Verteilung
    st.subheader('Sentiment Distribution in the Dataset')
    plot_sentiment_distribution(df)

    # Text bereinigen und in Listenform bringen
    df['cleaned_content'] = df['content'].apply(clean_text)

    # Häufigkeit der Sentiment-Klassen anzeigen
    st.subheader('Frequency of sentiment classes in the data set')
    plot_sentiment_class_distribution(df)

    # häufigste Wörter pro Sentiment
    top_words_per_sentiment = {}
    for sentiment in emotions:
        corpus = df[df['sentiment'] == sentiment]['cleaned_content']
        top_words = get_top_n_words(corpus, 10)
        top_words_per_sentiment[sentiment] = dict(top_words)

    st.subheader('Most frequent words in tweets according to sentiment categories')
    plot_top_words(top_words_per_sentiment)
