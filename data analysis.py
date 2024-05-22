import numpy as np
import pandas as pd
import pathlib
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

df=pd.read_csv(r"C:\Users\49170\Documents\FAU\ML4B\tweet_emotions.csv")
df = df[df.sentiment != 'empty']
df = df[df.sentiment != 'enthusiasm']
df = df[df.sentiment != 'fun']
df = df[df.sentiment != 'boredom']
df = df[df.sentiment != 'anger']
df = df[df.sentiment != 'relief']
df = df[df.sentiment != 'surprise']


stats = Counter(df["sentiment"])
print(dict(stats))


df_test, df_train = train_test_split(df , test_size=0.33, random_state=42)

print(df_test.size)