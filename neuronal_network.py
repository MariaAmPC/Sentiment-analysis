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
from sentence_transformers import SentenceTransformer
import functools
from sentence_transformers import SentenceTransformer



#Größe des Netzwerkes festlegen (size = Anzahl der hiddenlayer + Input und Output)
size=4
gr=[0]*size

gr[0]=768
gr[1]=30 
gr[2]=10
gr[3]=6

#Weigths und Biases festlegen: Weights zufällig, Biases auf 0
weight=[0]*size
bias=[0]*size
for i in range(size-1):
    weight[i] = np.random.uniform(-0.5,0.5,(gr[i+1],gr[i]))
    bias[i] = np.zeros((gr[i+1],1))
    

#Methoden definieren
def sigmoid(value): 
    return 1/ (1 + np.exp(-value))

def forward(bias, weight, x): 
    pre= bias+ weight @ x
    return(sigmoid(pre))

def getgr():
    return gr[0]

#Daten einlesen
df=pd.read_csv(r"https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/tweet_emotions.csv")

df = df[df.sentiment != 'empty']
df = df[df.sentiment != 'enthusiasm']
df = df[df.sentiment != 'fun']
df = df[df.sentiment != 'boredom']
df = df[df.sentiment != 'anger']
df = df[df.sentiment != 'relief']
df = df[df.sentiment != 'surprise']

df_test, df_train = train_test_split(df , test_size=0.33, random_state=42)

#BERT modell zum sätze einlesen
class BertModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer('bert-base-nli-mean-tokens')
        return cls._instance

model = BertModelSingleton.get_instance()
sentences = model.encode(df_train["content"].tolist())
  
#Labels einlesen
labels=np.empty((0,6))
for i in df_train['sentiment'].values:
    if i == "neutral":
        newrow = np.array([[1,0,0,0,0,0]])
    elif i == "worry":
        newrow = np.array([[0,1,0,0,0,0]])
    elif i == "happiness":
        newrow = np.array([[0,0,1,0,0,0]])
    elif i == "sadness":
        newrow = np.array([[0,0,0,1,0,0]])
    elif i == "love":
        newrow = np.array([[0,0,0,0,1,0]])
    elif i == "hate":
        newrow = np.array([[0,0,0,0,0,1]])
    else:
        print("ERROR" + i)
    labels = np.insert(labels, len(labels), newrow, axis=0)




#-------START NN-------


epoch = 20 #Anzahl der Epochen
correct = 0 #Anzahl korrekte Ergebnisse
count = 0 #Anzahl Durchläufe pro Epoche bzw. Testgröße
learnrate = 0.01

for epoche in range(epoch):
    w_change=[0]*size
    b_change=[0]*size
    
    for sentence,label in zip(sentences,labels):

        sentence.shape+=(1,)
        label.shape+=(1,)

        #foreward propagation
        l=[0]*size
        for i in range(1, size):
            if i == 1:
                l[1] = forward(bias[0], weight[0], sentence)
            else:
                l[i] = forward(bias[i-1], weight[i-1], l[i-1])
    

        #Error-wert berechnen
        err = 1/len(l[size-1]) * np.sum((l[size-1] - label)**2, axis=0)
        correct += int(np.argmax(l[size-1]) == np.argmax(label))
        count+=1
        
        #Backpropagation Start
        
        #Zwischenspeichern der Weights
        w_final = [i for i in weight]
 
        delta_l = [0]*size
        for i in range (size-1, 0, -1):
            if i == size-1:
                delta_l[i] = l[size-1] - label
            else:
                #Derivative berechnen: Backpropagation-weights * Ableitung von Sigmoid
                delta_l[i] = np.transpose(w_final[i]) @ delta_l[i+1] * (l[i]*(1-l[i]))
            if i == 1:
                w_change[i-1] += -learnrate* delta_l[i] @ np.transpose(sentence)
            else:
                #Vorüberhende Speicherung der Weight-Änderungen in w_change bevor nach Epoche Durchschnitt genommen wird
                w_final[i-1] += -learnrate* delta_l[i] @ np.transpose(l[i-1])
                w_change[i-1] += -learnrate* delta_l[i] @ np.transpose(l[i-1])

            b_change[i-1] += -learnrate* delta_l[i]


    #Updaten der Weight/ Biases nach Abschluss einer Epoche: Durchschnitt nehmen
    for i in range(size):
        weight[i] += w_change[i] / count
        bias[i] += b_change[i] / count
 

    #Ausgeben der Genauigkeit nach jeder Iteration    
    print(f"Acc: {round((correct/count)*100,2)} %")
    correct=0
    count=0

#trainiertes Modell speichern
dict={}
for i in range(size-1):
    dict[f'w{i}']=weight[i]
    dict[f'b{i}']=bias[i]
np.savez('neuronal_network.npz', **dict)




#-------TESTEN NN-------


"""
while True:
    index = int(input("Enter a number (0 - 65597): "))
    sentance = content[index]
    print(sentance)

    sentance.shape += (1,)
    
    # Forward propagation input -> hidden
    h_pre = b_i_l1 + w_i_l1 @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_l1_o + w_l1_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    
    l=[0]*size
    for i in range(1, size):
        if i == 1:
            l[1] = forward(bias[0],weight[0],img.reshape(784, 1))
        else:
            l[i] = forward(bias[i-1], weight[i-1], l[i-1])

    plt.title(f"Subscribe if its a {l[size-1].argmax()} :)")
    plt.show()

    """