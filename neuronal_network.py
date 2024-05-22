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


#Größe des Netzwerkes festlegen (size = Anzahl der hiddenlayer + Input und Output)
size=3
gr=[0]*size

gr[0]=300
gr[1]=30 
gr[2]=6

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
"""
-------EINLESEN DATEN-------
"""
"""
#Einlesen von anderem Datensatz, nur zum testen des NN. Wird danach wieder gelöscht
def get_mnist():
    url = "https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/mnist.npz"
    response = requests.get(url)
    content = BytesIO(response.content)
    # Die Datei öffnen
    with np.load(content) as f:
        # Daten aus der Datei laden
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

images,imlabels=get_mnist()
images=images[:500]
imlabels=imlabels[:500]
"""
#Daten einlesen
url_test =("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Testing_meme_dataset.csv")
url_train=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Training_meme_dataset.csv")
url_validation=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Validation_meme_dataset.csv")

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

#df_test= pd.read_csv(url_test,index_col=0)
#df_train= pd.read_csv(url_train,index_col=0)
#df_validation= pd.read_csv(url_validation,index_col=0)
    
###Sätze einlesen die vektoren haben später eine größe von 300

#Corpus definieren auf dem die Gewichte basieren
corpus = df_test['content']

#initialisierung des TF-IDF-Vektorisierers
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Laden des vortrainierten Word2Vec-Modells (sollte später nicht lokal passieren)

word2vec_model = KeyedVectors.load_word2vec_format(r"C:\Users\49170\Documents\FAU\ML4B\GoogleNews-vectors-negative300.bin", binary=True)

#funktion die uns den Vektor des Satztes returned 
def get_sentence_vector(sentence):
    tokens = word_tokenize(sentence.lower())
    tfidf_vector = tfidf_vectorizer.transform([" ".join(tokens)]) #verwandeln des satzes in vertikales array
    tfidf_scores = {word: tfidf_vector[0, tfidf_feature_names.tolist().index(word)] for word in tokens if word in tfidf_feature_names} #gewichtung der einzelnen wörter erfahren
    
    word_vectors = []
    for word in tokens:
        if word in word2vec_model.key_to_index and word in tfidf_scores:
            word_vector = word2vec_model.get_vector(word) * tfidf_scores[word] #vektor aus dem word2vec_model mit dem gewichtetem vektor aus dem TF system
            word_vectors.append(word_vector)
    
    #durchschnitt der gewichteten vektoren erlangen
    if word_vectors:
        sentence_vector = np.mean(word_vectors, axis=0)
    else:
        sentence_vector = np.zeros(word2vec_model.vector_size)
    
    return sentence_vector

sentences=np.empty((0,300))
for i in df_train['content'].values:
    sentences = np.insert(sentences, len(sentences), get_sentence_vector(i),axis=0)
#print(sentences)

  
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
learnrate = 0.005

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