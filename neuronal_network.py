import numpy as np
import pandas as pd
import pathlib
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#Größe des Netzwerkes festlegen (size = Anzahl der hiddenlayer + Input und Output)
size=3
gr=[0]*size

gr[0]=5
gr[1]=10
gr[2]=2

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

#Daten einlesen
url_test =("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Testing_meme_dataset.csv")
url_train=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Training_meme_dataset.csv")
url_validation=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Validation_meme_dataset.csv")


df_test= pd.read_csv(url_test,index_col=0)
df_train= pd.read_csv(url_train,index_col=0)
df_validation= pd.read_csv(url_validation,index_col=0)
    
#Sätze einlesen
sentences=np.empty((0,5))
for i in df_train.head(50)['sentence'].values:
    sentences = np.insert(sentences, len(sentences), np.array([[ord(i[0]),ord(i[1]),ord(i[2]),ord(i[3]),ord(i[4])]]),axis=0)
print(sentences)
  
#Labels einlesen
labels=np.empty((0,2))
for i in df_train.head(50)['label'].values:
    if i == "offensive":
        newrow = np.array([[1,0]])
    elif i == "Non-offensiv":
        newrow = np.array([[0,1]])
    else:
        print("ERROR")
    labels = np.insert(labels, len(labels), newrow, axis=0)


"""
-------START NN-------
"""

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



"""
-------TESTEN NN-------
"""

"""
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    
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