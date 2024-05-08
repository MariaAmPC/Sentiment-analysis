import numpy as np
import pandas as pd
import pathlib
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#Größe des Netzwerkes festlegen
gr_in=5
gr_l1=10
gr_out=2


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
images=images[:50]
imlabels=imlabels[:50]

#Daten einlesen
url_test =("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Testing_meme_dataset.csv")
url_train=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Training_meme_dataset.csv")
url_validation=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Validation_meme_dataset.csv")


df_test= pd.read_csv(url_test,index_col=0)
df_train= pd.read_csv(url_train,index_col=0)
df_validation= pd.read_csv(url_validation,index_col=0)

def sigmoid(value): 
    return 1/ (1 + np.exp(-value))

def forward(bias, weight, x): 
    pre= bias+ weight @ x
    return(sigmoid(pre))
    

"""""
sentences=[]
for i in df_train.head(5)['sentence'].values:
    sec=[]
    for n in i: 
        sec=np.append(sec,ord(n))
   # print(sec)
    sentences.append(sec)
print(sentences)

"""
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

#Gewichte der Neuronenverbindugnen zufällig festlegen
w_i_l1 = np.random.uniform(-0.5,0.5,(gr_l1,gr_in))
w_l1_o = np.random.uniform(-0.5,0.5,(gr_out,gr_l1))


#Gewichte der Biases auf 0 setzen
b_i_l1 = np.zeros((gr_l1,1))
b_l1_o = np.zeros((gr_out,1))

count = 10 #Anzahl der Durchläufe
correct = 0 #Anzahl korrekten Ergebnisse
learnrate = 0.01

for counter in range(count):
    for sentence,label in zip(sentences,labels):

        sentence.shape+=(1,)
        label.shape+=(1,)
        #print(sentence)
        #print(label)

        #foreward propagation
        l1 = forward(b_i_l1, w_i_l1, sentence)
        out = forward(b_l1_o, w_l1_o, l1)
    
        #print(out)

        #Error-wert berechnen
        err = 1/len(out) * np.sum((out - label)**2, axis=0)
        correct += int(np.argmax(out) == np.argmax(label))
        #print(err)

        #Derivative berechnen: Backpropagation-weights * Ableitung von Sigmoid
        delta_out = out - label
        w_l1_o += -learnrate* delta_out @ np.transpose(l1)
        b_l1_o += -learnrate* delta_out

        delta_l1 = np.transpose(w_l1_o) @ delta_out * (l1*(1-l1))
        w_i_l1 += -learnrate* delta_l1 @ np.transpose(sentence)
        b_i_l1 += -learnrate* delta_l1
        

        print(label)
        print(print(out))
        print(int(np.argmax(out) == np.argmax(label)))

    #Ausgeben der Genauigkeit nach jeder Iteration    
    print(f"Acc: {round((correct/sentences.shape[0])*100,2)} %")
    correct=0
""""
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

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
      """  