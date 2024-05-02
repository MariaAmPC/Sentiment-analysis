import numpy as np
import pandas as pd

#Größe des Netzwerkes festlegen
gr_in=5
gr_l1=10
gr_l2=10
gr_l3=10
gr_out=2

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

print(labels)
#Gewichte der Neuronenverbindugnen zufällig festlegen
w_i_l1 = np.random.uniform(-0.5,0.5,(gr_l1,gr_in))
w_l1_l2 = np.random.uniform(-0.5,0.5,(gr_l2,gr_l1))
w_l2_l3 = np.random.uniform(-0.5,0.5,(gr_l3,gr_l2))
w_l3_o = np.random.uniform(-0.5,0.5,(gr_out,gr_l3))

#Gewichte der Biases auf 0 setzen
b_i_l1 = np.zeros((gr_l1,1))
b_l1_l2 = np.zeros((gr_l2,1))
b_l2_l3 = np.zeros((gr_l3,1))
b_l3_o = np.zeros((gr_out,1))

count = 10 #Anzahl der Durchläufe
correct = 0 #Anzahl korrekten Ergebnisse
learnrate = .01

for counter in range(count):
    for sentence,label in zip(sentences,labels):

        sentence.shape+=(1,)
        label.shape+=(1,)
        #print(sentence)
        #print(label)

        #foreward propagation
        l1_pre = b_i_l1 + w_i_l1 @ sentence
        l1 = 1/ (1 + np.exp(-l1_pre))

        l2_pre = b_l1_l2 + w_l1_l2 @ l1
        l2 = 1/ (1 + np.exp(-l2_pre))

        l3_pre = b_l2_l3 + w_l2_l3 @ l2
        l3 = 1/ (1 + np.exp(-l3_pre))
        
        out_pre = b_l3_o + w_l3_o @ l3
        out = 1/ (1 + np.exp(-out_pre))
    
        #print(out)

        #Error-wert berechnen
        err = 1/len(out) * np.sum((out - label)**2, axis=0)
        correct += int(np.argmax(out) == np.argmax(label))
        #print(err)

        #Derivative berechnen: Backpropagation-weights * Ableitung von Sigmoid
        delta_out = out - label
        w_l3_o += -learnrate* delta_out @ np.transpose(l3)
        b_l3_o += -learnrate* delta_out

        delta_l3 = np.transpose(w_l3_o) @ delta_out * (l3*(l3-1))
        w_l2_l3 += -learnrate* delta_l3 @ np.transpose(l2)
        b_l2_l3 += -learnrate* delta_l3

        delta_l2 = np.transpose(w_l2_l3) @ delta_l3 * (l2*(l2-1))
        w_l1_l2 += -learnrate* delta_l2 @ np.transpose(l1)
        b_l1_l2 += -learnrate* delta_l2

        delta_l1 = np.transpose(w_l1_l2) @ delta_l2 * (l1*(l1-1))
        w_i_l1 += -learnrate* delta_l1 @ np.transpose(sentence)
        b_i_l1 += -learnrate* delta_l1

    #Ausgeben der Genauigkeit nach jeder Iteration    
    print(f"Acc: {round((correct/len(sentences))*100,2)} %")
    correct=0
 

        