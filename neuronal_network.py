import numpy as np
import pandas as pd

#Größe des Netzwerkes festlegen
gr_in=5
gr_l1=10
gr_l2=0
gr_l3=0
gr_out=2

#Daten einlesen
url_test =("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Testing_meme_dataset.csv")
url_train=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Training_meme_dataset.csv")
url_validation=("https://raw.githubusercontent.com/MariaAmPC/hate-speach/main/Validation_meme_dataset.csv")

df_test= pd.read_csv(url_test,index_col=0)
df_train= pd.read_csv(url_train,index_col=0)
df_validation= pd.read_csv(url_validation,index_col=0)

#Daten in Array schreiben
sentences = np.transpose(df_train.head(50)['sentence'].values)
labels = np.transpose(df_train.head(50)['label'].values)

sentences_test=np.random.uniform(0,20,(1,5)) #Test bis "sentences" richtig formatiert

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

count=3 #Anzahl der Durchläufe

for counter in range(count):
    for sentence,label in zip(sentences_test,labels):

        #foreward propagation
        l1_pre = b_i_l1 + w_i_l1 @ sentence
        l1 = 1/ (1 + np.exp(-l1_pre))
        print(l1)