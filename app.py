import streamlit as st
import numpy as np
from neuronal_network import w_i_l1, w_l1_o, b_i_l1, b_l1_o
import pandas as pd

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

def predict (sentence):

    # Eingabedaten initialisieren
    input_data = np.empty((5, 1))
    
    # Konvertiere den Satz in das numerische Format und fülle die Eingabedaten
    for i, char in enumerate(sentence[:5]):  # Nur die ersten fünf Zeichen des Satzes
        input_data[i, 0] = ord(char)

    # Führe die Vorwärtspropagierung durch
    l1 = forward(b_i_l1, w_i_l1, input_data)
    out = forward(b_l1_o, w_l1_o, l1)


    if out[0] > out [1]:
        return "offensive"
    else:
        return "non-offensive"
    

def main():
    st.title("Hate speech detection")

    # Eingabefeld für den Satz
    sentence = st.text_input("Geben Sie einen Satz ein:")

    # Vorhersage machen, wenn ein Satz eingegeben wird
    if st.button("Vorhersage"):
        if sentence:
            prediction = predict(sentence)
            st.write(f"Der Satz wurde als {prediction} eingestuft.")
        else:
            st.write("Bitte geben Sie einen Satz ein.")

if __name__ == "__main__":
    main()