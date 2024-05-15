import streamlit as st
import numpy as np

#Laden des trainierten Modells
network = np.load('neuronal_network.npz')

leng = int(len(network)/2)

weight=[0]*leng
bias=[0]*leng

for i in range(leng):
    weight[i]=network[f'w{i}']
    bias[i]=network[f'b{i}']


# Methode zur Vorwärtspropagierung definieren
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def forward(bias, weight, x):
    pre = bias + weight @ x
    return sigmoid(pre)


# Methode zur Vorhersage definieren
def predict(sentence):
    # Eingabedaten initialisieren
    input_data = np.zeros((5, 1))
    
    # Konvertiere den Text in das numerische Format und fülle die Eingabedaten
    for i, char in enumerate(sentence):
        if i < 5:  # Nur die ersten 784 Zeichen des Textes verwenden
            input_data[i, 0] = ord(char)

    
    l = [0] * leng
    for i in range(1, leng):
        if i == 1:
            l[1] = forward(bias[0], weight[0], input_data)
        else:
            l[i] = forward(bias[i - 1], weight[i - 1], l[i - 1])

    prediction = np.argmax(l[leng - 1])
    
    if prediction == 0:
        return "offensiv"
    else:
        return "nicht offensiv" 



# Streamlit App definieren
def main():
    st.title("Offensivitätsvorhersage")

    # Eingabefeld für den Text
    text = st.text_area("Geben Sie Ihren Text hier ein:")

    # Vorhersage machen, wenn Text eingegeben wird
    if st.button("Vorhersage"):
        if text:
            prediction = predict(text)
            st.write(f"Der Text wurde als {prediction} eingestuft.")
        else:
            st.write("Bitte geben Sie einen Text ein.")

if __name__ == "__main__":
    main()
