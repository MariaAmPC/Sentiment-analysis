import streamlit as st
import numpy as np

# Größe des neuronalen Netzwerks festlegen
size = 3
gr = [0] * size
gr[0] = 784
gr[1] = 30
gr[2] = 10

# Gewichte und Biases festlegen: Weights zufällig, Biases auf 0
weight = [0] * size
bias = [0] * size
for i in range(size - 1):
    weight[i] = np.random.uniform(-0.5, 0.5, (gr[i + 1], gr[i]))
    bias[i] = np.zeros((gr[i + 1], 1))

# Methode zur Vorwärtspropagierung definieren
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def forward(bias, weight, x):
    pre = bias + weight @ x
    return sigmoid(pre)

# Methode zur Vorhersage definieren
# Methode zur Vorhersage definieren
def predict(sentence):
    # Eingabedaten initialisieren
    input_data = np.zeros((784, 1))
    
    # Konvertiere den Text in das numerische Format und fülle die Eingabedaten
    for i, char in enumerate(sentence):
        if i < 784:  # Nur die ersten 784 Zeichen des Textes verwenden
            input_data[i, 0] = ord(char)

    l = [0] * size
    for i in range(1, size):
        if i == 1:
            l[1] = forward(bias[0], weight[0], input_data)
        else:
            l[i] = forward(bias[i - 1], weight[i - 1], l[i - 1])

    if l[size - 1][0] > l[size - 1][1]:
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
