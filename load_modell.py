import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
import os
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# Beispieltext
sentence_bsp = "This is a sample sentence to convert into vector."

def load_modell(sentence):

    npz_path = "keyed_vectors.npz"
    modell_path = r"C:\Users\Luis\Documents\Uni\Semester 4\ML4B\GoogleNews-vectors-negative300.bin"

    # Tokenisierung des Satzes
    tokens = word_tokenize(sentence.lower())

    # Prüfen, ob die .npz-Datei bereits existiert
    if not os.path.exists(npz_path):
        # Laden des vortrainierten Modells
        print("Lade das vortrainierte Modell...")
        word2vec_model = KeyedVectors.load_word2vec_format(modell_path, binary=True)
        
        # Extrahieren der Wörter und Vektoren
        vectors = word2vec_model.vectors
        words = word2vec_model.index_to_key
        
        # Speichern der Wörter und Vektoren in eine .npz-Datei
        np.savez(npz_path, vectors=vectors, words=words)
        print("Keyed Vectors wurden gespeichert.")


    # Laden der gespeicherten Keyed Vectors aus der .npz-Datei
    print("Lade die gespeicherten Keyed Vectors...")
    data = np.load(npz_path)
    vectors = data['vectors']
    words = data['words']
    print("Keyed Vectors wurden geladen.")


    # Berechnen der Word Embeddings für jedes Token im Satz
    #word_vectors = [word2vec_model.get_vector(word) for word in tokens if word in word2vec_model.key_to_index]
    word_vectors = [vectors[np.where(words == word)[0][0]] for word in tokens if word in words]

    # Berechnen des Durchschnitts der Word Embeddings
    if word_vectors:
        sentence_vector = sum(word_vectors) / len(word_vectors)
    else:
        # Wenn kein Wort im Modell vorhanden ist, dann wird der Satzvektor Null sein
        sentence_vector = [0] * word2vec_model.vector_size
    
    return sentence_vector

# Ergebnis: Der Satzvektor
print(load_modell(sentence_bsp))

"""
für die Konsole: 

pip install nltk
pip install gensim
pip install scipy==1.10.1

für das Modell 

https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

"""