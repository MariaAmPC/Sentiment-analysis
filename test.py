import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# Beispieltext
sentence = "This is a sample sentence to convert into vector."

# Tokenisierung des Satzes
tokens = word_tokenize(sentence.lower())

# Bereitstellen der Word2Vec-Modelldatei (ersetze "path/to/word2vec_model.bin" durch den Pfad zu deinem Word2Vec-Modell)
#word2vec_model = Word2Vec.load(r"C:\Users\49170\Documents\FAU\ML4B\GoogleNews-vectors-negative300.bin")
word2vec_model = KeyedVectors.load_word2vec_format(r"C:\Users\49170\Documents\FAU\ML4B\GoogleNews-vectors-negative300.bin", binary=True)

# Berechnen der Word Embeddings für jedes Token im Satz
word_vectors = [word2vec_model.get_vector(word) for word in tokens if word in word2vec_model.key_to_index]

# Berechnen des Durchschnitts der Word Embeddings
if word_vectors:
    sentence_vector = sum(word_vectors) / len(word_vectors)
else:
    # Wenn kein Wort im Modell vorhanden ist, dann wird der Satzvektor Null sein
    sentence_vector = [0] * word2vec_model.vector_size

# Ergebnis: Der Satzvektor
print(sentence_vector)

"""
für die Konsole: 

pip install nltk
pip install gensim
pip install scipy==1.10.1

für das Modell 

https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

"""