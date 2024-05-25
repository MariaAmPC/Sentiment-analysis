import numpy as np

# Laden Sie die npz-Datei
loaded_model = np.load('neuronal_network.npz')

# Überprüfen Sie die vorhandenen Schlüssel
print(loaded_model.keys())

# Überprüfen Sie, ob ein bestimmter Schlüssel vorhanden ist
if 'w2' in loaded_model.keys():
    print("Key 'w2' existiert in der npz-Datei.")
else:
    print("Key 'w2' existiert nicht in der npz-Datei.")
