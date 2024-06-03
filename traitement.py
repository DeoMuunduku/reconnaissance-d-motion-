import pandas as pd  # Importation de la bibliothèque pandas pour la manipulation des données
import cv2  # Importation de la bibliothèque OpenCV pour la vision par ordinateur
import numpy as np  # Importation de la bibliothèque NumPy pour les opérations sur les tableaux

# Chemin vers le fichier CSV contenant le dataset FER2013
dataset_path = 'fer2013/fer2013/fer2013.csv'
# Taille des images
image_size = (48, 48)

def load_fer2013_data():
    """Charge et prétraite les données du dataset FER2013."""
    data = pd.read_csv(dataset_path)  # Lecture des données à partir du fichier CSV
    pixel_values = data['pixels'].tolist()  # Extraction de la colonne des pixels
    faces = []
    
    # Transformation des séquences de pixels en images
    for pixel_sequence in pixel_values:
        face = np.array([int(pixel) for pixel in pixel_sequence.split(' ')])  # Conversion des pixels en tableau
        face = face.reshape(image_size)  # Mise en forme de l'image en 48x48
        face = cv2.resize(face.astype('uint8'), image_size)  # Redimensionnement de l'image
        faces.append(face.astype('float32'))  # Ajout de l'image à la liste des visages
    
    # Conversion des listes en tableaux NumPy
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # Ajout d'une dimension pour la compatibilité avec le modèle
    emotions = pd.get_dummies(data['emotion']).values  # Conversion des étiquettes d'émotions en one-hot encoding
    return faces, emotions  # Retour des images et des émotions

def preprocess_images(x, version2=True):
    """Prétraite les images pour les rendre compatibles avec le modèle."""
    x = x.astype('float32')  # Conversion des valeurs en float32
    x = x / 255.0  # Normalisation des valeurs de pixels entre 0 et 1
    if version2:
        x = (x - 0.5) * 2.0  # Si version2 est True, centrer les valeurs autour de 0 et les étendre à [-1, 1]
    return x  # Retour des images prétraitées

