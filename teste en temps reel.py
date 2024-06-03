#deo munduku , hanoi , 22/12/2023


from keras.preprocessing.image import img_to_array  # Importation de la fonction pour convertir une image en tableau
import imutils  # Importation de la bibliothèque imutils pour faciliter les manipulations d'images
import cv2  # Importation de la bibliothèque OpenCV pour la vision par ordinateur
from keras.models import load_model  # Importation de la fonction pour charger un modèle pré-entraîné
import numpy as np  # Importation de la bibliothèque NumPy pour les opérations sur les tableaux

# Chemins vers les modèles de détection et de classification des émotions
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Chargement des modèles
face_detector = cv2.CascadeClassifier(detection_model_path)  # Chargement du modèle de détection de visages
emotion_model = load_model(emotion_model_path, compile=False)  # Chargement du modèle de classification des émotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]  # Liste des émotions

# Initialisation de la capture vidéo
cv2.namedWindow('your_face')  # Création d'une fenêtre pour afficher la vidéo
video_capture = cv2.VideoCapture(0)  # Initialisation de la capture vidéo depuis la webcam

while True:  # Boucle infinie pour traiter les images de la vidéo en temps réel
    success, frame = video_capture.read()  # Lecture d'une image de la caméra
    if not success:
        break  # Sortie de la boucle si la capture échoue
    
    frame = imutils.resize(frame, width=300)  # Redimensionnement de l'image à une largeur de 300 pixels
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion de l'image en niveaux de gris
    detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)  # Détection des visages dans l'image

    probability_canvas = np.zeros((250, 300, 3), dtype="uint8")  # Création d'un canvas pour afficher les probabilités des émotions
    clone_frame = frame.copy()  # Copie de l'image pour afficher les résultats

    if len(detected_faces) > 0:  # Si des visages sont détectés
        largest_face = sorted(detected_faces, reverse=True, key=lambda x: (x[2] * x[3]))[0]  # Sélection du plus grand visage détecté
        (x, y, w, h) = largest_face
        
        face_roi = gray_frame[y:y + h, x:x + w]  # Extraction de la région d'intérêt (ROI) du visage en niveaux de gris
        face_roi = cv2.resize(face_roi, (64, 64))  # Redimensionnement de la ROI à 64x64 pixels
        face_roi = face_roi.astype("float") / 255.0  # Normalisation des valeurs de pixels
        face_roi = img_to_array(face_roi)  # Conversion de la ROI en tableau d'images
        face_roi = np.expand_dims(face_roi, axis=0)  # Ajout d'une dimension pour la compatibilité avec le modèle
        
        predictions = emotion_model.predict(face_roi)[0]  # Prédiction des émotions à partir de la ROI
        max_emotion_probability = np.max(predictions)  # Probabilité maximale parmi les prédictions
        emotion_label = EMOTIONS[predictions.argmax()]  # Étiquette de l'émotion avec la probabilité maximale
        
        cv2.putText(clone_frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  # Affichage de l'émotion prédite sur l'image originale
        cv2.rectangle(clone_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Dessin d'un rectangle autour du visage détecté
        
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predictions)):  # Pour chaque émotion et sa probabilité
            emotion_text = "{}: {:.2f}%".format(emotion, prob * 100)  # Texte affichant l'émotion et sa probabilité
            prob_width = int(prob * 300)  # Largeur du rectangle représentant la probabilité
            cv2.rectangle(probability_canvas, (7, (i * 35) + 5), (prob_width, (i * 35) + 35), (0, 0, 255), -1)  # Dessin du rectangle
            cv2.putText(probability_canvas, emotion_text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)  # Affichage du texte sur le canvas
    else:
        continue  # Continuer la boucle si aucun visage n'est détecté

    cv2.imshow('your_face', clone_frame)  # Affichage de l'image avec les résultats
    cv2.imshow("Probabilities", probability_canvas)  # Affichage du canvas avec les probabilités
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Arrêter la boucle si la touche 'q' est pressée
        break

video_capture.release()  # Libération de la capture vidéo
cv2.destroyAllWindows()  # Fermeture de toutes les fenêtres

