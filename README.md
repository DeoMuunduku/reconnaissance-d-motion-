# reconnaissance-d-motion-
# Projet de Reconnaissance d'Émotions

## Description
Ce projet se concentre sur la reconnaissance des émotions à partir d'images de visages. En utilisant des réseaux de neurones convolutifs (CNN), le système est capable d'identifier et de classifier différentes émotions humaines basées sur des expressions faciales. Les émotions reconnues incluent la colère, le dégoût, la peur, le bonheur, la tristesse, la surprise et la neutralité.

## Table des Matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Détails du Code](#détails-du-code)
- [Contributeurs](#contributeurs)

## Installation
Pour installer les dépendances nécessaires, exécutez :
```bash
pip install -r requirements.txt


Utilisation
Préparation des données
Le dataset utilisé pour ce projet est FER2013, qui contient des images de visages annotées avec des émotions. Pour charger et prétraiter ces données, utilisez le script load_and_process.py.

Entraînement des modèles
Pour entraîner un modèle, utilisez le script train_emotion_classifier.py. Vous pouvez modifier les paramètres d'entraînement directement dans le script.

Prédiction en temps réel
Pour effectuer des prédictions en temps réel à partir de la webcam, utilisez le script real_time_emotion_recognition.py.

Structure du Projet
load_and_process.py : Chargement et prétraitement des données du dataset FER2013.
train_emotion_classifier.py : Entraînement des modèles de classification des émotions.
real_time_emotion_recognition.py : Reconnaissance des émotions en temps réel à partir de la webcam.
models/cnn.py : Définition des différents modèles CNN utilisés.
Détails du Code
Chargement et Prétraitement des Données (load_and_process.py)
Ce script contient des fonctions pour charger le dataset FER2013 et le prétraiter pour le rendre compatible avec les modèles CNN.

Fonction load_fer2013_data : Charge les données du dataset, convertit les séquences de pixels en images et applique des transformations pour préparer les données à l'entraînement.
Fonction preprocess_images : Normalise les images pour améliorer les performances du modèle.
Entraînement des Modèles (train_emotion_classifier.py)
Ce script configure et entraîne les modèles CNN pour la reconnaissance des émotions.

Paramètres : Contient des hyperparamètres tels que la taille du batch, le nombre d'époques, la forme de l'entrée, etc.
Callbacks : Utilise des callbacks pour surveiller et améliorer le processus d'entraînement (e.g., enregistrement des meilleurs modèles, arrêt anticipé, réduction du taux d'apprentissage).
Entraînement : Utilise le générateur de données pour augmenter les données d'entraînement et entraîne le modèle en utilisant les données prétraitées.
Reconnaissance des Émotions en Temps Réel (real_time_emotion_recognition.py)
Ce script capture les images de la webcam, détecte les visages, et prédit les émotions en utilisant un modèle pré-entraîné.

Détection des Visages : Utilise un classificateur Haar pour détecter les visages dans les images capturées.
Classification des Émotions : Utilise un modèle CNN pré-entraîné pour prédire les émotions à partir des images de visages détectées.
Affichage : Affiche les résultats en temps réel avec les probabilités des émotions et les étiquettes correspondantes.
Modèles CNN (models/cnn.py)
Ce fichier contient la définition de plusieurs architectures CNN pour la reconnaissance des émotions.

simple_CNN : Un modèle simple avec plusieurs couches de convolution, de pooling et de dropout.
simpler_CNN : Une version simplifiée avec une architecture plus légère.
tiny_XCEPTION, mini_XCEPTION, big_XCEPTION : Différentes versions du modèle XCEPTION, chacune optimisée pour différents niveaux de complexité et de taille des données.
Contributeurs
Deo - Chercheur en intelligence artificielle, spécialisé en deep learning, vision par ordinateur et explicabilité des modèles prédictifs.
Pour plus d'informations, veuillez consulter les scripts individuels et les commentaires dans le code
