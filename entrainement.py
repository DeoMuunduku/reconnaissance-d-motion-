"""
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # Importation des callbacks nécessaires
from keras.preprocessing.image import ImageDataGenerator  # Importation du générateur d'images
from load_and_process import load_fer2013_data, preprocess_images  # Importation des fonctions de chargement et prétraitement des données
from models.cnn import mini_XCEPTION  # Importation du modèle mini_XCEPTION
from sklearn.model_selection import train_test_split  # Importation de la fonction de séparation des données en ensembles d'entraînement et de test

# Paramètres
batch_size = 32  # Taille du batch
num_epochs = 10000  # Nombre d'époques
input_shape = (48, 48, 1)  # Forme de l'entrée
validation_split = 0.2  # Fraction de validation
verbose = 1  # Niveau de verbosité
num_classes = 7  # Nombre de classes d'émotion
patience = 50  # Patience pour l'arrêt anticipé
base_path = 'models/'  # Chemin de base pour enregistrer les modèles

# Générateur de données
data_gen = ImageDataGenerator(
    featurewise_center=False,  # Centrage des caractéristiques
    featurewise_std_normalization=False,  # Normalisation des caractéristiques
    rotation_range=10,  # Plage de rotation
    width_shift_range=0.1,  # Plage de décalage horizontal
    height_shift_range=0.1,  # Plage de décalage vertical
    zoom_range=0.1,  # Plage de zoom
    horizontal_flip=True  # Flip horizontal
)

# Paramètres et compilation du modèle
model = mini_XCEPTION(input_shape, num_classes)  # Initialisation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compilation du modèle
model.summary()  # Affichage du résumé du modèle

# Définition des callbacks
log_file_path = base_path + 'emotion_training.log'  # Chemin du fichier de log
csv_logger = CSVLogger(log_file_path, append=False)  # Logger CSV
early_stop = EarlyStopping(monitor='val_loss', patience=patience)  # Arrêt anticipé
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience // 4, verbose=1)  # Réduction du taux d'apprentissage
trained_models_path = base_path + 'mini_XCEPTION'  # Chemin des modèles entraînés
model_checkpoint = ModelCheckpoint(trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True)  # Checkpoint du modèle
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]  # Liste des callbacks

# Chargement du dataset
faces, emotions = load_fer2013_data()  # Chargement des données
faces = preprocess_images(faces)  # Prétraitement des images
num_samples, num_classes = emotions.shape  # Nombre d'échantillons et de classes
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, shuffle=True)  # Séparation des données

# Entraînement du modèle
model.fit(
    data_gen.flow(x_train, y_train, batch_size=batch_size),  # Générateur de données pour l'entraînement
    steps_per_epoch=len(x_train) // batch_size,  # Nombre d'étapes par époque
    epochs=num_epochs,  # Nombre d'époques
    verbose=verbose,  # Niveau de verbosité
    callbacks=callbacks,  # Callbacks
    validation_data=(x_test, y_test)  # Données de validation
)

