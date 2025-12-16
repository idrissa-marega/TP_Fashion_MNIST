import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
print("Chargement des données Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisation (0-255 -> 0-1) pour accélérer l'apprentissage
x_train = x_train / 255.0
x_test = x_test / 255.0

# Noms des classes pour les graphiques
class_names = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

# ==========================================
# 2. FONCTION D'EXPERIMENTATION
# ==========================================
def entrainer_modele(nom, hidden_layers, lr=0.001, epochs=15):
    print(f"\n--- Entraînement : {nom} ---")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Entrée

    # Ajout dynamique des couches cachées
    for neurons in hidden_layers:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(10, activation='softmax')) # Sortie

    # Compilation avec Learning Rate personnalisé
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=(x_test, y_test), verbose=1)
    return history, model

# ==========================================
# 3. Lancement des 3 Expériences Clés
# ==========================================

# A. BASELINE (Consigne : "Perceptron 50 neurones")
hist_base, model_base = entrainer_modele("Baseline (50 neurones)", [50])

# B. AMÉLIORATION ARCHITECTURE (Plus profond : 128 + 64 neurones)
hist_deep, model_deep = entrainer_modele("Deep MLP (128+64)", [128, 64])

# C. TEST LEARNING RATE (Pour l'analyse : LR très faible)
hist_low_lr, _ = entrainer_modele("Low LR (0.0001)", [128, 64], lr=0.0001)

# ==========================================
# 4. GÉNÉRATION DES GRAPHIQUES POUR LE PPT
# ==========================================
plt.figure(figsize=(15, 6))

# GRAPHIQUE 1 : Comparaison des précisions
plt.subplot(1, 2, 1)
plt.plot(hist_base.history['val_accuracy'], label='Baseline (50n)', linestyle='--')
plt.plot(hist_deep.history['val_accuracy'], label='Deep (128n + 64n)', linewidth=2)
plt.plot(hist_low_lr.history['val_accuracy'], label='LR Faible (0.0001)')
plt.title('Comparaison des Performances (Précision)')
plt.xlabel('Époques')
plt.ylabel('Précision (Validation Accuracy)')
plt.legend()
plt.grid(True)

# GRAPHIQUE 2 : Analyse du Sur-apprentissage (Overfitting)
plt.subplot(1, 2, 2)
plt.plot(hist_deep.history['loss'], label='Perte Entraînement (Train)')
plt.plot(hist_deep.history['val_loss'], label='Perte Validation (Test)')
plt.title('Analyse du Sur-apprentissage (Modèle Deep)')
plt.xlabel('Époques')
plt.ylabel('Perte (Loss)')
plt.legend()
plt.grid(True)

plt.show()

# GRAPHIQUE 3 : Matrice de Confusion (Où le modèle se trompe-t-il ?)
y_pred = model_deep.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion (Meilleur Modèle)')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.show()