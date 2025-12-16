import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
print("Chargement du dataset Fashion-MNIST...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ajout du canal (CNN attend 4D : (batch, hauteur, largeur, canaux))
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

class_names = [
    'T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine'
]

# ==========================================
# 2. MODÈLE CNN - LeNet-5
# ==========================================
print("Construction du modèle CNN (LeNet-5)...")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# ==========================================
# 3. COMPILATION ET ENTRAÎNEMENT
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=15,
    validation_data=(x_test, y_test),
    verbose=1
)

# ==========================================
# 4. ÉVALUATION DU MODÈLE
# ==========================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nPrécision du CNN sur les données de test : {test_acc:.4f}")

# ==========================================
# 5. COURBES D'APPRENTISSAGE
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Précision du CNN")
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Perte du CNN")
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# 6. MATRICE DE CONFUSION
# ==========================================
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx,
    annot=True,
    fmt='g',
    cmap='Greens',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Matrice de confusion - CNN (LeNet-5)")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.show()
