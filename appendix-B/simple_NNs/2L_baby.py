import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 1] + X[:, 2] > 0).astype(int)

def build_model(hidden_units):
    model = keras.Sequential([
        layers.Dense(hidden_units, activation='relu', input_shape=(5,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

wide_model = build_model(20)
narrow_model = build_model(2)

history_wide = wide_model.fit(X, y, epochs=50, validation_split=0.2)
history_narrow = narrow_model.fit(X, y, epochs=50, validation_split=0.2)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_wide.history['loss'], label='Wide model')
plt.plot(history_narrow.history['loss'], label='Narrow model')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_wide.history['accuracy'], label='Wide model')
plt.plot(history_narrow.history['accuracy'], label='Narrow model')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()