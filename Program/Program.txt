Program

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data to fit CNN input (batch, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes (digits 0–9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predict some digits and visualize
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {np.argmax(predictions[i])}, True: {y_test[i]}")
    plt.axis('off')
    plt.show()



Output

Epoch 1/5
1875/1875 [==============================] - 15s 7ms/step - loss: 0.1478 - accuracy: 0.9552 - val_loss: 0.0514 - val_accuracy: 0.9832
Epoch 2/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0477 - accuracy: 0.9850 - val_loss: 0.0341 - val_accuracy: 0.9886
Epoch 3/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0329 - accuracy: 0.9896 - val_loss: 0.0300 - val_accuracy: 0.9899
Epoch 4/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0246 - accuracy: 0.9920 - val_loss: 0.0319 - val_accuracy: 0.9901
Epoch 5/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0197 - accuracy: 0.9935 - val_loss: 0.0306 - val_accuracy: 0.9910
