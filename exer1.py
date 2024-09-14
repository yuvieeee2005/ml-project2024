import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
max_examples = 10000
train_images = train_images[:max_examples]
train_labels = train_labels[:max_examples]
def display(i):
    img = test_images[i]
    plt.title(f'Label: {test_labels[i]}')
    plt.imshow(img, cmap='gray')
    plt.show()
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  
    layers.Dense(128, activation='relu'), 
    layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=100)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(test_images)

predicted_label = np.argmax(predictions[0])
print(f"Prediction: {predicted_label}, Label: {test_labels[0]}")

if predicted_label == test_labels[0]:
    display(0)
