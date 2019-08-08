import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(train_image,train_labels),(test_image,test_labels) = fashion_mnist.load_data()
print(train_labels[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_image.shape)
print(len(train_image))


print(train_labels)

print(test_image.shape)

plt.figure()
plt.imshow(train_image[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_image = train_image / 255.0
test_image = test_image /255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
        ])
model.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics = ["accuracy"])
model.fit(train_image,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_image,test_labels)
print("Test Accuracy: {test_acc}")
print("Test Accuracy",test_acc)
