from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import imdb
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

def train_and_test_mnist():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28*28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28*28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)

    print('test_acc:', test_acc)

def to_onehot_vector(seqs):
    ret = np.zeros((len(seqs), 10000))
    for i, seq in enumerate(seqs):
        ret[i][seq] = 1.0
    return ret

(train_sentences, train_labels), (test_sentences, test_labels) = imdb.load_data(num_words=10000)
train_x = to_onehot_vector(train_sentences)
test_x = to_onehot_vector(test_sentences)

train_y = train_labels.astype('float32')
test_y = test_labels.astype('float32')

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 4
history = model.fit(train_x,
          train_y,
          batch_size=512,
          epochs=epochs,
          validation_data=(test_x, test_y))

train_loss = history.history['loss']
train_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

result = model.evaluate(test_x, test_y)
print(result)
# plt.title('Training loss and accuracy')
# plt.plot(range(1, epochs + 1), train_loss, 'bo', label='Training Loss')
# plt.plot(range(1, epochs + 1), train_acc, 'b', label='Training accuracy')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# plt.clf()
# plt.title('Test loss and accuracy')
# plt.plot(range(1, epochs + 1), val_loss, 'bo', label='Test Loss')
# plt.plot(range(1, epochs + 1), val_acc, 'b', label='Test accuracy')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()