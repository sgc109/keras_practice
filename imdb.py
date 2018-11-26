import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import optimizers
from keras import models
from keras.datasets import imdb


def to_onehot_vector(seqs):
    ret = np.zeros((len(seqs), 10000))
    for i, seq in enumerate(seqs):
        ret[i][seq] = 1.0
    return ret


def internet_movie_binary_classifier_example():
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

    plt.clf()
    plt.title('Test loss and accuracy')
    plt.plot(range(1, epochs + 1), val_loss, 'bo', label='Test Loss')
    plt.plot(range(1, epochs + 1), val_acc, 'b', label='Test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()