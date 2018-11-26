from keras import models, layers, optimizers
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def to_onehot_vector(seqs):
    ret = np.zeros((len(seqs), 10000))
    for i, seq in enumerate(seqs):
        ret[i][seq] = 1.0
    return ret

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# word_to_id = reuters.get_word_index()
#
# id_to_word = dict([(id, word) for word, id in word_to_id.items()])
#
# print(' '.join([id_to_word.get(wid - 3, '?') for wid in train_data[0]]))
# print(train_data[0])
#
# print(train_data[0])

train_x = to_onehot_vector(train_data)
test_x = to_onehot_vector(test_data)


train_y = to_categorical(train_labels)
test_y = to_categorical(test_labels)

val_x = train_x[:1000]
val_y = train_y[:1000]
train_x = train_x[1000:]
train_y = train_y[1000:]

model = models.Sequential()
model.add(layer=layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layer=layers.Dense(128, activation='relu'))
model.add(layer=layers.Dense(46, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 5
history = model.fit(train_x, train_y, 512, epochs, validation_data=(val_x, val_y))

# plt.title('Training and Validation Accuracy')
# plt.plot(range(1, epochs + 1), history.history['acc'], 'b', label='Training')
# plt.plot(range(1, epochs + 1), history.history['val_acc'], 'bo', label='Validation')
# plt.legend()
# plt.show()

result = model.evaluate(test_x, test_y)
print(result)