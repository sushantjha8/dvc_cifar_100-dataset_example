import yaml
import tensorflow as tf
import numpy as np

import pickle
import ustils

params=yaml.safe_load(open("params.yaml"))['train']


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='softmax'),
])


callback = tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3)

model.compile(optimizer="adam",
            loss='categorical_crossentropy',callcacks=[callback] ,metrics=['acc'])



y_train = tf.one_hot(y_train.astype(np.int32), depth=100)
y_test = tf.one_hot(y_test.astype(np.int32), depth=100)

history = model.fit(x_train, y_train, batch_size=32,epochs=100)

ustils.write_model(model)