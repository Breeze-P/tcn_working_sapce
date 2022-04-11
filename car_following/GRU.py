import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIMENSION = 8
OUTPUT_DIMENSION = 2
TIME_STEP = 80
BATCH_SIZE = 32
HIDDEN_LAYER_NUMBER = 1
NEURONS_NUMBER = 64
DROPOUT_RATE = 0.2
TRAINING_EPOCHS = 15
RETRAINING_EPOCHS = 5
LR = 0.001

dataX = pd.read_csv('../dataX.csv', header=0)
dataY = pd.read_csv('../dataY.csv', header=0)
np_dataX = dataX.to_numpy()
np_dataY = dataY.to_numpy()
np_dataX = np_dataX.reshape([-1, 80, INPUT_DIMENSION])
# np_dataY = np_dataY[:, None, :]
data_size = np_dataY.shape[0]
print(f"data size:{data_size}")
permutation = list(np.random.permutation(data_size))
np_dataX = np_dataX[permutation]
np_dataY = np_dataY[permutation]

data_train_x = np_dataX[0: int(data_size * 0.8)]
data_train_y = np_dataY[0: int(data_size * 0.8)]
data_test_x = np_dataX[int(data_size * 0.8) + 1: data_size]
data_test_y = np_dataY[int(data_size * 0.8) + 1: data_size]

gru_model = tf.keras.models.Sequential()
gru_model.add(tf.keras.layers.GRU(NEURONS_NUMBER, input_shape=(TIME_STEP, INPUT_DIMENSION),
                                  return_sequences=False, activation='tanh', dropout=DROPOUT_RATE))
gru_model.add(tf.keras.layers.Dropout(0.2))
gru_model.add(tf.keras.layers.Dense(OUTPUT_DIMENSION))
adam = tf.keras.optimizers.Adam(learning_rate=LR)
gru_model.summary()
gru_model.compile(optimizer=adam, loss='mse')
history = gru_model.fit(data_train_x, data_train_y, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS)
epochs = len(history.history['loss'])
ax = plt.subplot(2, 1, 1)
ax.plot(range(epochs), history.history['loss'], label='loss')
ax.legend()
test_data_x = data_test_x[0: 200]
test_data_y = data_test_y[0: 200, 1]
predict_data = gru_model.predict(test_data_x)
bx = plt.subplot(2, 1, 2)
ax.set_title("GRU")
bx.plot(range(200), test_data_y, label='real')
bx.plot(range(200), predict_data[:, 1], label='predict')
bx.legend()
plt.show()
gru_model.save('gru.hdf5')
