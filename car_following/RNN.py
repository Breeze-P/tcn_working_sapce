import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIMENSION = 8
OUTPUT_DIMENSION = 2
TIME_STEP = 80
BATCH_SIZE = 16
HIDDEN_LAYER_NUMBER = 1
NEURONS_NUMBER = 50
DROPOUT_RATE = 0.2
TRAINING_EPOCHS = 15
RETRAINING_EPOCHS = 5
LR = 0.001

dataX = pd.read_csv('../dataX.csv', header=0)
dataY = pd.read_csv('../dataY.csv', header=0)
np_dataX = dataX.to_numpy()
np_dataY = dataY.to_numpy()
np_dataX = np_dataX.reshape([-1, 80, INPUT_DIMENSION])
# np_dataY = np_dataY.reshape([-1, 80, OUTPUT_DIMENSION])
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

rnn_model = tf.keras.models.Sequential()
rnn_model.add(tf.keras.layers.SimpleRNN(64, input_shape=(TIME_STEP, INPUT_DIMENSION),
                                        return_sequences=False, activation='tanh'))
# rnn_model.add(tf.keras.layers.SimpleRNN(NEURONS_NUMBER, return_sequences=True))
# rnn_model.add(tf.keras.layers.SimpleRNN(NEURONS_NUMBER))
rnn_model.add(tf.keras.layers.Dense(OUTPUT_DIMENSION))
adam = tf.keras.optimizers.Adam(learning_rate=LR)
rnn_model.summary()
rnn_model.compile(optimizer=adam, loss='mse')
history = rnn_model.fit(data_train_x, data_train_y, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS,
                        validation_data=(data_test_x, data_test_y))
# 保存模型，以便使用时加载
epochs = len(history.history['loss'])
ax = plt.subplot(2, 1, 1)
ax.plot(range(epochs), history.history['loss'], label='loss')
ax.plot(range(epochs), history.history['val_loss'], label='val_loss')
ax.legend()
test_data_x = data_test_x[0: 200]
test_data_y = data_test_y[0: 200, 1]
predict_data = rnn_model.predict(test_data_x)
bx = plt.subplot(2, 1, 2)
ax.set_title("RNN")
bx.plot(range(200), test_data_y, label='real')
bx.plot(range(200), predict_data[:, 1], label='predict')
bx.legend()
plt.show()
rnn_model.save('rnn.hdf5')
