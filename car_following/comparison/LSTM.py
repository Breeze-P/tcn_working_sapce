import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIMENSION = 8
OUTPUT_DIMENSION = 2
TIME_STEP = 80
BATCH_SIZE = 32
HIDDEN_LAYER_NUMBER = 1
NEURONS_NUMBER = 100
DROPOUT_RATE = 0.2
TRAINING_EPOCHS = 15
RETRAINING_EPOCHS = 5
LR = 0.001

dataX = pd.read_csv('../dataX.csv', header=0)
dataY = pd.read_csv('../dataY.csv', header=0)
np_dataX = dataX.to_numpy()
np_dataY = dataY.to_numpy()
np_dataX = np_dataX.reshape([-1, 80, INPUT_DIMENSION])
data_size = np_dataY.shape[0]
print(f"data size:{data_size}")
permutation = list(np.random.permutation(data_size))
np_dataX = np_dataX[permutation]
np_dataY = np_dataY[permutation]

data_train_x = np_dataX[0: int(data_size * 0.8)]
data_train_y = np_dataY[0: int(data_size * 0.8)]
data_test_x = np_dataX[int(data_size * 0.8) + 1: data_size]
data_test_y = np_dataY[int(data_size * 0.8) + 1: data_size]


# 第一个参数传模型，第二个传测试x，第三个传测试y，最后一个传x轴范围
def drawChart(model, x_data, y_data, range_val):
    # 切分数据
    x_data = x_data[0: range_val]
    predict_data = model.predict(x_data)
    # 画x轴，plot那一行最后一个参数为0是x轴，为1是y轴
    graph_up = plt.subplot(2, 1, 1)
    graph_up.plot(range(range_val), y_data[0: range_val, 0], label='real')
    graph_up.plot(range(range_val), predict_data[:, 0], label='predict')
    graph_up.set_title("LSTM us-101 x-axis")  # 标题
    graph_up.legend()  # 图例
    # 画y轴
    graph_down = plt.subplot(2, 1, 2)
    graph_down.plot(range(range_val), y_data[0: range_val, 1], label='real')
    graph_down.plot(range(range_val), predict_data[:, 1], label='predict')
    graph_down.set_title("LSTM us-101 y-axis")
    graph_down.legend()
    plt.tight_layout()
    plt.show()


# 创建模型，如果加载模型只画图则下面代码不用执行，注释
lstm_model = tf.keras.models.Sequential()
lstm_model.add(tf.keras.layers.LSTM(NEURONS_NUMBER, input_shape=(TIME_STEP, INPUT_DIMENSION),
                                    dropout=DROPOUT_RATE, activation='sigmoid', return_sequences=False))
lstm_model.add(tf.keras.layers.Dense(OUTPUT_DIMENSION))
adam = tf.keras.optimizers.Adam(learning_rate=LR)
lstm_model.summary()
lstm_model.compile(optimizer=adam, loss='mse')
history = lstm_model.fit(data_train_x, data_train_y, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS,
                         validation_data=(data_test_x, data_test_y))

# 绘制loss图
epochs = len(history.history['loss'])
ax = plt.subplot(2, 1, 1)
ax.plot(range(epochs), history.history['loss'], label='loss')
ax.plot(range(epochs), history.history['val_loss'], label='val_loss')
ax.legend()
ax.set_title("LSTM")
test_data_x = data_test_x[0: 200]
test_data_y = data_test_y[0: 200, 1]
predict_y = lstm_model.predict(test_data_x)

# 绘制数据拟合图
bx = plt.subplot(2, 1, 2)
bx.plot(range(200), test_data_y, label='real')
bx.plot(range(200), predict_y[:, 1], label='predict')
bx.legend()
plt.show()
lstm_model.save('LSTM.h5')  # 保存模型

# 只画图的话把上面代码注释
lstm_model = tf.keras.models.load_model('LSTM.h5')  # 加载模型
drawChart(lstm_model, data_test_x, data_test_y, 200)
