# 绘制各网络总对比图
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from TCNstructure.model import TCN

INPUT_DIMENSION = 8
OUTPUT_DIMENSION = 2
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



# 给四个网络加载训练模型
rnn_model = tf.keras.models.load_model('rnn.hdf5')
gru_model = tf.keras.models.load_model('gru.hdf5')
lstm_model = tf.keras.models.load_model('LSTM.h5')
tcn_model = TCN(INPUT_DIMENSION, OUTPUT_DIMENSION, [32, 32, 32, 32], kernel_size=4, dropout=0.2)  # 构建模型
tcn_model.load_state_dict(torch.load('model_tcn_4_4_32_10_dic_new.pt'))

# 切分并处理数据，获取网络中的预测数据，range_val为图中x轴范围，可修改
range_val = data_test_y.shape[0]
test_data_x = data_test_x[0: range_val]
rnn_predict_data = rnn_model.predict(test_data_x)
gru_predict_data = gru_model.predict(test_data_x)
lstm_predict_data = lstm_model.predict(test_data_x)
tcn_data_x = torch.from_numpy(test_data_x)
tcn_data_x = tcn_data_x.type(torch.FloatTensor)
tcn_data_x = torch.transpose(tcn_data_x, 1, 2)  # tcn和别的网络shape不一样，转换维度
tcn_predict_data = tcn_model(tcn_data_x)
tcn_predict_data = tcn_predict_data.squeeze(2)  # 消除最后一个维度
tcn_predict_data = tcn_predict_data.detach().numpy()

# 画图
ax = plt.subplot(1, 1, 1)
ax.plot(range(range_val), data_test_y[0: range_val, 0], label='real')
ax.plot(range(range_val), tcn_predict_data[:, 0], label='TCN')
ax.plot(range(range_val), rnn_predict_data[:, 0], label='RNN')
ax.plot(range(range_val), gru_predict_data[:, 0], label='GRU')
ax.plot(range(range_val), lstm_predict_data[:, 0], label='LSTM')
# ax.plot(data_test_y[0: range_val, 0], data_test_y[0: range_val, 1], label='real')
# ax.plot(tcn_predict_data[:, 0], tcn_predict_data[:, 1], label='TCN')
# ax.plot(rnn_predict_data[:, 0], rnn_predict_data[:, 1], label='RNN')
# ax.plot(gru_predict_data[:, 0], gru_predict_data[:, 1], label='GRU')
# ax.plot(lstm_predict_data[:, 0], lstm_predict_data[:, 1], label='LSTM')
ax.set_title("Generalization of Neural Network Models in us-101")
ax.legend(loc='lower right')  # 设置图例
plt.show()

temp_y = torch.from_numpy(data_test_y)
tcn_loss = F.mse_loss(torch.from_numpy(tcn_predict_data), temp_y)
lstm_loss = F.mse_loss(torch.from_numpy(lstm_predict_data), temp_y)
rnn_loss = F.mse_loss(torch.from_numpy(rnn_predict_data), temp_y)
gru_loss = F.mse_loss(torch.from_numpy(gru_predict_data), temp_y)
print(f"TCN loss:{tcn_loss.item()}")
print(f"LSTM loss:{lstm_loss.item()}")
print(f"RNN loss:{rnn_loss.item()}")
print(f"GRU loss:{gru_loss.item()}")

