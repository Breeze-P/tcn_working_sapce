# 训练
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("./")
from model import TCN

INPUT_DIMENSION = 8
OUTPUT_DIMENSION = 2

# parser = argparse.ArgumentParser(description='Sequence Modeling - Simulation of zhu2020')
# parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                     help='batch size (default: 32)')
batch_size = 32
# parser.add_argument('--cuda', action='store_false',
#                     help='use CUDA (default: True)')
cuda = False
# parser.add_argument('--dropout', type=float, default=0.2,
#                     help='dropout applied to layers (default: 0.0)')
dropout = 0.2
# parser.add_argument('--clip', type=float, default=-1,
#                     help='gradient clip, -1 means no clip (default: -1)')
clip = 1
# parser.add_argument('--epochs', type=int, default=10,
#                     help='upper epoch limit (default: 10)')
epochs = 10
# parser.add_argument('--ksize', type=int, default=4,
#                     help='kernel size (default: 7)')
ksize = 4
# parser.add_argument('--levels', type=int, default=4,
#                     help='# of levels (default: 8)')
levels = 4
# parser.add_argument('--seq_len', type=int, default=80,
#                     help='sequence length (default: 400)')
seq_len = 80
# parser.add_argument('--log-interval', type=int, default=5, metavar='N',
#                     help='report interval (default: 100')
log_interval = 5
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='initial learning rate (default: 4e-3)')
lr = 0.001
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer to use (default: Adam)')
optimI = 'Adam'
# parser.add_argument('--nhid', type=int, default=32,
#                     help='number of hidden units per layer (default: 30)')
nhid = 32
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed (default: 1111)')
seed = 1111
# args = parser.parse_args()

dataX = pd.read_csv('data/dataX.csv', header=0)
dataY = pd.read_csv('data/dataY.csv', header=0)
np_dataX = dataX.to_numpy()
np_dataY = dataY.to_numpy()
np_dataX = np_dataX.reshape([-1, 80, INPUT_DIMENSION])
np_dataY = np_dataY.reshape([-1, 1, OUTPUT_DIMENSION])
data_size = np_dataY.shape[0]
end_line = int(data_size * 0.8)
print(f"data size:{data_size}")
permutation = list(np.random.permutation(data_size))
np_dataX = np_dataX[permutation]
np_dataY = np_dataY[permutation]

X_train = np_dataX[0: int(data_size * 0.8)]
Y_train = np_dataY[0: int(data_size * 0.8)]
X_test = np_dataX[int(data_size * 0.8) + 1: data_size]
Y_test = np_dataY[int(data_size * 0.8) + 1: data_size]

# 打乱
np.random.seed(1)
np.random.shuffle(X_train)
X_train = torch.from_numpy(X_train)
np.random.seed(1)
np.random.shuffle(Y_train)
Y_train = torch.from_numpy(Y_train)
np.random.seed(1)
np.random.shuffle(X_test)
X_test = torch.from_numpy(X_test)
np.random.seed(1)
np.random.shuffle(Y_test)
Y_test = torch.from_numpy(Y_test)

# 交换维度
X_train = torch.transpose(X_train, 1, 2)
Y_train = torch.transpose(Y_train, 1, 2)
X_test = torch.transpose(X_test, 1, 2)
Y_test = torch.transpose(Y_test, 1, 2)

# 转float32
X_train = X_train.to(torch.float32)
Y_train = Y_train.to(torch.float32)
X_test = X_test.to(torch.float32)
Y_test = Y_test.to(torch.float32)

input_channels = INPUT_DIMENSION  # 输入维度
n_classes = 2  # 输出维度
batch_size = batch_size  # 获取batch的大小
seq_length = seq_len  # 获取序列长度
epochs = epochs  # 获取epochs

# print(args)

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [nhid] * levels  # 计算channel size
kernel_size = ksize  # 核心数
dropout = dropout  # 丢弃率
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)  # 构建模型

# if args.cuda:  # 数据更新为适配显卡类型
#     model.cuda()
#     X_train = X_train.cuda()
#     Y_train = Y_train.cuda()
#     X_test = X_test.cuda()
#     Y_test = Y_test.cuda()

# lr = args.lr
optimizer = getattr(optim, optimI)(model.parameters(), lr=lr)  # 优化器


def train(epoch):  # 定义训练函数
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, end_line, batch_size):  # 按batch训练
        if i + batch_size > end_line:  # 剩余数据大于batch size
            x, y = X_train[i:], Y_train[i:]  # 数据切片
        else:  # 剩余数据小于batch size
            x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
        optimizer.zero_grad()
        output = model(x)
        # output = output.reshape(output.shape[0], 1, output.shape[1])
        loss = F.mse_loss(output, y)  # mse损失函数
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()  # 计算loss

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i + batch_size, end_line)
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, end_line, 100. * processed / end_line, lr, cur_loss))
            total_loss = 0


def evaluate(x, y):  # 对模型进行评估
    model.eval()
    with torch.no_grad():
        output = model(x)
        test_loss = F.mse_loss(output, y)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item(), output


# def predict():
#     pre_val = model(X_test)
#     # 获取预测值
#     pre_val = pre_val.data.cpu().numpy()
#     # 画个图看看，到底拟合成啥样了？
#     x0 = X_test[:, :1].cpu().numpy()
#     x1 = x0[..., :1].reshape(1000, 1)
#     y0 = Y_test[:].cpu().numpy()
#     plt.plot(x1, y0, 'ro', label='original data')
#     plt.plot(sorted(x1), sorted(pre_val), label='training data', color='blue')
#     plt.show()

loss_storage = []

for ep in range(1, epochs + 1):  # 按epochs大小进行训练
    train(ep)
    tloss, predictY = evaluate(X_test, Y_test)
    loss_storage.append(tloss)

# predictY = predictY[0: drawVal, 0, :]
# realY = Y_sim[0: drawVal, 0, :]
# predictY.reshape(len(predictY))
# realY.reshape(len(realY))
# predictY = predictY.squeeze()
# realY = realY.squeeze()
# predictY = predictY.numpy()
# realY = realY.numpy()
# draw_predict(range(0, drawVal, 1), realY, predictY)
# predict()
# with open("model.pt", 'wb') as f:  #将模型进行保存，暂时不用
#     print('Save model!\n')
#     torch.save(model, f)

from TCN.car_following.utils import draw_predict
# 画收敛图
# TODO
draw_predict(range(0, epochs, 1), loss_storage, x_label='epoch', y_label='mse_loss', title='tcn_each_loss_4_4_32_10_new')
# 测试集Loss
tloss, predictY = evaluate(X_test, Y_test)
drawVal = 200
drawY = Y_test.numpy()[:200, 1, 0]
drawY_ =  predictY[:200,1,0].detach().numpy()
# print('Y',drawY)
# print('Y_',drawY_)
# TODO
draw_predict(range(0, 200, 1), drawY, drawY_, title='tcn_predict_200_4_4_32_10_new')

# TODO
with open('models/model_tcn_4_4_32_10_dic_new.pt', 'wb') as f:
  torch.save(model.state_dict(), f)
f.close()