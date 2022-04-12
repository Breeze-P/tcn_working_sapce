import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from comparison.TCNstructure.model import TCN

sys.path.append("/")

parser = argparse.ArgumentParser(description='Sequence Modeling - Simulation of zhu2020')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='sequence length (default: 400)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

# torch.manual_seed(args.seed)  # 生成随机数的种子
if torch.cuda.is_available():
    print(1)
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 8  # 输入维度
n_classes = 2  # 输出维度
batch_size = args.batch_size  # 获取batch的大小
seq_length = args.seq_len  # 获取序列长度
epochs = args.epochs  # 获取epochs

print(args)
# print("Producing data...")
# X_train, Y_train = data_generator(50000, seq_length)  # 生成随机数据
# X_test, Y_test = data_generator(1000, seq_length)  # 生成随机数据
x_csv_path = 'dataX_us-101.csv'
y_csv_path = 'dataY_us-101.csv'
ori_x = pd.read_csv(x_csv_path)
ori_y = pd.read_csv(y_csv_path)
cut_num = int(len(ori_x) / 80)  # == len(ori_y)

# 数据切分 train为训练数据 test为每轮epoch测试数据 sim为训练结果模拟数据,比例为0.64 : 0.16 : 0.2
X_train = ori_x[0: int(cut_num * 0.8) * 80]
size_num = int(len(X_train) / 80)
X_test = X_train[int(size_num * 0.8) * 80: size_num * 80]
X_train = X_train[0: int(size_num * 0.8) * 80]
X_sim = ori_x[int(cut_num * 0.8) * 80: cut_num * 80]
Y_train = ori_y[0: int(cut_num * 0.8)]
Y_test = Y_train[int(size_num * 0.8): size_num]
Y_train = Y_train[0: int(size_num * 0.8)]
Y_sim = ori_y[int(cut_num * 0.8): cut_num]
# 维度转换
X_train = torch.from_numpy(X_train.values)
X_train = X_train.view([-1, 80, input_channels])
X_train = torch.transpose(X_train, 1, 2)
X_test = torch.from_numpy(X_test.values)
X_test = X_test.view([-1, 80, input_channels])
X_test = torch.transpose(X_test, 1, 2)
X_sim = torch.from_numpy(X_sim.values)
X_sim = X_sim.view([-1, 80, input_channels])
X_sim = torch.transpose(X_sim, 1, 2)
Y_train = torch.from_numpy(Y_train.values)
Y_train = Y_train.view([-1, 1, n_classes])
Y_train = torch.transpose(Y_train, 1, 2)
Y_test = torch.from_numpy(Y_test.values)
Y_test = Y_test.view([-1, 1, n_classes])
Y_test = torch.transpose(Y_test, 1, 2)
Y_sim = torch.from_numpy(Y_sim.values)
Y_sim = Y_sim.view([-1, 1, n_classes])
Y_sim = torch.transpose(Y_sim, 1, 2)
# 转换为FloatTensor类型
X_train = X_train.type(torch.FloatTensor)
X_test = X_test.type(torch.FloatTensor)
X_sim = X_sim.type(torch.FloatTensor)
Y_train = Y_train.type(torch.FloatTensor)
Y_test = Y_test.type(torch.FloatTensor)
Y_sim = Y_sim.type(torch.FloatTensor)
# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid] * args.levels  # 计算channel size
kernel_size = args.ksize  # 核心数
dropout = args.dropout  # 丢弃率
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)  # 构建模型
model.load_state_dict(torch.load('model_tcn_4_4_32_10_dic_new.pt'))


if args.cuda:  # 数据更新为适配显卡类型
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)  # 优化器


def train(epoch):  # 定义训练函数
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):  # 按batch训练
        if i + batch_size > X_train.size(0):  # 剩余数据大于batch size
            x, y = X_train[i:], Y_train[i:]  # 数据切片
        else:  # 剩余数据小于batch size
            x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
        optimizer.zero_grad()
        output = model(x)
        # output = output.reshape(output.shape[0], 1, output.shape[1])
        loss = F.mse_loss(output, y)  # mse损失函数
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()  # 计算loss

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i + batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100. * processed / X_train.size(0), lr, cur_loss))
            total_loss = 0


def evaluate(x, y):  # 对模型进行评估
    model.eval()
    with torch.no_grad():
        output = model(x)
        test_loss = F.mse_loss(output, y)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item(), output


def draw_predict(t, y, y_, x_label='time', y_label='Index Y', title=''):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, y, 'bo')
    ax.plot(t, y_, 'r+')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.show()



# for ep in range(1, epochs + 1):  # 按epochs大小进行训练
#     train(ep)
#     tloss, predictY = evaluate(X_test, Y_test)


# drawVal = 200
# predictY = predictY[0: drawVal, 0, :]
# realY = Y_sim[0: drawVal, 0, :]
# predictY.reshape(len(predictY))
# realY.reshape(len(realY))
# predictY = predictY.squeeze()
# realY = realY.squeeze()
# predictY = predictY.numpy()
# realY = realY.numpy()
tloss, predictY = evaluate(X_sim, Y_sim)
# epochs = len(tloss)
# ax = plt.subplot(2, 1, 1)
# ax.plot(range(epochs), tloss, label='loss')
# ax.legend()
# ax.set_title("TCN us-101")
X_sim = torch.transpose(X_sim, 1, 2)
Y_sim = torch.transpose(Y_sim, 1, 2)
predict_data = torch.transpose(predictY, 1, 2)
real_data_x_axis = Y_sim[0: 200, :, 0]
predict_data_x_axis = predict_data[0: 200, :, 0]
real_data_y_axis = Y_sim[0: 200, :, 1]
predict_data_y_axis = predict_data[0: 200, :, 1]
ax = plt.subplot(2, 1, 1)
ax.plot(range(200), real_data_x_axis, label='real')
ax.plot(range(200), predict_data_x_axis, label='predict')
ax.set_title("TCN us-101 x-axis")
ax.legend()
bx = plt.subplot(2, 1, 2)
bx.plot(range(200), real_data_y_axis, label='real')
bx.plot(range(200), real_data_y_axis, label='predict')
bx.set_title("TCN us-101 y-axis")
bx.legend()
plt.tight_layout()
plt.show()
# predict()
# with open("model.pt", 'wb') as f:  #将模型进行保存，暂时不用
#     print('Save model!\n')
#     torch.save(model, f)
