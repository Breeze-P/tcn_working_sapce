# Data maker
import pandas as pd
import numpy as np
import torch

# print("Producing data...")
# X_train, Y_train = data_generator(50000, seq_length)  # 生成随机数据
# X_test, Y_test = data_generator(1000, seq_length)  # 生成随机数据
all_data_path = 'data/finalResult_new.csv'
x_csv_path = 'data/dataX.csv'
y_csv_path = 'data/dataY.csv'
ori_all = pd.read_csv(all_data_path)
# ori_x = pd.read_csv(x_csv_path)
# ori_y = pd.read_csv(y_csv_path)

fix_all = ori_all.loc[:, ['cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'lead_v', 'cur_a', 'lead_a']]
cut_num = int(len(fix_all) / 81)  # == len(ori_y)

# 数据提取
fix_x = []
fix_y = []
temp_list = []
for i in range(len(fix_all)):
    if i != 0 and i % 81 == 0:
        temp = [fix_all.values[i][2: 4]]
        fix_y.append(temp)
        fix_x.append(temp_list)
        temp_list = []
    else:
        temp = [fix_all.values[i][:].tolist()]
        temp_list.append(fix_all.values[i][:])

# 数据切分 train为训练数据 test为每轮epoch测试数据 sim为训练结果模拟数据,比例为0.64 : 0.16 : 0.2
# X_train = fix_x[0: int(cut_num * 0.8) * 80]
# size_num = int(len(X_train) / 80)
# X_test = X_train[int(size_num * 0.8) * 80: size_num * 80]
# X_train = X_train[0: int(size_num * 0.8) * 80]
# X_sim = fix_x[int(cut_num * 0.8) * 80: int(cut_num * 80)]
# Y_train = fix_y[0: int(cut_num * 0.8)]
# Y_test = Y_train[int(size_num * 0.8): size_num]
# Y_train = Y_train[0: int(size_num * 0.8)]
# Y_sim = fix_y[int(cut_num * 0.8): int(cut_num)]
# 维度转换
X_train = np.array(fix_x)
Y_train = np.array(fix_y)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)
# X_sim = np.array(X_sim)
# Y_sim = np.array(Y_sim)
X_train = torch.from_numpy(X_train.astype(np.float32))
# X_train = X_train.view([-1, 80, 8])
X_train = torch.transpose(X_train, 1, 2)
# X_test = torch.from_numpy(X_test.astype(np.float32))
# X_test = X_test.view([-1, 80, 8])
# X_test = torch.transpose(X_test, 1, 2)
# X_sim = torch.from_numpy(X_sim.astype(np.float32))
# X_sim = X_sim.view([-1, 80, 8])
# X_sim = torch.transpose(X_sim, 1, 2)
Y_train = torch.from_numpy(Y_train.astype(np.float32))
# Y_train = Y_train.view([-1, 1, 2])
Y_train = torch.transpose(Y_train, 1, 2)
# Y_test = torch.from_numpy(Y_test.astype(np.float32))
# Y_test = Y_test.view([-1, 1, 2])
# Y_test = torch.transpose(Y_test, 1, 2)
# Y_sim = torch.from_numpy(Y_sim.astype(np.float32))
# Y_sim = Y_sim.view([-1, 1, 2])
# Y_sim = torch.transpose(Y_sim, 1, 2)
# # 转换为FloatTensor类型
# X_train = X_train.type(torch.FloatTensor)
# X_test = X_test.type(torch.FloatTensor)
# X_sim = X_sim.type(torch.FloatTensor)
# Y_train = Y_train.type(torch.FloatTensor)
# Y_test = Y_test.type(torch.FloatTensor)
# Y_sim = Y_sim.type(torch.FloatTensor)
#
# # 打乱
# X_train = X_train.numpy()
# np.random.seed(1234)
# np.random.shuffle(X_train)
# X_train = torch.from_numpy(X_train)
# Y_train = Y_train.numpy()
# np.random.seed(1234)
# np.random.shuffle(Y_train)
# Y_train = torch.from_numpy(Y_train)
# X_test = X_test.numpy()
# np.random.seed(1234)
# np.random.shuffle(X_test)
# X_test = torch.from_numpy(X_test)
# Y_test = Y_test.numpy()
# np.random.seed(1234)
# np.random.shuffle(Y_test)
# Y_test = torch.from_numpy(Y_test)

# 转float32
# X_train = X_train.to(torch.float32)
# Y_train = Y_train.to(torch.float32)
# X_test = X_test.to(torch.float32)
# Y_test = Y_test.to(torch.float32)
