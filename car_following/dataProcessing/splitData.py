# 第六次处理，将处理完的数据按网络模型切分为x和y,其中x为(?, timestep, 8),y为(?, 2)
import pandas as pd

data = pd.read_csv("finalResult_us-101_new.csv", header=0)

nameX = ['cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a']
nameY = ['cur_x', 'cur_y']
dataX = []
dataY = []

for one in data.index:
    lists = data.loc[one].values[2:10]
    if (one + 1) % 81 == 0 and one != 0:
        temp = lists[0: 2]
        dataY.append(temp)
    else:
        dataX.append(lists)
outputX = pd.DataFrame(columns=nameX, data=dataX)
outputY = pd.DataFrame(columns=nameY, data=dataY)
outputX.to_csv('dataX_us-101.csv', index=False)
outputY.to_csv('dataY_us-101.csv', index=False)
