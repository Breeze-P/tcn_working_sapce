import pandas as pd
import sys

data = pd.read_csv("data/finalResult_new.csv", header=0)
dataX = []
dataY = []
nameX = ['cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a']
nameY = ['cur_x', 'cur_y']
count = 0
max_val = data.shape[0]

for one in data.index:
    lists = data.loc[one].values[2:10]
    temp = lists[0: 2]
    if (one + 1) % 81 == 0 and one != 0:
        dataY.append(temp)
    elif one % 81 == 0:
        dataX.append(lists)
    else:
        dataX.append(lists)
        dataY.append(temp)

outputX = pd.DataFrame(columns=nameX, data=dataX)
outputY = pd.DataFrame(columns=nameY, data=dataY)
outputX.to_csv('dataX_80_timestep.csv', index=False)
outputY.to_csv('dataY_80_timestep.csv', index=False)

