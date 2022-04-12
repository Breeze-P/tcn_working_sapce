import pandas as pd

data = pd.read_csv("finalResult_new.csv", header=0)

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
outputX.to_csv('dataX.csv', index=False)
outputY.to_csv('dataY.csv', index=False)
