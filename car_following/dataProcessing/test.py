# 第三次处理数据，将符合15s以上，跟随同一辆车，不变换车道的数据筛选出来
# 预计用时 18m26s

import pandas as pd

data = pd.read_csv("NGSIM_ascendingSort_i-80.csv", header=0)

name = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Space_Headway']
content = []
pastList = []
tempList = []
count = 0

for one in data.index:
    lists = data.loc[one].values[1:10]
    if len(tempList) == 0:
        pastList = lists
        count += 1
        tempList.append(lists)
        continue
    if lists[0] == pastList[0] and lists[2] == pastList[2] \
            and lists[6] == pastList[6] and lists[7] == pastList[7]:
        pastList = lists
        count += 1
        tempList.append(lists)
        continue
    else:
        if count >= 150:
            for things in tempList:
                content.append(things)
        tempList.clear()
        count = 0
        pastList = lists
        count += 1
        tempList.append(lists)

output = pd.DataFrame(columns=name, data=content)
output.to_csv('NGSIM_suitableData_i-80.csv')
