# 第四次处理数据，将数据按数据集要求进行计算并得出结果
# 预计用时 21m26s
import pandas as pd

data = pd.read_csv("NGSIM_suitableData_i-80.csv", header=0)

name = ['Vehicle_ID', 'Total_Frames', 'data_Num', 'v_Vel', 'v_Acc', 'v_Pre']
title = ['spacing', 'following_vehicle_speed', 'relative_speed', 'following_vehicle_acceleration']
content = []
pastList = []
count = 0

for one in data.index:
    lists = data.loc[one].values[1:10]
    if one == 0:
        pastList = lists
        count = 1
        continue
    if lists[0] == pastList[0] and lists[2] == pastList[2]:
        pre_speed = (lists[3] + lists[8] - pastList[3] - pastList[8]) * 10
        relative_speed = pre_speed - lists[4]
        tempList = [lists[8], lists[4],  relative_speed, lists[5]]
        # tempList = [lists[0], lists[2], count, lists[4], lists[5], pre_speed]
        content.append(tempList)
        pastList = lists
        count += 1
    else:
        pastList = lists
        count = 1

output = pd.DataFrame(columns=title, data=content)
output.to_csv('NGSIM_finalResult_i-80.csv', index=False)
