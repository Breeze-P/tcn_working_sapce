# 第一次处理数据，将重复数据和class不是2（非轿车的数据）排除

import pandas as pd
import sys

data = pd.read_csv("NGSIM.csv", header=0)
content = []
name = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc',
        'Lane_ID', 'Preceding', 'Space_Headway']
pastList = []
sames = 0

for one in data.index:
    lists = data.loc[one].values[0:25]
    if one == 0:
        pastList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[11], lists[12], lists[13],
                    lists[20], lists[22]]
        content.append(pastList)
        continue
    if lists[10] == 2 and lists[20] != 0 and lists[22] != 0 and lists[11] != 0 and (lists[24] == 'us-101'):
        if lists[0] != pastList[0]:
            tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[11], lists[12],
                        lists[13], lists[20], lists[22]]
            pastList = tempList
            content.append(tempList)
        else:
            if pastList[1] != lists[1] and pastList[2] != lists[2]:
                tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[11], lists[12],
                            lists[13], lists[20], lists[22]]
                pastList = tempList
                content.append(tempList)
            else:
                sames += 1
    if one % 10000 == 0:
        print(one)
output = pd.DataFrame(columns=name, data=content)
output.to_csv('NGSIM_classFilter_us-101_new.csv', index=False)
