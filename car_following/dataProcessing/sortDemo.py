# 第二次处理数据，将相同车同一时段数据汇总并排序

import pandas as pd
import sys

data = pd.read_csv("NGSIM_classFilter_us-101_new.csv")

name = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc',
        'Lane_ID', 'Preceding', 'Space_Headway']

content1 = []
for one in data.index:
    lists = data.loc[one].values[:]
    # tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[6], lists[7], lists[8], ]
    content1.append(lists)
content = sorted(content1, key=lambda x: (x[3], x[5], x[8]))
# data.sort_values(by=['Global_Time', 'Local_Y'], axis=0)

output = pd.DataFrame(columns=name, data=content)
output.to_csv('NGSIM_ascendingSort_us-101_new.csv', index=False)
