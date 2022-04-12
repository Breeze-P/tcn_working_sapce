# 将筛选好的数据进行排序

import pandas as pd
import sys

data = pd.read_csv("test_select.csv", header=0)

name = ['Vehicle_ID', 'Frame_ID', 'cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'Lane_ID']

content1 = []
for one in data.index:
    lists = data.loc[one].values[:]
    # tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[6], lists[7], lists[8], ]
    content1.append(lists)
content = sorted(content1, key=lambda x: (x[0], x[1]))
# data.sort_values(by=['Global_Time', 'Local_Y'], axis=0)

output = pd.DataFrame(columns=name, data=content)
output.to_csv('sorted_i-80_new.csv', index=False)