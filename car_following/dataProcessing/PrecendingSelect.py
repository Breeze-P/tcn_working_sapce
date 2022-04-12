# 第三次处理数据，找寻前车将前后车数据汇总为同一列

import pandas as pd
import sys

data = pd.read_csv("NGSIM_ascendingSort_us-101_new.csv", header=0)
# data = pd.read_csv("123.csv", header=0)
name = ['Vehicle_ID', 'Frame_ID', 'cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'Lane_ID']
content = []
max_val = data.shape[0]

for one in range(0, max_val - 1):
    lists = data.loc[one].values[:]
    two = one + 1
    while two < max_val:
        next_lists = data.loc[two].values[:]
        if next_lists[1] != lists[1]:  # 筛选同一时间
            break
        if lists[8] == next_lists[8] and 0 < lists[10] < 100:  # 同一车道，距离小于100
            temp_lists = [lists[0], lists[1], lists[4], lists[5], next_lists[4], next_lists[5],
                          lists[6], lists[7], next_lists[6], next_lists[7], lists[8]]
            content.append(temp_lists)
            break
        two += 1

output = pd.DataFrame(columns=name, data=content)
output.to_csv('selected_us_101.csv', index=False)


