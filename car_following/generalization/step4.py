import pandas as pd

data = pd.read_csv("HighD_1_sorted.csv", header=0)
name = ['Frame_ID', 'Vehicle_ID', 'cur_x', 'cur_y', 'lead_x',
        'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'lane_ID']
content = []
max_val = data.shape[0]

for one in range(0, max_val - 1):
    lists = data.loc[one].values[:]
    two = one + 1
    while two < max_val:
        next_lists = data.loc[two].values[:]
        if next_lists[0] != lists[0]:  # 筛选同一时间
            break
        if lists[7] == next_lists[7] and 0 < abs(lists[2] - next_lists[2]) < 100:  # 同一车道，距离小于100
            temp_lists = [lists[0], lists[1], lists[2], lists[3], next_lists[2], next_lists[3],
                          lists[4], lists[5], next_lists[4], next_lists[5], lists[7]]
            content.append(temp_lists)
            break
        two += 1

output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_merge.csv', index=False)
