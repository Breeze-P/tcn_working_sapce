import pandas as pd

data = pd.read_csv("HighD_1_merge.csv", header=0)
name = ['Frame_ID', 'Vehicle_ID', 'cur_x', 'cur_y', 'lead_x',
        'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'lane_ID']
content1 = []

for one in data.index:
    lists = data.loc[one].values[:]
    # tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[6], lists[7], lists[8], ]
    content1.append(lists)
content = sorted(content1, key=lambda x: (x[1], x[0]))

output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_merge_sorted.csv', index=False)
