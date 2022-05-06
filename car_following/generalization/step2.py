import pandas as pd

data = pd.read_csv("HighD_1.csv", header=0)
name = ['Frame_ID', 'Vehicle_ID', 'x', 'y', 'v', 'a', 'preceding_ID', 'lane_ID']
content = []
max_val = data.shape[0]
index = 0

while index < max_val:
    cur_list = data.loc[index].values[:]
    right = index + 1
    if right >= max_val:
        break
    next_list = data.loc[right].values[:]
    while next_list[1] == cur_list[1]:
        right += 1
        if right >= max_val:
            break
        next_list = data.loc[right].values[:]
    if right - index > 80:
        period = data[index: right]
        period = period.values.tolist()
        content += period
    index = right


print(len(content))
output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_1.csv', index=False)