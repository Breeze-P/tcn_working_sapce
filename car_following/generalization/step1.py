import pandas as pd

data = pd.read_csv("../../highd-dataset-v1.0/data/01_tracks.csv", header=0)
name = ['Frame_ID', 'Vehicle_ID', 'x', 'y', 'v', 'a', 'preceding_ID', 'lane_ID']
content = []
max_val = data.shape[0]

for one in range(0, max_val):
    lists = data.loc[one].values[:]
    if lists[6] > 0:
        line = [lists[0], lists[1], lists[2], lists[3], lists[6], lists[8], lists[16], lists[24]]
        content.append(line)

print(len(content))
output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1.csv', index=False)