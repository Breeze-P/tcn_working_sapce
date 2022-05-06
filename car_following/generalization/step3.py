import pandas as pd

data = pd.read_csv("HighD_1_1.csv", header=0)
name = ['Frame_ID', 'Vehicle_ID', 'x', 'y', 'v', 'a', 'preceding_ID', 'lane_ID']
content = []
content1 = []
max_val = data.shape[0]

for one in data.index:
    lists = data.loc[one].values[:]
    # tempList = [lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], lists[6], lists[7], lists[8], ]
    content1.append(lists)
content = sorted(content1, key=lambda x: (x[0], x[1]))
output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_sorted.csv', index=False)


# while cur_left < max_val:
#     cur_list = data.loc[cur_left].values[:]
#     if cur_list[6] == 0:
#         continue
#     lead_left = 0
#     lead_list = data.loc[lead_left].values[:]
#     while lead_left < max_val:
#         lead_list = data.loc[lead_left].values[:]
#         lead_left += 1
#         if cur_list[6] == lead_list[1] and lead_list[0] == cur_list[0]:
#             break
#     period = []
#     while cur_list[6] == lead_list[1]:
#         temp = [cur_list[0], cur_list[1], cur_list[2], cur_list[3], lead_list[2], lead_list[3],
#                 cur_list[4], cur_list[5], lead_list[4], lead_list[5]]
#         period.append(temp)
#         cur_left += 1
#         lead_left += 1
#         if cur_left >= max_val or lead_left >= max_val:
#             break
#         cur_list = data.loc[cur_left].values[:]
#         lead_list = data.loc[lead_left].values[:]
#     if len(period) >= 81:
#         content += period


print(len(content))
output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_fin.csv', index=False)
