
import pandas as pd

# data = pd.read_csv("123.csv", header=0)
data = pd.read_csv("HighD_1_merge_sorted.csv", header=0)

name = ['Frame_ID', 'Vehicle_ID', 'cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'Lane_ID']
content = []
max_val = data.shape[0]
sign = 0
result = 0
left = 0


def printCurrent(valid, total):
    if valid % 1000 == 0:
        print(f"Current Situation:{valid} valid data and {total} data passed")


while left < max_val:
    cur_list = data.loc[left].values[:]
    frame = cur_list[0]
    right = left + 1
    if right >= max_val:
        break
    count = 0
    sign = 0
    next_list = data.loc[right].values[:]
    while next_list[1] == cur_list[1] and next_list[0] == frame + 1:
        frame += 1
        right += 1
        if right >= max_val:
            break
        if right - left == 81:
            period = data[left: right]
            period = period.values.tolist()
            content += period
            result += 1
            sign = 1
            printCurrent(result, left)
            break
        next_list = data.loc[right].values[:]
    if sign == 1:
        while cur_list[1] == next_list[1]:
            left += 10
            right += 10
            if right >= max_val:
                break
            next_list = data.loc[right - 1].values[:]
            if next_list[0] == frame + 10:
                period = data[left: right]
                period = period.values.tolist()
                content += period
                result += 1
                frame += 10
                printCurrent(result, left)
            else:
                left = right - 11
                break
    left += 1

print("###################")
printCurrent(result, left)
output = pd.DataFrame(columns=name, data=content)
output.to_csv('HighD_1_fin.csv', index=False)
