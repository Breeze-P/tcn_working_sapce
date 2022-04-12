# 第五次处理，将排序好的数据按时间步长排序，长于时间步长的数据按10frame后移按新一段数据处理
import pandas as pd

# data = pd.read_csv("123.csv", header=0)
data = pd.read_csv("sorted_us-101_new.csv", header=0)

name = ['Vehicle_ID', 'Frame_ID', 'cur_x', 'cur_y', 'lead_x', 'lead_y', 'cur_v', 'cur_a', 'lead_v', 'lead_a', 'Lane_ID']
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
    frame = cur_list[1]
    right = left + 1
    if right >= max_val:
        break
    count = 0
    sign = 0
    next_list = data.loc[right].values[:]
    while next_list[0] == cur_list[0] and next_list[1] == frame + 1:
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
        while cur_list[0] == next_list[0]:
            left += 10
            right += 10
            if right >= max_val:
                break
            next_list = data.loc[right - 1].values[:]
            if next_list[1] == frame + 10:
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
output.to_csv('finalResult_us-101_new.csv', index=False)
