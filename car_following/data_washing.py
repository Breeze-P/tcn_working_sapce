
import pandas as pd

# data = pd.read_csv("123.csv", header=0)
data = pd.read_csv("sorted_i-80_new.csv", header=0)

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
        if right - left == 61:
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

# while one < max_val:
#     cur_list = data.loc[one].values[:]
#     frame = cur_list[1]
#     two = one + 1
#     if two >= max_val:
#         break
#     count = 0
#     period = [cur_list]
#     next_list = data.loc[two].values[:]
#     while cur_list[0] == next_list[0]:
#         if next_list[1] == frame + 1:
#             period.append(next_list)
#             count += 1
#             frame = next_list[1]
#         if count == 80:
#             content += period
#             result += 1
#             one += 9
#             if result % 1000 == 0:
#                 print(result)
#             break
#         two += 1
#         if two >= max_val:
#             break
#         next_list = data.loc[two].values[:]
#     one += 1
print("###################")
printCurrent(result, left)
output = pd.DataFrame(columns=name, data=content)
output.to_csv('finalResult_new.csv', index=False)
