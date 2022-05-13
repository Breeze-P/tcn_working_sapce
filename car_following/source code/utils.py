import matplotlib.pyplot as plt

# 绘图函数
def draw_predict(t, y, y_=[], x_label='time', y_label='Index Y', title=''):
    flag = len(y_) > 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t, y, 'bo-')
    if flag:
      ax.plot(t, y_, 'r+-')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.show()


def draw_predict_combine(loss, y, y_):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ax1.plot(range(1, len(loss) + 1), loss, 'bo-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('val_loss')
    ax2.plot(range(200), y, 'bo-', range(200), y_, 'r+-')
    ax2.set_ylabel('Index Y')

    fig.show()


import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(data)  # (3, 3)
sel_data = data[:, [1, 0, 2]]
print(sel_data)