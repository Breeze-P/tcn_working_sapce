import torch.nn as nn
from torch.nn.utils import weight_norm


# 定义实现因果卷积的类（继承自类nn.Module），其中super(Chomp1d, self).__init__()表示对继承自父类的属性进行初始化。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        # 通过增加Padding的方式并对卷积后的张量做切片而实现因果卷积

    def forward(self, x):  # 重写forward层定义各层 之间的关系
        return x[:, :, :-self.chomp_size].contiguous()


# chomp_size 等于 padding=(kernel_size-1) * dilation_size，
# x 为一般一维空洞卷积后的结果。张量 x 的第一维是批量大小，第二维是通道数量而第三维就是序列长度。
# 如果我们删除卷积后的倒数 padding 个激活值，就相当于将卷积输出向左移动 padding 个位置而实现因果卷积。


# 定义残差块，即两个一维卷积与恒等映射
class TemporalBlock(nn.Module):
    # kernel_size 卷积核大小  n_inputs 输入  n_outputs 输出  stride 步长
    # padding 填充  dilation 膨胀/扩张  dropout丢弃率
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定义第一个空洞卷积层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 根据第一个卷积层的输出与padding大小实现因果卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()  # 线性激活函数
        self.dropout1 = nn.Dropout(dropout)  # dropout正则化

        # 定义第二个空洞卷积层，结构和前面的完全一样
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        #  时序模型，将以上所有模块堆叠到一起
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # padding保证了输入序列与输出序列的长度相等，但卷积前的通道数与卷积后的通道数不一定一样。
        # 如果通道数不一样，那么需要对输入x做一个逐元素的一维卷积以使得它的纬度与前面两个卷积相等。
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    # 初始化权重为从均值为0，标准差为0.01的正态分布中采样的随机值
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # 结合卷积与输入的恒等映射（或输入的逐元素卷积），并投入ReLU 激活函数完成残差模块
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TCN主体
class TemporalConvNet(nn.Module):
    #  num_channels通道长度(根据其他参数自己算出来的)
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # num_channels为各层卷积运算的输出通道数或卷积核数量，它的长度即需要执行的卷积层数量

        # 空洞卷积的扩张系数若随着网络层级的增加而成指数级增加，则可以增大感受野并不丢弃任何输入序列的元素
        for i in range(num_levels):
            # 扩张系数
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i] # 输入和输出残差为1
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
