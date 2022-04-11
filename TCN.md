# TCN

Temporal convolutional network 时间卷积网络

三个重要概念：

* Causal Conv：因果卷积
* Dilated Conv：空洞卷积
* Residual block：残存块

研究模型：sequence model 时序模型



研究点：the memory retention characteristics of recurrent networks 记忆保留特性

TCN优势场景： where a long history is required.



TCN的两个重要特征：

* 输入输出长度相同
* 不受未来渗透，只接受历史的影响

实现方式：TCN = 1D FCN + causal convolutions

缺点：为了感受有效的、超长的历史长度，我们需要更深的网络结构或一个超大的过滤器（fliter）。

解决方法：空洞卷积——指数级别增大感受野 ⬆️（不需要那么深的网络）；残存块避免deeper network的问题 （解决过深网路的问题）⬇️；



1 D FCN：

> Each hidden layer is the same length as the input layer, and zero padding of length (kernel size − 1) is added to keep subsequent layers the same length as previous ones. 
>
> 每个隐藏层与输入层的长度相同，并且添加长度的零填充（内核大小 - 1）以保持后续层与前一层的长度相同。



causal convolution：

> convolutions where an output at time t is convolved only with elements from time t and earlier in the previous layer.
>
> 因果卷积，其中时间 t 的输出仅与时间 t 和前一层中更早的元素进行卷积。



residual block：<a href="https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec">ResNet</a>

> Since a TCN’s receptive field depends on the network depth **n** as well as filter size **k** and dilation factor **d**, stabilization of deeper and larger TCNs becomes important.

> Our residual block is overall trying to learn the true output, *H(x). If* you look closely at the image above, you will realize that since we have an identity connection coming from *x*, the layers are actually trying to learn the residual, *R(x)*. So to summarize, the layers in a traditional network are learning the true output (*H(x)*), whereas the layers in a residual network are learning the residual (*R(x)*). Hence, the name: *Residual Block*.



deep network的问题：

* 梯度爆炸/消失
* 过拟合



### TCNs优缺点

Advantages：

* Parallelism: 并行

* Flexible receptive field size: 灵活的感受野大小
* **Stable gradients**: 稳定的提梯度「避免了梯度爆炸的问题」
* Low memory requirement for training: 训练时需要更少的内存
* Variable length inputs: 可变长度输入「通过滑动1D卷积核来接受任意长度的输入」

Disadvantages：

* Data storage during evaluation：测试间断需要跟多的内存（需要有效历史长度的数据）
* **Potential parameter change for a transfer of domain**：不同场景适配的「历史长度」不同，需要调整「k」、「d」以更改感受野（？不是越大越好）



### 论文结构索引

1. Introduction
2. Background
3. TCN 介绍主要方法：时间卷积网络
   1. 时序模型
   2. Causal Convolutions：因果卷积
   3. Dilated Convolutions：空洞卷积
   4. Residual Connections： 残存连接 避免梯度消失
4. Discusion: 总结
5. 



待查名词：

* 卷积网络
  * channel_num：想到RGB的例子
  * 作为特征提取的工具

* kernel
* filter
* ReLU 整流线性单元「非线性层」
* WeightNorm 归一化



链接：

* <a href="https://blog.csdn.net/qq_36269513/article/details/80420363">fcn(全卷积神经网络)和cnn(卷积神经网络)的区别</a>
* <a href="https://www.zhihu.com/question/54149221">空洞卷积的理解</a>
* <a href="https://blog.csdn.net/program_developer/article/details/80958716">感受野的理解</a>
* <a href="https://blog.csdn.net/qq_27586341/article/details/90751794">TCN概览</a>
* <a href="https://blog.csdn.net/qq_27825451/article/details/90550890">torch.nn.module</a>
* <a href="https://blog.csdn.net/qq_34107425/article/details/105522916">卷积角度理解</a>





汇报：

介绍TCN

规划：一周掌握这个网络（还原部分实验）（同步郐世扬调研应用场景）「兜底场景：基本的跟车模型」



可操作点：





### 对port_music的调研

* 卷积（Convolution）、填充（Padding）「避免边缘信息损失」、步长(Stride)「压缩细腻」。

* filter：卷积核的集合

* Channels：CNN模型的宽度，隧道的数量。（RGB）

> **某一层滤波器的通道数 = 上一层特征图的通道数。**如上图所示，我们输入一张 ![[公式]](https://www.zhihu.com/equation?tex=6%5Ctimes6%5Ctimes3) 的RGB图片，那么滤波器（ ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3%5Ctimes3) ）也要有三个通道。
>
> **某一层输出特征图的通道数 = 当前层滤波器的个数。**如上图所示，当只有一个filter时，输出特征图（ ![[公式]](https://www.zhihu.com/equation?tex=4%5Ctimes4) ）的通道数为1；当有2个filter时，输出特征图（![[公式]](https://www.zhihu.com/equation?tex=4%5Ctimes4%5Ctimes2) )的通道数为2。
>
> <img src="https://pic3.zhimg.com/80/v2-fc70463d7f82f7268ee23b7235515f4a_1440w.jpg" alt="img" style="zoom:67%;" />

* 激活函数作用：非线性处理。

* forward：定义模型前向传播方法。
* 实验为了得到超参kernel_size、num_channels(nhid、levels)



### 应用场景方案/实验方案制定

#### 总体思路

1. TCN单一场景预测优化「提升x, y的准确性」（图标分析：loss曲线、预测曲线）
   1. 单步预测
   2. 多步预测
2. 横向比较：
   1. 同样的场景（channel_num/input_size/特征数量、时间步长）的情况下，不同模型之间的效果比较（预测曲线）
   2. 不及的地方skip至step1进一步提高模型精度
3. 论文产出



#### 分工

数据处理、「网络结构设计、实验调参」、matplotlib数据可视化、论文结构调研与反哺实验步骤



#### Plan I

简单跟车场景：

**INIT** 「bitch, channel_num, timestep」

输入维度：「32, 4, 80」

* 32：bitch大小
* 4：channel_num 4个维度的入参特征，包含目标🚗(Target)在某一时刻的x, y和跟驰🚗(Subject)在某一时刻的x, y
* 80：timestep时间步长

输出维度：「32, 2, 1」

* 2：下一时刻跟驰🚗的x, y

可调整参数：

* 数据层面
  * channel_num的数量，选取更多特征「加速度、速度、七车...」
  * timestep，提升/减少时间步长，提升预测准确性/提速模型收敛
* 网络结构层面
  * kernel_size 卷积核大小
  * nhids 隐藏层细胞数
  * level 层数
  * epoch 轮数
  * 激活函数、损失函数、优化函数、非线性、过拟合...



both in terms of convergence and in final accuracy on the task.

收敛性：loss下降的速度、最终的准确性、在不同时间步长下的适应性（longer memory下更具优势、「channel_num参数数量这个应该和具体的场景有关

### 实验记录

### TCN

| knize | level | nhid | epoch | 结果                                                         |
| ----- | ----- | ---- | ----- | ------------------------------------------------------------ |
| 4     | 4     | 32   | 10    | Test set: Average loss: 2.328249<br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409210635079.png" alt="image-20220409210635079" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409210643228.png" alt="image-20220409210643228" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409210702541.png" alt="image-20220409210702541" style="zoom:50%;" /> |
| 2     | 4     | 32   | 10    | Test set: Average loss: 27.643204<br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409000409118.png" alt="image-20220409000409118" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409000420777.png" alt="image-20220409000420777" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220409000458623.png" alt="image-20220409000458623" style="zoom:50%;" /> |
| 8     | 4     | 32   | 10    | <img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408152925016.png" alt="image-20220408152925016" style="zoom:50%;" /><br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408152941686.png" alt="image-20220408152941686" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408152951686.png" alt="image-20220408152951686" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408153034456.png" alt="image-20220408153034456" style="zoom:50%;" /> |
| 6     | 4     | 32   | 10    | <img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408162426817.png" alt="image-20220408162426817" style="zoom:50%;" /><br /><br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408162436237.png" alt="image-20220408162436237" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408162448895.png" alt="image-20220408162448895" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408162519240.png" alt="image-20220408162519240" style="zoom:50%;" /> |
| 4     | 8     | 32   | 10    | Test set: Average loss: 43.825653<br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408183851142.png" alt="image-20220408183851142" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408183942857.png" alt="image-20220408183942857" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408184058501.png" alt="image-20220408184058501" style="zoom:50%;" /> |
| 4     | 4     | 64   | 10    | Test set: Average loss: 35.814930<br /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408222433462.png" alt="image-20220408222433462" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408222449278.png" alt="image-20220408222449278" style="zoom:50%;" /><img src="/Users/bytedance/Library/Application Support/typora-user-images/image-20220408222552234.png" alt="image-20220408222552234" style="zoom:50%;" /> |
|       |       |      |       |                                                              |
|       |       |      |       |                                                              |

### LSTM

|      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |

数据重新筛选算法：

* 时间（frameID）与Y的排序
* 找、构建跟车对（前后两车在同一车道且检举小于100），保留前后车x、y、a、v，时间，车辆ID，车道
* 跟车对依据目标车ID和时间、跟车ID排序，通过过时间和ID的方式构造标签



* 80时间步长、i-80路段的各个网络比较：基本的TCN优势
* 60、100时间步长下、i-80路段各个网络比较：不同时间步长下网络的收敛性
* 80时间步长、其他路段的各个网路比较：泛化性
