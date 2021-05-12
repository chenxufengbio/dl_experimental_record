# CNN

![](https://pic.imgdb.cn/item/60953d44d1a9ae528f098183.jpg)

卷积神经网络（Convolutional Neural Networks, CNN）是一类包含卷积计算且具有深度结构的前馈神经网络（Feedforward Neural Networks），由若干个卷积层和池化层组成，是深度学习（deep learning）的代表算法之一，尤其在图像处理方面的表现十分出色。



> - 1962年，Hubel和Wiesel 通过对猫脑视觉皮层的研究，首次提出了一种新的概念“感受野”，这对后来人工神经网络的发展有重要的启示作用。**感受野（Receptive Field）**是卷积神经网络每一层输出的特征图（feature map）上的像素点在输入图片上映射的区域大小。再通俗点的解释是，特征图上的一个点对应输入图上的区域。
> - 1980年，Fukushima基于生物神经学的感受野理论提出了神经认知机和权重共享的卷积神经层，这被视为卷积神经网络的雏形。
> - 1989年，LeCun结合反向传播算法与权值共享的卷积神经层发明了卷积神经网络，并首次将卷积神经网络成功应用到美国邮局的手写字符识别系统中。
> - 1998年，LeCun提出了卷积神经网络的经典网络模型LeNet-5，并再次提高手写字符识别的正确率。





## 基本结构



 CNN的基本结构由**输入层、卷积层（convolutional layer）、ReLU层（Rectified Linear Units layer）**、**池化层（pooling layer，也称为取样层）、全连接层及输出层**构成。卷积层和池化层一般会取若干个，采用卷积层和池化层交替设置，即一个卷积层连接一个池化层，池化层后再连接一个卷积层，依此类推。由于卷积层中输出特征图的每个神经元与其输入进行局部连接，并通过对应的**连接权值**与**局部输入**进行加权求和再加上偏置值，得到该神经元输入值，该过程等同于卷积过程，CNN也由此而得名。



### 卷积层（Convolution Layer）

卷积层是构建卷积神经网络的**核心层**，它产生了网络中大部分的**计算量**。卷积神经网路中每层卷积层由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法优化得到的。卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征。

#### **局部感知（local Connectivity）**

**普通神经网络**把输入层和隐藏层进行“**全连接(Full Connected)**“的设计。从计算的角度来讲，相对较小的图像从整幅图像中计算特征是可行的。但是，如果是更大的图像（如 96x96 的图像），要通过这种全连接网络的这种方法来学习整幅图像上的特征，**从计算角度而言，将变得非常耗时**。你需要设计 10 的 4 次方（=10000）个输入单元，假设你要学习 100 个特征，那么就有 10 的 6 次方个参数需要去学习。与 28x28 的小块图像相比较， 96x96 的图像使用前向输送或者后向传导的计算方式，计算过程也会慢 10 的 2 次方（=100）倍。

所以如果用全连接神经网络处理大尺寸图像具有三个明显的缺点：

（1）首先将图像展开为向量会丢失空间信息；

（2）其次参数过多效率低下，训练困难；

（3）同时大量的参数也很快会导致网络**过拟合**。

卷积层解决这类问题的一种简单方法是**对隐含单元和输入单元间的连接加以限制**：每个隐含单元仅仅只能连接输入单元的一部分。例如，每个隐含单元仅仅连接输入图像的一小片相邻区域。（对于不同于图像输入的输入形式，也会有一些特别的连接到单隐含层的输入信号“连接区域”选择方式。如音频作为一种信号输入方式，一个隐含单元所需要连接的输入单元的子集，可能仅仅是一段音频输入所对应的某个时间段上的信号。)

每个隐含单元连接的输入区域大小叫r神经元的**感受野(receptive field)**。

由于卷积层的神经元也是三维的，所以也具有深度。卷积层的参数包含一系列**过滤器（filter）**（卷积核/Kernel），每个过滤器训练一个深度，有几个过滤器输出单元就具有多少深度。

具体如下图所示，样例输入单元大小是32×32×3, 输出单元的深度是5, 对于输出单元不同深度的同一位置，与输入图片连接的区域是相同的，但是参数（过滤器）不同。

虽然每个输出单元只是连接输入的一部分，但是值的计算方法是没有变的，都是**权重和输入的点积，然后加上偏置**，这点与普通神经网络是一样的

![](https://pic.imgdb.cn/item/609685dad1a9ae528fae8ba5.jpg)



![](https://pic.imgdb.cn/item/60968743d1a9ae528fbf8245.jpg)

#### **参数（权值）共享（Parameter Sharing）**

应用参数共享可以大量减少参数数量，参数共享基于一个假设：如果图像中的一点（x1, y1）包含的特征很重要，那么它应该和图像中的另一点（x2, y2）一样重要。换种说法，我们把同一深度的平面叫做深度切片(depth slice)（(e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55])），那么同一个切片应该共享同一组权重和偏置。我们仍然可以使用梯度下降的方法来学习这些权值，只需要对原始算法做一些小的改动， 这里共享权值的梯度是所有共享参数的梯度的总和。

我们不禁会问为什么要权重共享呢？一方面，重复单元能够对特征进行识别，而不考虑它在可视域中的位置。另一方面，权值共享使得我们能更有效的进行特征抽取，因为它极大的减少了需要学习的自由变量的个数。通过控制模型的规模，卷积网络对视觉问题可以具有很好的泛化能力。

#### **神经元的空间排列**（**Spatial arrangement**）

一个输出单元的大小有以下三个量控制：depth, stride 和 zero-padding。

**深度(depth) :** 卷积核的个数，顾名思义，它控制输出单元的深度，也就是filter的个数，连接同一块区域的神经元个数。又名：depth column
**步幅(stride)：**它控制在同一深度的相邻两个隐含单元，与他们相连接的输入区域的距离。如果步幅很小（比如 stride = 1）的话，相邻隐含单元的输入区域的重叠部分会很多; 步幅很大则重叠区域变少。
**补零(zero-padding) ：** 我们可以通过在输入单元周围补零来改变输入单元整体大小，从而控制输出单元的空间大小
**感受野（receptive field）：**Kernel size



#### 卷积操作（Convolution）

虽然卷积层得名于卷积（convolution）运算，但我们通常在卷积层中使用更加直观的互相关（cross-correlation）运算。在二维卷积层中，一个二维输入数组和一个二维核（kernel）数组通过互相关运算输出一个二维数组。 我们用一个具体例子来解释二维互相关运算的含义。如图5.1所示，输入是一个高和宽均为3的二维数组。我们将该数组的形状记为（3，3）。核数组的高和宽分别为2。该数组在卷积计算中又称卷积核或过滤器（filter）。卷积核窗口（又称卷积窗口）的形状取决于卷积核的高和宽，即2×2

下图阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：0×0+1×1+3×2+4×3=19

![](https://pic.imgdb.cn/item/60968913d1a9ae528fd586af.jpg)

![](https://pic.imgdb.cn/item/6096974ad1a9ae528f905b07.jpg)







### 池化层（Pooling Layer）

通常在卷积层之后会得到维度很大的特征，将特征切成几个区域，取其最大值或平均值，得到新的、维度较小的特征。

池化（pool）即下采样（downsamples），目的是为了减少特征图。池化操作对每个深度切片独立，规模一般为 2＊2，相对于卷积层进行卷积运算，池化层进行的运算一般有以下几种：
* 最大池化（Max Pooling）。取4个点的最大值。这是最常用的池化方法。
* 均值池化（Mean Pooling）。取4个点的均值。
* 高斯池化。借鉴高斯模糊的方法。不常用。
* 可训练池化。训练函数 ff ，接受4个点为输入，出入1个点。不常用。

最常见的池化层是规模为2*2， 步幅为2，对输入的每个深度切片进行下采样。每个MAX操作对四个数进行，如下图所示：
![](https://pic.imgdb.cn/item/6097ebc3d1a9ae528f254e6a.png)



池化操作将保存**深度大小不变**。

如果池化层的输入单元大小不是二的整数倍，一般采取边缘补零（zero-padding）的方式补成2的倍数，然后再池化。

![](https://pic.imgdb.cn/item/6097ebdbd1a9ae528f268ac4.png)





### 全连接层（Fully-Connected Layer）

把所有局部特征结合变成全局特征，用来计算最后每一类的得分。

对于任意一个卷积层，要把它变成全连接层只需要把权重变成一个巨大的矩阵，其中大部分都是0 除了一些特定区块（因为局部感知），而且好多区块的权值还相同（由于权重共享）。
* 相反地，对于任何一个全连接层也可以变为卷积层。比如，一个K＝4096 的全连接层，输入层大小为 7∗7∗512，它可以等效为一个 F=7, P=0, S=1, K=4096F=7, P=0, S=1, K=4096 的卷积层。换言之，我们把 filter size 正好设置为整个输入层大小。



## 实战应用

### Pytorch实现

详见[构建基本模型（pytorch）](docs/实验流程/构建模型.md)

### 生物交叉应用案例

随着高通量技术的发展，大量的组学数据产出，不断地挑战现有的传统分析方法，如何将这些生物数据转化为有价值的知识是生物信息学面临的重要问题之一，自本世纪初以来，深度学习得到了迅速发展，如今在各个领域都展现了最先进的表现，在生物信息学领域，深度学习目前的应用也已经比较广泛。

 卷积神经网络（CNNs）因其出色的空间信息分析能力而成为最成功的图像处理深度学习模型之一。**CNNs在基因组学中的应用依赖于卷积层对于图像特征的提取**。Zeng等人描述了通过将基因组序列的一个bin理解为图像来实现CNNs从计算机视觉领域到基因组学的迁移(Zeng et al 2016)。

卷积神经网络的亮点是训练过程中**自适应特征提取**的灵活性。例如，CNNs可用于发现有意义的小变异重复模式，如基因组序列基序，这使得CNNs适用于motif识别和序列的分类。Zhou和Troyanskaya等人早在2015年就通过基于深度学习的算法模型**DeepSEA**实现了预测单核苷酸敏感性序列的改变的染色质效应，并且通过电子诱变可以解析任何序列中的信息特征，为研究单核苷酸多态性的功能效应提供了有效的方法（Zhou and Troyanskaya 2015）。Xiong等人则研究出了**Deep Bayes**模型，该模型仅使用DNA序列进行训练，能够准确地对致病变异体进行分类，并对异常剪接在疾病中的作用提供新的假设（Xiong et al 2015）。Alipanahi等人和Zeng等人成功应用CNNs对蛋白质结合的序列特异性进行建模学习（Alipanahi et al 2015），这些应用都已经证明CNNs在针对基序识别的问题上处于领先地位。此外递归神经网络（RNNs）和自动编码器（autoencoder）也在不同的生物领域展现了很优秀的性能。

随着深度学习在基因组学中不断显示出成功，研究人员期望深度学习比简单的统计或机器学习方法具有更高的准确性。为此，当今绝大多数的工作都是从超越经典深度架构的更高级模型，或者采用混合模型来处理基因组问题。Angermueller等人利用两个CNN子模型和一个融合模块来预测DNA甲基化状态。CNN的两个子模型采用不同的输入，因此专注于不同的目的。CpG模块拟合细胞内和跨细胞的CpG位点之间的相关性，而DNA模块检测信息序列模式(基序)（Angermueller et al 2017）。DanQ 是一种混合卷积和递归深度神经网络，用于直接从序列中预测非编码DNA的功能。脱氧核糖核酸序列作为四个碱基的onehot表示输入到一个简单的CNN中，目的是扫描基序位点（Quang and Xie 2016）。

**以下两个github项目是对于深度学习在生物领域的应用进行的总结：**

[deeplearning-biology](https://github.com/hussius/deeplearning-biology)

[Deep Learning for Genomics: A Concise Overview](https://github.com/klsfct/DLforGenomics)

**接下来主要对本次项目中使用的两个CNN进行结构解析**：

### DeepSEA

```python
class DeepSEA(nn.Module):
    def __init__(self, sequence_length=500, n_genomic_features=2):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        reduce_by = conv_kernel_size - 1

        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        predict = self.classifier(reshape_out)
        return predict
```

```
DeepSEA(
  (conv_net): Sequential(
    (0): Conv1d(4, 320, kernel_size=(8,), stride=(1,))
    (1): ReLU(inplace=True)
    (2): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
    (4): Conv1d(320, 480, kernel_size=(8,), stride=(1,))
    (5): ReLU(inplace=True)
    (6): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    (7): Dropout(p=0.2, inplace=False)
    (8): Conv1d(480, 960, kernel_size=(8,), stride=(1,))
    (9): ReLU(inplace=True)
    (10): Dropout(p=0.5, inplace=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=21120, out_features=2, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=2, out_features=2, bias=True)
    (3): Sigmoid()
  )
)
```

##### Conv_net

##### （0）**一维卷积层 Conv1d**

https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

- in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度
- out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
- kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
- stride(int or tuple, optional) - 卷积步长
- padding (int or tuple, optional)- 输入的每一条边补充0的层数
- dilation(int or tuple, `optional``) – 卷积核元素之间的间距
- groups(int, optional) – 从输入通道到输出通道的阻塞连接数
- bias(bool, optional) - 如果bias=True，添加偏置

**Conv1d(4, 320, kernel_size=(8,), stride=(1,))**

##### （1）**激活函数 ReLU层**

https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU

**ReLU(inplace=True)**

```python
class torch.nn.ReLU(inplace=False)
```

inplace参数，如果inplace = True，则输出会直接覆盖到输入中，可以节省显存

ReLU(x)=max(0, x)

![](https://pic.imgdb.cn/item/60974a33d1a9ae528f91f839.jpg)

从ReLU函数图像可知，它是分段线性函数，所有的负值和0为0，所有的正值不变，这种操作被称为单侧抑制。ReLU函数图像其实也可以不是这个样子，只要能起到单侧抑制的作用，对原图翻转、镜像都可以。

当训练一个深度分类模型的时候，和目标相关的特征往往也就那么几个，因此通过ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据。正因为有了这单侧抑制，才使得神经网络中的神经元也具有了稀疏激活性。尤其体现在深度神经网络模型(如CNN)中，当模型增加N层之后，理论上ReLU神经元的激活率将降低2的N次方倍。

不用simgoid和tanh作为激活函数，而用ReLU作为激活函数的原因是：加速收敛。因为sigmoid和tanh都是饱和(saturating)的。何为饱和？可理解为把这两者的函数曲线和导数曲线plot出来：他们的导数都是倒过来的碗状，也就是越接近目标，对应的导数越小。而ReLu的导数对于大于0的部分恒为1。于是ReLU确实可以在BP的时候能够将梯度很好地传到较前面的网络。


#####  **(2): 池化层 MaxPool1d**

https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html?highlight=maxpool1d#torch.nn.MaxPool1d

```python
class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

- kernel_size(int or tuple) - max pooling的窗口大小

- stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size

- padding(int or tuple, optional) - 输入的每一条边补充0的层数

- dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数

- return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助

- ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

**MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)**



##### **(3): 正则化层 Dropout**

https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout

```python
class torch.nn.Dropout(p=0.5, inplace=False)
```

- p:被舍弃的概率，失活概率
- inplace:输出是否覆盖输入

 **Dropout(p=0.2, inplace=False)**

后面的六层与前面四层是重复的

```
CONV1D -> RELU -> MaxPool1d -> Dropout
```

​    (4): Conv1d(320, 480, kernel_size=(8,), stride=(1,))
​    (5): ReLU(inplace=True)
​    (6): MaxPool1d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
​    (7): Dropout(p=0.2, inplace=False)
​    (8): Conv1d(480, 960, kernel_size=(8,), stride=(1,))
​    (9): ReLU(inplace=True)
​    (10): Dropout(p=0.5, inplace=False)

#### classifier

**对卷积层输出的结果进行reshape，最终得到二元分类的预测值**

(0): Linear(in_features=21120, out_features=2, bias=True)
(1): ReLU(inplace=True)
(2): Linear(in_features=2, out_features=2, bias=True)
(3): Sigmoid()



### DeepHsitone

![](https://pic.imgdb.cn/item/60977bc9d1a9ae528fa37139.jpg)

![](https://pic.imgdb.cn/item/60977e70d1a9ae528fc6767a.jpg)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
from torch.optim import  Optimizer
import math 
from torch.nn.parameter import Parameter

class BasicBlock(nn.Module):
	def __init__(self, in_planes, grow_rate,):
		super(BasicBlock, self).__init__()
		self.block = nn.Sequential(
			nn.BatchNorm2d(in_planes),
			nn.ReLU(),
			nn.Conv2d(in_planes, grow_rate, (1,9), 1, (0,4)),
			#nn.Dropout2d(0.2)
		)
	def forward(self, x):
		out = self.block(x)
		return torch.cat([x, out],1)

class DenseBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, grow_rate,):
		super(DenseBlock, self).__init__()
		layers = []
		for i in range(nb_layers):
			layers.append(BasicBlock(in_planes + i*grow_rate, grow_rate,))
		self.layer = nn.Sequential(*layers)
	def forward(self, x):
		return self.layer(x)


class ModuleDense(nn.Module):
	def __init__(self):
		super(ModuleDense, self).__init__()


		self.conv1 = nn.Sequential(
			nn.Conv2d(1,128,(4,9),1,(0,4)),
			#nn.Dropout2d(0.2),
			)
	
		self.block1 = DenseBlock(3, 128, 128)	
		self.trans1 = nn.Sequential(
			nn.BatchNorm2d(128+3*128),
			nn.ReLU(),
			nn.Conv2d(128+3*128, 256, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,4)),
		)
		self.block2 = DenseBlock(3,256,256)
		self.trans2 = nn.Sequential(
			nn.BatchNorm2d(256+3*256),
			nn.ReLU(),
			nn.Conv2d(256+3*256, 512, (1,1),1),
			#nn.Dropout2d(0.2),
			nn.MaxPool2d((1,4)),
		)
		self.out_size = 500 // 4 // 4  * 512

	def forward(self, seq):
		n, h, w = seq.size()
		
		seq = seq.view(n,1,4,w)
	
		out = self.conv1(seq)
		out = self.block1(out)
		out = self.trans1(out)
		out = self.block2(out)
		out = self.trans2(out)
		n, c, h, w = out.size()
		out = out.view(n,c*h*w) 
		return out



class NetDeepHistone(nn.Module):
	def __init__(self):
		super(NetDeepHistone, self).__init__()
		print('DeepHistone(Dense) is used.')
		self.seq_map = ModuleDense()
		self.seq_len = self.seq_map.out_size
		seq_len = self.seq_len

		self.linear_map = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(int(seq_len),925),
			nn.BatchNorm1d(925),
			nn.ReLU(),
			#nn.Dropout(0.1),
			nn.Linear(925,5),
			nn.Sigmoid(),
		)

	def forward(self, seq):
		flat_seq = self.seq_map(seq)	
		out = self.linear_map(flat_seq)
		return out


class DeepHistone():
	def __init__(self,use_gpu,learning_rate=0.001):
		self.forward_fn = NetDeepHistone()
		self.criterion  = nn.BCELoss()
		self.optimizer  = optim.Adam(self.forward_fn.parameters(), lr=learning_rate, weight_decay = 0)
		self.use_gpu    = use_gpu
		if self.use_gpu : self.criterion,self.forward_fn = self.criterion.cuda(), self.forward_fn.cuda()

	def updateLR(self, fold):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] *= fold

	def train_on_batch(self,seq_batch,lab_batch,): 
		self.forward_fn.train()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		lab_batch  = Variable(torch.Tensor(lab_batch))
		if self.use_gpu: seq_batch, lab_batch = seq_batch.cuda(), lab_batch.cuda()
		output = self.forward_fn(seq_batch)
		loss = self.criterion(output,lab_batch)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.cpu().data

	def eval_on_batch(self,seq_batch,lab_batch,):
		self.forward_fn.eval()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		lab_batch  = Variable(torch.Tensor(lab_batch))
		if self.use_gpu: seq_batch,  lab_batch = seq_batch.cuda(), lab_batch.cuda()
		output = self.forward_fn(seq_batch)
		loss = self.criterion(output,lab_batch)
		return loss.cpu().data,output.cpu().data.numpy()
			
	def test_on_batch(self, seq_batch):
		self.forward_fn.eval()
		seq_batch  = Variable(torch.Tensor(seq_batch))
		if self.use_gpu: seq_batch = seq_batch.cuda()
		output = self.forward_fn(seq_batch)
		pred = output.cpu().data.numpy()
		return pred
	
	def save_model(self, path):
		torch.save(self.forward_fn.state_dict(), path)


	def load_model(self, path):
		self.forward_fn.load_state_dict(torch.load(path))
```



```
NetDeepHistone(
  (seq_map): ModuleDense(
    (conv1): Sequential(
      (0): Conv2d(1, 128, kernel_size=(4, 9), stride=(1, 1), padding=(0, 4))
    )
    (block1): DenseBlock(
      (layer): Sequential(
        (0): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
        (1): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(256, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
        (2): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(384, 128, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
      )
    )
    (trans1): Sequential(
      (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): ReLU()
      (2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (3): MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
    )
    (block2): DenseBlock(
      (layer): Sequential(
        (0): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(256, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
        (1): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(512, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
        (2): BasicBlock(
          (block): Sequential(
            (0): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(768, 256, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
          )
        )
      )
    )
    (trans2): Sequential(
      (0): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): ReLU()
      (2): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
      (3): MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0, dilation=1, ceil_mode=False)
    )
  )
  (linear_map): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=15872, out_features=925, bias=True)
    (2): BatchNorm1d(925, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Linear(in_features=925, out_features=5, bias=True)
    (5): Sigmoid()
  )
)
```

#### seq_map



##### 二维卷积层 conv2d

```python
class   torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=,bias=True,padding_mode='zero')
```



- in_channels:就是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。这个形参是确定权重等可学习参数的shape所必需的。

- out_channels:也很好理解，即期望的四维输出张量的channels数，不再多说。

- kernel_size:卷积核的大小，一般我们会使用5x5、3x3这种左右两个数相同的卷积核，因此这种情况只需要写kernel_size = 5这样的就行了。如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）。

- stride = 1:卷积核在图像窗口上每次平移的间隔，即所谓的步长。这个概念和Tensorflow等其他框架没什么区别，不再多言。

- padding = 0:Pytorch与Tensorflow在卷积层实现上最大的差别就在于padding上。
    Padding即所谓的图像填充，后面的int型常数代表填充的多少（行数、列数），默认为0。需要注意的是这里的填充包括图像的上下左右，以padding = 1为例，若原始图像大小为32x32，那么padding后的图像大小就变成了34x34，而不是33x33。
      Pytorch不同于Tensorflow的地方在于，Tensorflow提供的是padding的模式，比如same、valid，且不同模式对应了不同的输出图像尺寸计算公式。而Pytorch则需要手动输入padding的数量，当然，Pytorch这种实现好处就在于输出图像尺寸计算公式是唯一的，即

      ![](https://pic.imgdb.cn/item/60977fe7d1a9ae528fd8cdcd.jpg)

   当然，上面的公式过于复杂难以记忆。大多数情况下的kernel_size、padding左右两数均相同，且不采用空洞卷积（dilation默认为1），因此只需要记 O = （I - K + 2P）/ S +1这种在深度学习课程里学过的公式就好了。

- dilation = 1:这个参数决定了是否采用空洞卷积，默认为1（不采用）。从中文上来讲，这个参数的意义从卷积核上的一个参数到另一个参数需要走过的距离，那当然默认是1了，毕竟不可能两个不同的参数占同一个地方吧（为0）。

- groups = 1:决定了是否采用分组卷积，groups参数可以参考groups参数详解

- bias = True:即是否要添加偏置参数作为可学习参数的一个，默认为True。

- padding_mode = ‘zeros’:即padding的模式，默认采用零填充。

##### Dense block

归一化 BatchNorm2d()

```python
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

![](https://pic.imgdb.cn/item/6097810cd1a9ae528fe78cb1.jpg)

- num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量

- eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5

- momentum：一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）

- affine：当设为true时，会给定可以学习的系数矩阵gamma和beta



二维池化 MaxPool2d()

```python
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

- kernel_size(int or tuple) - max pooling的窗口大小，
- stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
- padding(int or tuple, optional) - 输入的每一条边补充0的层数
- dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
- return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
- ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作



##### trans



#### linear_map

(0): Dropout(p=0.5, inplace=False)
(1): Linear(in_features=15872, out_features=925, bias=True)
(2): BatchNorm1d(925, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(3): ReLU()
(4): Linear(in_features=925, out_features=5, bias=True)
(5): Sigmoid()