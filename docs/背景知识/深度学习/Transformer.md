# Transformer





![image-20210501192731840](https://img.imgdb.cn/item/60938111d1a9ae528ff3d666.png)

## Encoder

![1](https://img.imgdb.cn/item/60938121d1a9ae528ff4a71f.png)

### 1 输入部分

#### Embedding

word2vec

自动初始化

#### **位置编码**

RNN：所有的time steps共享一套参数，只更新一套UWV

<img src="https://img.imgdb.cn/item/60938131d1a9ae528ff572b0.png" alt="image-20210501211804128" style="zoom:67%;" />

RNN的梯度消失：总梯度和被近距离梯度主导，被远距离梯度忽略不计

天然的时序关系非常符合



Transformer是可以并行化的，所有的单词一起处理，不考虑时序关系，增加了速度，忽略了顺序关系，所以需要位置编码

**位置编码公式**

<img src="https://img.imgdb.cn/item/60938141d1a9ae528ff6436e.png" alt="image-20210501212146963" style="zoom:67%;" />

偶数位置使用sin，奇数位置使用cos

![image-20210501212328290](https://img.imgdb.cn/item/60938151d1a9ae528ff7152e.png)



**拓展**

位置嵌入的作用

同一个位置，使用sin和cos可以表示绝对位置，绝对位置向量信息中包含着相对位置向量信息

![image-20210501212501854](https://img.imgdb.cn/item/60938169d1a9ae528ff84021.png)

### 2 注意力机制

#### 基本的注意力机制

<img src="https://img.imgdb.cn/item/6093817ed1a9ae528ff947c8.png" alt="image-20210501212809366" style="zoom:67%;" />



人类在观察一张图片时，肯定会有注意力的差别，如图颜色越深代表对此部分的注意力越大，更加关注该区域

比如说提出问题“**婴儿在干嘛**”，我们需要提取图片中和该问题有关的信息

**注意力机制公式**

![image-20210501213056462](https://img.imgdb.cn/item/6093818ad1a9ae528ff9e3b3.png)

QKV三个矩阵最关键，Q（Query：查询向量）K（Key：键向量）V（Value：值向量）



<img src="https://img.imgdb.cn/item/6093819ad1a9ae528ffaa847.png" alt="image-20210501213423400" style="zoom:67%;" />







**第一步**：Q和K做点乘，得到s

相似度计算：点乘，MLP（网络），cos相似性

点乘：一个向量在另一个向量上投影的长度，是一个标量，可以反映两个向量之间的相似度，点乘结果越大说明距离越近越相似更关注

**第二步**：对s进行类Softmax()归一化，生成a（相似度，相加为一）

**第三步**：对V进行加权平均，相加相乘，得到最终结果Attention value





#### Transformer中的注意力机制

**如何获取QKV**

![image-20210501214412368](https://img.imgdb.cn/item/609381a9d1a9ae528ffb628f.png)

初始的词向量X，与对应的W矩阵（权重矩阵，可以随机初始化迭代更新）相乘，获取QKV三个向量

![image-20210501214650247](https://img.imgdb.cn/item/609381bcd1a9ae528ffc5522.png)

Divide by 8 根号下dk ：由于**Softmax本身梯度很小**，但是S值很大，所以容易发生**梯度消失**，才需要进行此处理

实际代码使用矩阵进行并行操作

<img src="https://img.imgdb.cn/item/609381c9d1a9ae528ffcf0d4.png" alt="image-20210501215031067" style="zoom: 67%;" />

**多头注意力机制**

 ![image-20210501215213404](https://img.imgdb.cn/item/609381d9d1a9ae528ffda64f.png)



相当于将原始数据使用不同的参数在多个空间内进行并行计算，

![image-20210501215304463](https://img.imgdb.cn/item/609381f0d1a9ae528ffec03b.png)



最后将多个head 的多个输出Z共同输出



### 残差和LayNorm

 ![image-20210501215527588](https://img.imgdb.cn/item/60938203d1a9ae528fffa5e5.png)



处理流程：将词向量x进行位置编码，生成新的词向量X，作为输入，经过self-attention处理，输出Z，然后用Z和X进行对位相加作为残差结果，再进行LayerNorm()处理

**残差**

<img src="https://img.imgdb.cn/item/60938215d1a9ae528f008960.png" alt="image-20210501215847277" style="zoom:67%;" />

为什么残差结构会有用？

<img src="https://img.imgdb.cn/item/60938221d1a9ae528f011689.png" alt="image-20210501220435397" style="zoom:67%;" />

连乘应为1+ (のXc/のXb)*(のXb/のXaout)

确保了梯度不会为零，**缓解梯度消失**，可以使网络加深

#### **Layer Normalization**

为什么使用LN而不使用BN？

BatchNormlization：BN在NLP任务中效果很差，BN本身的作用是在存在多个特征的情况下，对多个特征进行归一化使其能迅速收敛，可以解决内部协变量偏移，缓解了梯度饱和问题，加快收敛

BN针对一个batch中的所有样本的所有特征进行Normliaze，每个特征都是对应的，

缺点：

- batch_size较小时，效果差，使用一个batch中所有样本的均值和方差来模拟整体的均值和方差，但是如果样本很小不具有代表性
- 在NLP中，RNN的输入是动态的，与时序有关，如果是一个1*20的向量，就不能用很小的batch来代表整体

LayerNormalization：针对一个样本的所有单词做缩放（均值方差），BN是针对每个单词的不同特征进行缩放，不适合NLP 

https://academic.oup.com/bib/



### 3 Feed Forward前馈神经网络

两层的全连接 ＋ 残差LN



## Decoder

![image-20210501222359782](https://img.imgdb.cn/item/60938230d1a9ae528f01ea46.png)

### 1 Masked Multi-Head Attention

需要对当前单词和之后的单词做Mask（掩盖）

**为什么需要Mask？**

如果不进行mask，在进行训练时，是由所有的单词共同提供的信息进行计算，但是预测时不能看到未来时刻的单词，会产生训练和预测的gap，所以需要掩盖后面的两个单词提供的信息

  ![image-20210501222726779](https://img.imgdb.cn/item/60938240d1a9ae528f02c8cb.png)



### 2 交互层

![image-20210501223053694](https://img.imgdb.cn/item/60938252d1a9ae528f03c6ae.png)

所有的Encoder生成一个输出，然后这一个输出对每个Decoder进行交互

![image-20210501223154302](https://img.imgdb.cn/item/6093825fd1a9ae528f048ad1.png)

Encoder生成KV矩阵，Decoder生成Q矩阵，交互层Q矩阵来自于本身，KV矩阵来自于Encoders，进行多头注意力机制



# Nt_Transformer

![](https://pic.imgdb.cn/item/609787f6d1a9ae528f3bf721.jpg)

## K-mers with 1-D convolutions

生物信息学处理DNA序列常用方法：K-mers，就是将一条完整的DNA序列，转化为长度为k的小片段，和1-D卷积层利用slide window提取信息的方法类似

![](https://pic.imgdb.cn/item/60978bfcd1a9ae528f6de419.png)

S ：sequence vector(length = l) 

K : convolution kernel (size  = 3)  

O: output of vector dot（点乘） 

p：position

stride = 1 步长为1 ，每次移动一个单位

使用一个长为3的一维卷积核对序列S做点乘（卷积），从一条DNA序列中提取3-mers的有效信息





**核苷酸编码**：首先将每个核苷酸转化为embeddings嵌入层（fixed size = dmodel）词向量，对于每条DNA序列产生一个tensor （l，dmodel）

**位置编码**：采用Transformer中对于词向量的编码方式，奇数使用cos，偶数使用sin，生成K作为整体输入，既包含核苷酸编码的信息又包含相对位置信息

![](https://pic.imgdb.cn/item/60978c0dd1a9ae528f6eeac7.png)



![](https://pic.imgdb.cn/item/60978c1cd1a9ae528f6fb17f.png)



**传统k-mers的缺点：**

泛化能力差，只能对生成的k-mers组合进行学习，但是生成的k-mers数量有限，比如数据集中有promoter motif TATAAT 但是如果一个相似的motif TATATT不在数据集中就无法进行学习

**克服方法：**

- 使用convolutions对序列进行特征提取，能够对TATAAT和TATATT的相似度进行量化，从而更好地进行泛化

- 大量的参数：4的k次方


## Transformer encoder

**multi-head self-attention mechanism**

- 多头注意力机制能偶线性地将dmodel维度的K，Q，V投射到低维度中进行表示

- 多头注意力机制能够直接在整条序列上进行操作

- 多头注意力机制允许不同的head学习输入数据不同的隐藏模式，从而提升性能


![](https://pic.imgdb.cn/item/60978c2ad1a9ae528f707aae.png)

**详见Transformer**

- self-attention机制能够计算出每个k-mers其他所有k-mers（包括自身）的联系，这样就能建立全局依赖关系（global dependencies） 

- 与CNN和RNN加强模型对于局部区域的保守性（sparse local connectivity）学习的特点不同，Transformer对于整体的依赖关系有着更好的表现




**position-wise feedforward network** 

![](https://pic.imgdb.cn/item/60978c37d1a9ae528f712e79.png)

- transformer encoder layer：两个transforms中间夹杂着一个ReLU激活函数

- 多个layers堆叠成一个encoder



```python
 model=NucleicTransformer(opts.ntoken, opts.nclass, opts.ninp, opts.nhead, opts.nhid,
                           opts.nlayers, opts.kmer_aggregation, kmers=opts.kmers,
                           dropout=opts.dropout).to(device)
```

| Parameters   | default         | explain                             |
| ------------ | --------------- | ----------------------------------- |
| gpu_id       | 0               | 选择使用的gpu                       |
| path         |                 | 数据集存放路径（DNA seq 和labels）  |
| epochs       | 150             |                                     |
| batch_size   | 24              | 一次训练所抓取的数据样本数量        |
| weight_decay | 0               | 权重衰减（L2正则化，防止过拟合）    |
| ntoken       | 4               | 表示核苷酸所需要的维度（固定值为4） |
| nclass       | 2               | 分类数                              |
| ninp         | 512             |                                     |
| nhead        | 8               | multi-head self-attention           |
| bhid         | 1048            |                                     |
| mlayers      | 6               |                                     |
| save_freq    | 1               | 每训练多少批存储一次结果            |
| dropout      | 0.1             | 丢弃概率防止过拟合                  |
| warmup_steps | 3200            |                                     |
| lr_scale     | 0.1             | 学习率                              |
| nmute        | 18              | 突变数                              |
| kmers        | [2，3，4，5，6] |                                     |
| n_fold       | 5               | 交叉验证                            |
| fold         | 0               | 选择哪一个fold进行训练              |

