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



