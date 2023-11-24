# Vision Transformer



Vision Transformer 超详细解读 (原理分析+代码解读) (一) - 科技猛兽的文章 - 知乎
https://zhuanlan.zhihu.com/p/340149804

![image-20231122112108739](E:\研究生\科研\Transformer\Transformer相关\Vision Transformer.assets\image-20231122112108739.png)

但是这个有点多。。



### 回顾Transformer相关



##### Positional Encoding

- **1.4 Positional Encoding：**

以上是multi-head self-attention的原理，但是还有一个问题是：现在的self-attention中没有位置的信息，一个单词向量的“近在咫尺”位置的单词向量和“远在天涯”位置的单词向量效果是一样的，没有表示位置的信息(No position information in self attention)。所以你输入"A打了B"或者"B打了A"的效果其实是一样的，因为并没有考虑位置的信息。所以在self-attention原来的paper中，作者为了解决这个问题所做的事情是如下图16所示：

![img](https://pic3.zhimg.com/80/v2-b8886621fc841085300f5bb21de26f0e_720w.webp)

![img](https://pic4.zhimg.com/80/v2-7814595d02ef37cb762b3ef998fae267_720w.webp)

![image-20231122105422002](E:\研究生\科研\Transformer\Transformer相关\Vision Transformer.assets\image-20231122105422002.png)



##### 不同Normalization方法的对比



![img](https://pic3.zhimg.com/80/v2-53267aa305030eb71376296a6fd14cde_720w.webp)

图22：不同Normalization方法的对比



其中，Batch Normalization和Layer Normalization的对比可以概括为图22，**Batch Normalization强行让一个batch的数据的某个channel**的$\mu=0$,$\sigma=1$,  而**Layer Normalization让一个数据的所有channel**的$\mu=0$,$\sigma=1$,  



### 3 Transformer+Detection：引入视觉领域的首创DETR

。。。



### Vision Transformer的讲解

狗都能看懂的Vision Transformer的讲解和代码实现   http://t.csdnimg.cn/8sTTi

1、ViT介绍
从深度学习暴发以来，CNN一直是CV领域的主流模型，而且取得了很好的效果，相比之下，基于self-attention结构的Transformer在NLP领域大放异彩。虽然Transformer结构已经成为NLP领域的标准，但在计算机视觉领域的应用还非常有限。

ViT（vision transformer）是Google在2020年提出的直接将Transformer应用在图像分类的模型，通过这篇文章的实验，给出的最佳模型在ImageNet1K上能够达到88.55%的准确率（先在Google自家的JFT数据集上进行了预训练），说明Transformer在CV领域确实是有效的，而且效果还挺惊人。

![result](https://img-blog.csdnimg.cn/76c9042baca14ea3b69536e6eb4246fb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54Ot6KGA5Y6o5biI6ZW_,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

2、模型详解
在讲解ViT原理之前，读者需要有self-attention的相关知识，之前博文有讲过，这里就不复述了。

下图是Vision Transformer的结构，乍一看和self-attention的结构非常像。主要由三个模块组成：

* Linear Projection (Patch + Position 的Embedding层)

* Transformer Encoder（详细结构见图右边）

* MLP Head（分类层）

  

  ![ViT](https://img-blog.csdnimg.cn/21d9e2b2b88b4a64896624ab0e5bebe6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54Ot6KGA5Y6o5biI6ZW_,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

https://blog.csdn.net/weixin_42392454/article/details/122667271

Embedding层详解
从图中看，Transformer Encoder读入的是一小块一小块这样的图片。这样做的原因是：**将一个个小块图像视为token**（token可以理解为NLP中的一个字或词），在Transformer中计算每个token之间的相关性。这一点就和卷积神经网络有很大区别了。以往的CNN，以卷积 + 池化的方式不断下采样，这样理论上模型可以通过加深模型深度，达到增大感受野的目的。不过这样会有两个缺点：

1、实际结果中显示，CNN对边缘的响应很弱。这也非常好理解，越靠边缘的像素，因为被卷积次数少，自然在梯度更新时，贡献更少。
CNN只能和临近像素计算相关性。由于其滑窗卷积的特性，无法对非领域的像素共同计算，例如左上角的像素无法和右下角的像素联合卷积。这就导致了某些空间信息是无法利用的。同时根据MAE论文中所说的，自然图像具有冗余性，即相邻像素点代表的信息是差不多的，所以只计算领域像素无法最大化利用图像特征。
回到ViT中，仅仅把图像拆分成小块（patch）是不够的，Transformer Encoder需要的是一个向量，shape为[num_token, token_dim]。对于图片数据来说，shape为[H,W,C]是不符合要求的，所以就需要转换，要将图片数据通过这个Embedding层转换成token。以ViT-B/16为例：

假设输入图像为224x224x3，一个token原始图像shape为16x16x3，那这样就可以将图像拆分成( 224 / 16 ) ^2 = 196个patch，然后将每个patch线性映射至一维向量中，那么这个一维向量的长度即为16 ∗ 16 ∗ 3 = 768 维。将196个token叠加在一起最后维度就是[196, 768]。

在代码实现中，patch的裁剪是用一个patch_size大小的卷积同时以patch_size的步长进行卷积实现的：

