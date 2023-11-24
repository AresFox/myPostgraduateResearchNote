# NeRF

## NeRF

理论与论文学习



参考：

看完了，但是好像没怎么讲原理，不是很清晰：

【NeRF系列公开课01 | 基于NeRF的三维内容生成】 https://www.bilibili.com/video/BV1d34y1n7fn/?share_source=copy_web&vd_source=067de257d5f13e60e5b36da1a0ec151e

以下PPT来自：

【十分钟带你快速入门NeRF原理】 https://www.bilibili.com/video/BV1o34y1P7Md/?share_source=copy_web&vd_source=067de257d5f13e60e5b36da1a0ec151e

[NeRF简介_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Qd4y1r7ZX/?spm_id_from=333.337.search-card.all.click&vd_source=f2def4aba42c7ed69fc648e1a2029c7b)



### 1.1 背景

视角合成方法通常使用一个中间3D场景表征作为中介来生成高质量的虚拟视角。根据表示形式，3D场景表征可以分为“显式”和“隐式”表示。显式表示（explicit representation），包括Mesh，Point Cloud，Voxel，Volume等。显式表示的优点是能够对场景进行显式建模，从而合成照片级的虚拟视角。缺点是这种离散表示因为不够精细化会造成重叠等伪影，而且最重要的，它们对内存的消耗限制了高分辨率场景的应用。

![img](https://pic2.zhimg.com/80/v2-f77c747f7c6e64003069b8b8c2c67049_720w.webp)

隐式表示（implicit representation），通常用一个函数来描述场景几何。隐式表示使用一个MLP模拟该函数，输入3D空间坐标，输出对应的几何信息。隐式表示的好处是它一种连续的表示，能够适用于大分辨率场景，而且通常不需要3D信号进行监督。在NeRF之前，它的缺点在于无法生成照片级的虚拟视角，如occupancy field、signed distance function（SDF）。

![img](https://pic2.zhimg.com/80/v2-f67b817fbced71336c465f196bbc20f9_720w.webp)

我们需要理解的是，无论是显式表示还是隐式表示，都是对3D场景进行表征。这种表征并不是凭空臆测或者天马行空的，而是根据现实中数据格式进行发展。例如现实中的3D数据主要有面数据、点数据、体数据，所以对应催生了一些Mesh、Point Cloud、Volume等中间表示。隐式表示则是借鉴了图形学中的一些表示形式，例如signed distance function。



![image-20231021200117238](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021200117238.png)



### 1.2 NeRF方法

NeRF首次利用隐式表示实现了照片级的视角合成效果，与之前方法不同的是，它选择了Volume作为中间表示，尝试重建一个隐式的Volume。NeRF的主要贡献：

•提出了一种5D neural radiance field 的方法来实现复杂场景的隐式表示。

•基于经典的Volume rendering提出了一种可微渲染的流程，包括一个层级的采样策略

•提出了一种位置编码（positional encoding）将5D坐标映射到高维空间。

![img](https://pic4.zhimg.com/80/v2-0796367dc33ac526ff5feaa54f66c3d7_720w.webp)

NeRF基于MLP合成Volume的隐式表示，并通过Volume rendering渲染到2D图片计算损失

在理解NeRF之前，我们首先需要厘清两个概念，**神经场（Neural field）**与**体渲染（Volume rendering）**。

![img](https://album.biliimg.com/bfs/new_dyn/eafe3481a77e30ccc96c8a2c19690c4c3494380627299296.png@!web-comment-note.webp)

#### 总体流程：

**NeRF，一种用于三维场景表达的隐函数**

那么NeRF到底是什么？NeRF，全称为neural radiance field，翻译为神经辐射场。这里的辐射场，并不是新鲜玩意。任何结构，如果可以用空间位置信息查询到对应的一组 颜色+密度 信息，那么它就可以是这里所说的辐射场。对于有图形学背景的人来说，3D Texture就是一个很好的例子。只不过，NeRF使用了神经网络来表示一个辐射场。

话不多说，我们先从整体上直观的来认识NeRF。我做了一个图详细的描述了一下NeRF的整个pipeline：

![img](https://pic2.zhimg.com/80/v2-6bf894d1fd88981551751cf4337b931d_720w.webp)

​																						NeRF pipeline

整体看下来NeRF的整个想法不算复杂：在forward用体渲染采样神经网络的辐射场，然后把结果和ground truth做对比，反向优化网络参数，不断迭代使得辐射场接近真实值。这么看来，入门最大的障碍，大概就是懂机器学习的不懂图形学，搞图形学的不懂机器学习吧（

于是为了方便理解，在上图中我把图形学和机器学习的部分，分别用不同颜色来表示，以供大家对症下药。绿色的部分，是属于图形学的组件，基本上只涉及体渲染。蓝色的部分，是涉及到机器学习的组件，也只是最基本的MLP而已。

之所以用神经网络来表示辐射场，一是比起类似3D Texture这种volume的表示，神经网络没有分辨率的限制，也不会随着表达精度的上升而光速增长体积。另一个原因，就是神经网络的参数是可学习的，神经网络中的计算/查询是可微的（后面讲到ngp，我们就会发现，只要满足这两个条件，三维表征也可以是其他形式的）。因此，NeRF才能通过不断更新参数，来使得表征更加接近真实值，完成高质量的新视角合成任务。

（--https://zhuanlan.zhihu.com/p/631284285）

### 1.3 神经场

神经场在Neural Fields in Visual Computing and Beyond[1]这篇文章中得到了非常详尽的阐述，简单来说：

场（field）是为所有（连续）空间和/或时间坐标定义的量（标量），如电磁场，重力场等。因此当我们在讨论场时，我们在讨论一个连续的概念，而且他是将一个高维的向量映射到一个标量。

神经场表示用神经网络进行全部或者部分参数化的场。

在视觉领域，我们可以认为神经场就是**以空间坐标或者其他维度（时间、相机位姿等）作为输入，通过一个MLP网络模拟目标函数，生成一个目标标量（颜色、深度等）的过程**。

​																																		



### 1.4 体渲染

简单来说，体渲染就是模拟光线穿过一系列粒子，发生一些反射、散射、吸收，最后到达人眼的过程。没错，其实不管是你硬表面软表面，NeRF都是是把各种物体都当做不同密度的粒子、烟雾一类的东西去建模的。

![image-20231021201104105](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021201104105.png)





关于体渲染的理解推荐大家阅读State of the art on neural rendering[2]。体渲染简而言之是从体数据渲染得到2D图片的过程。

现实生活中，有一些领域采集的3D数据是以体数据格式存储的，例如医疗中的CT和MRI、地质信息、气象信息等，这些数据需要渲染到2D图像才能够被人类理解。除此之外体数据建模（Volume）相比于传统的Mesh、Point，更加适合模拟光照、烟雾、火焰等非刚体，因此也在图形学中有很多应用。

体数据的渲染主要是指通过**追踪光线进入场景并对光线长度进行某种积分来生成图像或视频，**具体实现的方法包括：Ray Casting，Ray Marching，Ray Tracing。

基于体渲染的研究在NeRF之前有很多，因为体渲染是一种可微渲染，非常适合与基于统计的深度学习相结合。目前可微渲染领域也有了一些研究，是未来计算机视觉和计算图形学结合的一个重要方向。



具体：

![image-20231021214649619](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021214649619.png)

![image-20231021214607686](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021214607686.png)

不一定是完全均匀的 而是有随机间隔的(t1 t2 t3)



其实就是在光线上取一段路径做积分。对计算机来说，则是在光线上产生一些step，t，得到一系列采样点的位置r(t)和观察方向d，然后把这些点作为神经网络的输入，得到对应的密度和颜色。最后再根据距离相机远近计算一个衰减权重T(t)，把这些采样点做一个加权和就ok了。这与我们在pipeline图中的描述也是一致的。

### 1.4++ 体渲染数学公式推导



![img](https://pic4.zhimg.com/80/v2-9d819df7e99895b0eac179ea4c425f2b_720w.webp)

https://blog.csdn.net/YuhsiHu/article/details/124318473



![image-20231021215207082](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021215207082.png)

![image-20231022112031083](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022112031083.png)

这里的 $ \tau(t)$ 相当于 $ nerf $ 的 $\sigma$ 





![image-20231022112906061](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022112906061.png)

表示s位置光发出的强度



##### 吸收发射模型

实际上，空间中的粒子会遮挡入射光，并添加自己的光。 因此，一个现实的微分方程应该包括源项 g ( s ) 和衰减项 I ( s )  。我们只需要将前两种模型进行简单的数值加和（微分方程右侧加在一起），就可以得到这个模型的传递方程:

![image-20231022113622222](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022113622222.png)



![image-20231022113931849](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022113931849.png)

##### 转变为NeRF当中的形式

上面的式子和NeRF原文中仍然有差别，这是因为NeRF和Max的文章中使用的坐标不同。Max文章中的坐标是让相机在D坐标，而无穷远点在0坐标，这样前面的推导就是正确的。但是NeRF中的坐标，是让相机在坐标原点，无穷远坐标就是无穷远，这样就可以得到：

![image-20231022114504582](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022114504582.png)

![image-20231022114513900](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022114513900.png)

论文里给出的黎曼和的形式如下：

![img](https://pic2.zhimg.com/80/v2-f58e92c5f909e8a130037edc5c55182d_720w.webp)

​													体渲染公式的黎曼和形式



总的来说，体渲染的过程并不复杂，重要的是其积分形式带来的可微性质，使得基于梯度的机器学习算法，能够迭代更新可学习参数，从而解决场景优化问题。具体而言，公式中Ti的黎曼和形式是可微的，Ci是密度![image-20231022155254629](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022155254629.png) 和颜色 ![image-20231022155301243](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022155301243.png)相乘得到的，二者都是通过神经网络获取的，过程也是可微的。

c 本身的颜色值

![image-20231022121245026](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022121245026.png)





![img](https://album.biliimg.com/bfs/new_dyn/cb3a04395e92d9c1744451266315836a3494380627299296.png@!web-comment-note.webp)



### 1.5 位置编码

（提升神经辐射场的方法之一）

![image-20231021200710295](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231021200710295.png)

因为神经网络倾向于学习低频信息，所以我们给输入映射到高频空间，这样能更容易学习高频信息

上图可以看出 没加入位置编码，细节就会比较缺失



**chatgpt：**

为什么这个位置编码能让他高频信息更容易获得呢



位置编码中的正弦（sin）和余弦（cos）函数之所以能够帮助模型更容易获取高频信息，是因为它们引入了不同频率的周期性振荡模式，这有助于模型理解和捕捉场景中的细微变化和细节。以下是解释这一点的原因：

1. **周期性模式**：正弦和余弦函数都是周期性函数。通过在位置编码中使用这些函数，不同位置的编码将在不同频率的周期性模式下振荡。这意味着编码的不同维度将对场景中的不同频率的特征具有更敏感的响应。**高频信息**通常以**快速振荡**的形式存在，因此**正弦和余弦**函数有助于模型更好地捕捉这些高频特征。
2. **多尺度表示**：正弦和余弦函数具有不同的频率。通过在位置编码中使用不同频率的正弦和余弦函数，模型可以同时考虑多个尺度的特征。这意味着模型可以同时捕捉高频和低频信息，从而更全面地理解场景的细节。
3. **模型可学习性**：正弦和余弦函数的频率是可学习的参数。在训练中，神经网络可以自动调整这些频率以适应不同的场景和任务。这意味着模型可以根据需要调整编码的频率，以更好地捕捉高频信息。

综合来说，位置编码中的正弦和余弦函数提供了一种有效的机制，使神经网络更容易获取高频信息，从而提高了模型对细节的感知和渲染的质量。这对于处理复杂的三维场景重建和渲染任务非常有帮助。



关于使用位置编码的理由，也是基于过去大家对MLP的观察。人们发现MLP的参数在学习的时候，会倾向于学习低频信息，所以直接对网络使用5D输入，效果会很糊。使用位置编码后，原本在低维接近的点，在高维会离得很远。换句话说，MLP在高维学到的低频信息，往往对低维来说不是低频的。



### 1.6 分层采样

![image-20231022142202488](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022142202488.png)

（提升神经辐射场的方法之一）

由于如果是俊宇64个采样点这样的话，很可能会有很多采样点在空气中，浪费了。因此采用2层网络。

粗网络 均匀采样64个点

精细网络  使用粗网络的输出结果，当做权重采样

![image-20231022142443808](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022142443808.png)

先是对cross网络使用较少的采样点，粗采样一遍。之后按照粗采样到的位置的密度值，在相应区间生成不同数量的采样点。原则上重点照顾拥有更高密度的位置，密度越高，使用越多的采样点进行精细采样。



![image-20231022142632626](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022142632626.png)

（很像包围盒思想）





### 1.7 具体实现细节：



![img](https://album.biliimg.com/bfs/new_dyn/646a059eab30bfb369cb8979d43bbf493494380627299296.png@!web-comment-note.webp)

损失函数

![image-20231022143736477](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022143736477.png)



均方误差MSE 计算渲染出的点位与真实gt（ground truth）的差值



![image-20231022144211614](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022144211614.png)



### 1.8 评价指标

![image-20231022144617899](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022144617899.png)



### 1.9 劣势

![image-20231022145124384](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022145124384.png)

### 1.10 NeRF总结

从我个人的观点来看，我举得NeRF最大的贡献，是**实现了神经场（Neural Field）与图形学组件Volume rendering的有效结合**。NeRF本身的方法实现是非常简洁的，简洁而有效，说明这种组合是合理的。这也启发我们探索更多视觉和图形学的交叉领域，事实上这一方面的探索还比较少。

另一方面，NeRF的简洁也说明了他本身存在很多问题。接下来我们将结合NeRF的问题以及NeRF的应用，概述一下NeRF的发展。



![image-20231022145645810](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022145645810.png)



## Instant-NGP

https://zhuanlan.zhihu.com/p/631284285

**用参数化显式表达加速pipeline**

先直观的感受一下Instant-NGP的pipeline：

<img src="https://pic1.zhimg.com/80/v2-4e7016eb074496cf0235a7ff685d8c24_720w.webp" alt="img" style="zoom:200%;" />

​								Instant-NGP pipeline

流程一眼看上去和NeRF差不多，但是密密麻麻的小字直觉上就让人觉得很不详（

这里为先直接给出Instant-NGP与NeRF的异同：

1. 同样基于体渲染
2. 不同于NeRF的MLP，NGP使用稀疏的参数化的voxel grid作为场景表达；
3. 基于梯度，同时优化场景和MLP（其中一个MLP用作decoder）。

可以看出，大的框架还是一样的，最重要的不同，是NGP选取了参数化的voxel grid作为场景表达。通过学习，让voxel中保存的参数成为场景密度的形状。

MLP最大的问题就是慢。为了能高质量重建场景，往往需要一个比较大的网络，每个采样点过一遍网络就会耗费大量时间。而在grid内插值就快的多。但是grid要表达高精度的场景，就需要高密度的voxel，会造成极高的内存占用。考虑到场景中有很多地方是空白的，所以NVIDIA就提出了一种稀疏的结构来表达场景。



### Hashing of voxel grid

先简单介绍一下voxel grid和它的hash table形式。voxel，可以认为是空间中的一个**小立方体**。voxel grid，就可以想象成空间中一系列共用顶点的小正方体，这些小正方体的每个顶点，就是一个voxel vertex。这个voxel可以保存任何信息，一般我们拿来保存一些只和**位置**有关的信息，比如，**密度**和**漫反射颜色**。我们可以用一个三维数组来表示一个voxel grid：[N, N, N]，同样也可以用一个N^3的一维表来表示：

![img](https://pic2.zhimg.com/80/v2-63b7cc9af4ad2978d6324158c020705d_720w.webp)

​								grid和它的table

如上图左边就是一个N为2的voxel grid。如果我们按顺序给每一个顶点一个index，然后把其中的密度值放入一个一维表对应的index中，就能得到一个voxel grid的table形式。如果顶点所对应的table中的index是通过hash算法得到的，那这个一维表就是一个hash grid。具体用到的 spatial hash function，原论文和知乎上有很多讲解，这里不做说明。

一般情况下，我们需要查询的点，并不会和grid的顶点重合。所以查询的时候，需要**先找到采样点在哪个小立方体内**，然后再对立方体的**八个顶点做插值**来求解。这一过程在NGP pipeline图中也有说明。



### Multiresolution hash encoding [多分辨率哈希编码]

​	如果我们要表达一个精细的场景，使用hash grid的时候，哈希表的大小往往是小于顶点数量的。这样就可能造成哈希冲突，表中的密度不知道应该表达哪个顶点的才好。为了避免这种不好的事情发生，NGP提出了multiresolution hash encoding的方法，翻译为多分辨率哈希编码。

​	具体而言，我们把相同空间，用**不同大小的grid表达**，比如从16x16x16 到 512x512x512的M个分辨率的grid。然后我们把hash table的大小T设为固定值，比如64x64x64。这样一来，我们就得到了M个大小为T的hash table，这些表放在一起就是原文中的multiple separate hash tables。当grid小于64的时候，总是不会发生冲突的。超过64的grid，冲突就冲突了，反正最后的密度，是以不同权重混合每一个分辨率的grid的密度得到的（`m:w1*md1 + w2*md2 + w3*md3 `    md:密度）。至于权重谁高谁低，则是MLP通过loss学到的。

​	从输入输出的角度而言，Multiresolution hash encoding最后会把一个(x,y,z)的位置信息，转变为该位置上的multiple separate hash tables中的密度信息。输出一个LxF维的向量，作为网络的输入。（这是一个针对输入的再赋权）

**L是总共有多少种不同的分辨率**，（L对应的是特定分辨率体素对应的编码层，按照之前表格给出的超参数，层数被设置为16，即体素分辨率变化分为16级，）

**F是每个顶点保存的密度特征编码的维度**。

之后**MLP**为该采样点每个分辨率上的密度**分配权重**，并做特征decode，得到最终的密度值。

​	这里放个原文中的图表给各位做参考：

![img](https://pic2.zhimg.com/80/v2-c1acfc5b41bc9eed212af063e1cc9a8d_720w.webp)

​												Instant-NGP原文中的算法流程图



-哈希：根据内容就能直接定位数据位置

 Instant-ngp不仅拥有可训练的权重值Φ，同时还拥有可训练的编码权重𝜃。



## Instant-NGP中的MLP

NGP中同样有两个MLP，一个是类似NeRF的外观网络，另一个则大为不同。NGP中和密度相关的MLP，功能是把multiresolution hash encoding输出的，采样点所对应的多个分辨率上的密度特征编码，按照不同的权重混合，同时做decode得到真正的密度值。相比起NeRF中MLP 60维的输入，NGP原文的MLP只有32维输入，同时网络的层数也少很多，是一个小型的MLP，跑起来要比NeRF的网络快很多。

## Instant-NGP小结

可以看出，NGP改进了图形学中已有的结构并应用到体渲染框架中，来加速从二维到三维的重建。并且即使和NeRF一样使用了MLP，功能目的却完全不同。





## 实践：

1、

https://www.bilibili.com/video/BV1q84y1U7Qf/?spm_id_from=333.999.0.0&vd_source=f2def4aba42c7ed69fc648e1a2029c7b



2、

[AI（Nerf）扫描3d场景并导入虚幻5！Luma AI教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bL41187M3/?spm_id_from=333.788&vd_source=f2def4aba42c7ed69fc648e1a2029c7b)

使用Luma AI(https://lumalabs.ai/)将扫描现实场景并导入虚幻，使用了Nerf技术。导入虚幻插件https://docs.lumalabs.ai/9DdnisfQaLN1sn。导入到UE5后不仅可以随意改变摄像机，还可以随意打光或者把场景里放入其他物体。

![image-20231022120715072](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022120715072.png)

![image-20231022120732394](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231022120732394.png)





原文

https://github.com/NVlabs/instant-ngp



windows

https://github.com/bycloudai/instant-ngp-Windows

https://www.youtube.com/watch?v=kq9xlvz73Rg

##### LINUX

https://www.jianshu.com/p/02c3d3cce99b