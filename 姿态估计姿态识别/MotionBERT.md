# MotionBERT

应该是目前最好的

[Walter0807/MotionBERT: [ICCV 2023\] PyTorch Implementation of "MotionBERT: A Unified Perspective on Learning Human Motion Representations" (github.com)](https://github.com/Walter0807/MotionBERT)

[【论文笔记】Unified Human-Centric Model 系列之MotionBERT - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/645577331)

[MotionBert论文解读及详细复现教程-CSDN博客](https://blog.csdn.net/m0_56661101/article/details/131780054)

https://aitechtogether.com/python/137509.html



以下大多来自：

[【论文笔记】Unified Human-Centric Model 系列之MotionBERT - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/645577331)



## Motivation

从论文的标题就可以看出，这篇文章希望能够针对Human-centric任务构建一种统一的预训练范式来学习通用的人体运动表征，从而帮助下游任务（例如人体姿态估计、动作识别、网格估计等）的完成。

统一预训练范式的难点在于，不同下游任务的数据集中，所包含的标签类型是不同的，**如何利用这些标签差异极大的数据集**构建统一的预训练任务，这将会是一个难点。同时，如果能够构建出来这样一种统一的任务，那也就相当于**运用了比原来任何一种下游任务更多的数据**，从而帮助性能的提升。

（统一人体运动分析的子任务的一个主要障碍是数据资源的异质性。动作捕捉系统[34,63]提供了带有标记和传感器的高保真3D运动，但捕获的视频的外观通常局限于简单的室内场景。动作识别数据集提供了动作语义的注释，但它们要么不包含人体姿势标签[15,82]，要么特征为日常活动的有限动作[52,53,81]。野外（wild，有遮挡等，可以看in the wild这篇人体姿态估计文章）人类视频可以通过不同的外观和动作大规模访问，但获得精确的2D姿势注释需要付出巨大的努力，而获得正确标记的数据(GT) 3D关节位置几乎是不可能的。因此，现有的研究基本上集中在一个特定的子任务使用孤立类型的运动数据。）

本文的解决方案是，在预训练阶段进行**从noisy 2D observations恢复3D motion**的任务，从而帮助模型理解和学习人体的潜在三维结构。至于为什么要构建这样一个预训练任务，Method的Unified Pretraining部分将进行解释。



## 引言

总的来说，本工作的贡献是三方面的：

1）我们提供了通过学习人类运动表示解决各种面向人的视频任务的新视角。

2）我们提出了一种预训练方法，利用大规模但**异质**的人体运动资源，学习通用的人体运动表示。我们的方法可以同时利用3D mocap数据的精度和户外RGB视频的多样性。

3）我们设计了一个双流Transformer网络，配备了串联的空时自注意力块，可以作为人体运动建模的通用骨架。实验表明，上述设计使得可以将通用的人体运动表示转移到多个下游任务，超越了任务特定的最先进的方法。

原文链接：https://blog.csdn.net/m0_56661101/article/details/131780054

我们证明了预训练的运动表示可以很好地推广到动作识别任务，预训练-微调框架是一个适合的解决一次性挑战的方案。

## 相关工作

本节回顾了四个主要领域的研究：人体运动表示学习、3D人体姿态估计、基于骨架的行为识别和人体网格恢复。作者提到了这些领域内的关键技术和方法，包括隐马尔科夫模型、图形模型、深度度量学习、CNN、Transformer等。此外，作者强调了本研究工作的创新之处，即提出了一个统一的预训练-微调框架，可以整合各种异质数据资源，并在各种下游任务中表现出通用性。


## Method

本文的方法框架如下图所示。首先进行从2D估计3D的预训练任务，训练Motion Encoder来学习一个好的人体运动表征。然后在预训练结束后，利用该Motion Encoder（DSTformer），在不同的下游任务上利用特定数据集进行finetune。

输入是二维骨骼姿态

![img](https://pic4.zhimg.com/80/v2-369974bb5da8748ec188e7b4336452ef_720w.webp)

​																			MotionBERT方法框架图

### Network Architecture

​	由于Motion Encoder需要能够从noisy 2D skeletons恢复出3D motion，它应当具备充分的捕捉人体依赖关系和运动特点的能力。因此，文章对于Motion Encoder的网络结构进行了设计，提出了一种Dual-stream Spatio-temporal Transformer (DSTformer) 【m双流时空Transformer 】来捕捉运动序列中长距离的时空依赖关系，网络结构如下图所示。

![img](https://pic2.zhimg.com/80/v2-17efb158c4318732d7e9402b184a2e7d_720w.webp)

​									图1-DSTformer网络结构图



​	上图1显示了2D到3D转换的网络架构。给定一个**输入2D骨骼序列**，我们首先将其投影到一个**高维特征**，然后添加可学习的空间位置编码和时间位置编码。然后我们使用序列到序列模型**DSTformer**计算特征，其中N是网络深度。我们使用带有**tanh激活函数**的**线性层**来计算动作表示。最后，我们对动作表示进行线性变换来估计3D动作。这里，T表示序列长度，J表示身体关节的数量。Cin、Cf、Ce、Cout分别表示输入、特征、嵌入和输出的通道数量。我们首先介绍DSTformer的基本构建块，即带有**多头自注意力（MHSA）的空间和时间块**，然后解释**DSTformer架构设计**。



模型的输入为  $x ∈ \R^{T*J*C_{in}}$，其中 $C_{in}=2 $，因为输入特征为2D坐标。在送入Transformer的网络结构之前，需要先进行三个步骤——

1. 先将输入坐标投影至特征空间中：将时间维度与batch size合并，然后通过一层全连接网络将 $C_{in}$ 维的特征投影为 $C_f$维（代码默认值为256，论文中设置为512），见代码中的self.joints_embed网络层。（将其投影到高维特征$F_{0} ∈ \R^{T*J*C_{f}}$上）

2. 加入时间和空间维度的位置编码：

   ​	对于空间维度的位置编码，加入形状为$ 1×J×C_{f}$ 的可学习参数（加入可学习的空间位置编码![image-20231101122008483](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101122008483.png))，不同时刻是统一的值，见代码中的self.temp_embed参数；

   ​	对于时间维度的位置编码，加入形状为 $ 1×T×C_{f}$ 的可学习参数（加入可学习的空间位置编码)![image-20231101122106902](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101122106902.png)![image-20231101122101675](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101122101675.png)，不同关键点是统一的值，见代码中的self.pos_embed参数。这里需要注意中间变量的各种reshape操作。

3. 随机地丢弃一些关键点的数据：通过dropout实现，见代码中的self.pos_drop网络层。

   

【T表示序列长度，J表示身体关节的数量。Cin、Cf、Ce、Cout分别为输入通道数量、特征通道数量、嵌入通道数量、输出通道数量，N网络深度。】



进行完这两个步骤之后，可以将其送入N层级联的DSTformer网络中，接下来将对一层DSTformer的结构进行介绍。

然后输入DSTformer来计算![image-20231101144524705](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101144524705.png)

我们对$F^N$应用具有tanh激活的线性层来计算运动表示![image-20231101144915600](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101144915600.png)

最后，我们对E应用线性变换来估计3D运动的![image-20231101145121906](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20231101145121906.png)



DSTformer中包含两种结构的子网络：Spatial MHSA和Temporal MHSA。它们也很好理解，分别是在空间和时间维度进行attention计算，而在另一个维度的计算是并行的，可以通过“DSTformer网络结构图”的右下角子图进行理解。

**空间块** Spatial MHSA
空间多头自注意力（S-MHSA）旨在模拟**同一时间步内关节之间**的关系。我们使用自注意力来获取每个head的查询、键和值。

**时间块** Temporal MHSA
时间多头自注意力（T-MHSA）旨在模拟**一个身体关节在时间步中**的关系。它的计算过程与S-MHSA相似，只是MHSA应用于每个关节的时间特征，并在空间维度上并行化。

**双流空间时间转换器**DSTformer
	给定捕获帧内和帧间身体关节交互的空间和时间MHSA，我们组装基本构建块以在流中融合空间和时间信息。我们设计了一个双流架构，基于以下假设：1）两个流应该能够模拟全面的时空上下文。2）每个流应该专门处理不同的时空方面。3）两个流应该被融合在一起，融合权重应根据输入的时空特性动态平衡。因此，我们以不同的顺序堆叠空间和时间MHSA块，形成两个并行的计算分支。两个分支的输出特征使用由注意力回归器预测的自适应权重进行融合。然后重复N次双流融合模块。



文章认为，先做Spatial MHSA再做Temporal MHSA提取的特征，和先做Temporal MHSA再做Spatial MHSA提取的特征是不同的。因此文章中设计的DSTformer块中有两个分支，两个分支中Spatial和Temporal MHSA的顺序是相反的，再提取完两种特征后再进行合并。

![img](https://pic2.zhimg.com/80/v2-17efb158c4318732d7e9402b184a2e7d_720w.webp)

​									图1-DSTformer网络结构图																			

（再放一次这个图）

最简单的合并方式就是计算两种特征的平均值，然而这篇工作希望模型能够根据输入的特性动态地调整两种特征的比例，因此引入了$\alpha _{ST}$ ,$\alpha _{TS}$  参数。首先将两个分支的输出（记为 y1,y2 ）在特征维度进行concat，然后通过一个投影矩阵将其投影为维度为2的向量并计算softmax结果，记为 $\alpha _{ST}$ ,$\alpha _{TS}$ ，二者的和为1，就可以用来调整 y1,y2 的比重。这样得到的结果作为一个DSTformer block的最终输出。

经过几层级联block之后，再经过一层全连接和Tanh，得到的就是motion embedding。如果需要进行3D motion的恢复，则再加入head（一层全连接层）即可。

以下是官方代码中摘录出来的DSTformer的Pytorch实现，可以帮助理解。

```python
class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4, 
                 num_joints=17, maxlen=243, 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                st_mode="stage_st")
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()            
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, dim_out, global_pool=''):
        self.dim_out = dim_out
        self.head = nn.Linear(self.dim_feat, dim_out) if dim_out > 0 else nn.Identity()

    def forward(self, x, return_rep=False):   
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.temp_embed[:,:F,:,:]
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)
        alphas = []
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
            else:
                x = (x_st + x_ts)*0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)         # [B, F, J, dim_feat]
        if return_rep:
            return x
        x = self.head(x)
        return x

    def get_representation(self, x):
        return self.forward(x, return_rep=True)
```



### Unified Pretraining  统一预训练

在统一的预训练任务中，我们能够利用的有两种类型的数据。

第一种是大量存在的3D mocap data：将3D动捕数据随机地投影得到一些2D骨架数据，然后在2D骨架数据中随机mask掉一些关键点并加入噪声，从而模拟真实数据中的遮挡、检测失败和误差现象。对于这种数据，可以加入位置误差损失和速度误差损失用于预训练。

第二种是一些没有3D GT的2D数据，比如in-the-wild(野外视频)的人体姿态数据集。在这种情况下，可以通过手动标注或者2D estimation的方式获取一些2D骨架数据，然后通过2D重投影误差进行训练。

总的损失函数表达式如下：

![img](https://pic4.zhimg.com/80/v2-c2e186c3adb6d3b01ddb442f0ac3a997_720w.webp)

统一预训练任务损失函数

*从这里我们也能够看出，之所以文章会采取2D-3D lifting作为预训练任务，一大原因就是对于各种人体相关的数据集，都能够通过手动标注或2D估计来获得2D骨架数据，保证了对于各种heterogenous数据集的利用率。*

### Task-Specific Finetuning

对于不同的下游任务，本文采取了minimalist design原则，即设计尽量简单的head。

1. 3D姿态估计任务：只需要reuse预训练模型即可。
2. 基于骨架的动作识别任务：global average pooling+MLP with one hidden layer.
3. 人体网格恢复：利用每一帧的motion embedding估计pose参数，而由于shape参数在一段视频中具有一致性，因此在temporal维度对所有motion embedding计算平均，然后回归出shape参数。

## Experiments

预训练过程中使用的数据集：

- Human3.6M (train subset)
- AMASS (一个整合了几乎所有marker-based Mocap数据集的数据集)
- PoseTrack (in-the-wild)
- InstaVariety (in-the-wild)

实验结果如下。结论是本文提出的预训练+微调方式的统一模型能够在三种任务上达到SOTA效果。这个结果也很合理，毕竟对于每一个下游任务而言，都相当于利用了更多数据。同时，即使不进行预训练，但从网络结构的角度来看，DSTformer也很能打。

![img](https://pic3.zhimg.com/80/v2-136e2ac0012792e0b13f059fe681df7a_720w.webp)

![img](https://pic1.zhimg.com/80/v2-0b0f99b202ef31fcbf6c142d26c2c7d0_720w.webp)

![img](https://pic4.zhimg.com/80/v2-33de59de82490ef0611fd905c2511363_720w.webp)

## Ablation Study

1. 训练方式的影响

2. 1. Finetune vs. Scratch: Finetune相比于Training from scratch能够更快收敛且效果更好，见Figure 3。
   2. Partial Finetuning: frozen motion encoder只微调header，相比于随机初始化的motion encoder效果显著更好，见Table 4。
   3. Pretraining Strategies: 简单的2D-3D lifting的预训练对于每个任务都有帮助，并且对2D数据做corruption也很有帮助，见Table 5。

3. 不同Backbone

4. 1. 在TCN和PoseFormer上，相同的预训练任务也很有帮助，见Table 6。
   2. 对比了各种model architecture variants，发现dual-stream和adaptive fusion的方式是有帮助的，见Table 7。

![img](https://pic1.zhimg.com/80/v2-852c0dc48a2e7f2357eb11ebd502e0f0_720w.webp)

![img](https://pic2.zhimg.com/80/v2-f8df8658c5014e271723c10d9f1090ed_720w.webp)

![img](https://pic3.zhimg.com/80/v2-dc861d5f9c575850ab50393624c85836_720w.webp)

![img](https://pic4.zhimg.com/80/v2-becb0f92fae6d341afa2f1b29167f5df_720w.webp)

![img](https://pic4.zhimg.com/80/v2-b60dcaa30a42b53f2e43fd4a40931867_720w.webp)

## Conclusion

这篇文章从预训练任务的定义的角度为human-centric的统一模型提供了一个可行的思路。然而直观感受上，这种预模型只适用于以2D skeleton输入的human-centric任务，这也是这篇工作的一个局限性吧。