# stable diffuse 换装 

官方

https://stablediffusionweb.com/#ai-image-generator



博客等

给人生成/换衣服

https://zhuanlan.zhihu.com/p/630899907



https://www.youtube.com/watch?v=kE5f7r5j3wg



【Stable Diffusion 换衣服教程】 https://www.bilibili.com/video/BV1HV4y1A7yJ/?share_source=copy_web&vd_source=067de257d5f13e60e5b36da1a0ec151e

https://post.smzdm.com/p/a5ol2kdx/

给衣服生成人

【解密AI新思路：给生成人物穿上指定淘宝衣服！】 https://www.bilibili.com/video/BV1p14y1R71y/?share_source=copy_web&vd_source=067de257d5f13e60e5b36da1a0ec151e



### 论文

学长提供：

Text2Human 

https://arxiv.org/abs/2205.15996

文章：

https://cloud.tencent.com/developer/article/2092660

**Text2Human** **仅通过提供关于性别和衣服的文字描述来生成一个人的图像。**

> 代码：https://github.com/yumingj/Text2Human 
>
> 论文：https://arxiv.org/pdf/2205.15996.pdf 
>
> 标题：Text2Human: Text-Driven Controllable Human Image Generation





https://arxiv.org/abs/2211.06235

gpt：

这篇论文的主要工作是介绍了一种名为"HumanDiffusion"的方法，用于实现基于文本的人物图像生成。这是跨模态图像生成中的一个新兴且具有挑战性的任务。通过控制生成人物图像，可以应用于数字人际互动和虚拟试穿等广泛领域。然而，先前的方法大多采用单一模态信息作为先验条件（例如，姿势引导的人物图像生成），或者使用预设的文字来生成人物图像。本文提出了一种更用户友好的方法，即使用自由词汇构成的句子以及可编辑的语义姿势地图来描述人物外貌。

为了实现这一目标，论文提出了一个粗粒度到细粒度的对齐扩散框架"HumanDiffusion"，用于基于文本生成人物图像。具体来说，论文提出了两个协作模块，分别是"Stylized Memory Retrieval (SMR)" 模块，用于数据处理中的细粒特征提取，以及"Multi-scale Cross-modality Alignment (MCA)" 模块，用于扩散中的粗粒度到细粒度特征对齐。这两个模块确保了文本和图像的对齐质量，从图像级别到特征级别，从低分辨率到高分辨率。

通过这种方法，"HumanDiffusion" 实现了具有期望语义姿势的开放词汇的人物图像生成。作者在DeepFashion数据集上进行了广泛的实验证明了他们的方法相对于先前的方法的卓越性能。此外，对于具有各种细节和不常见姿势的复杂人物图像，也可以获得更好的结果。





### Control Net

代码地址：[github.com/lllyasviel/…](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Flllyasviel%2FControlNet)

论文地址：[arxiv.org/abs/2302.05…](https://link.juejin.cn/?target=https%3A%2F%2Farxiv.org%2Fabs%2F2302.05543v1)



分析：

https://juejin.cn/post/7210369671656505399


#### 论文阅读

##### **Related Work**

### **2.2. 图像扩散Image Diffusion**

**图像扩散模型**最早由Sohl- Dickstein等人[80]引入，最近被应用于图像生成[17, 42]。[潜在扩散模型](https://www.zhihu.com/search?q=潜在扩散模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3273259612})（LDM）[71]在潜在图像空间[19]中执行扩散步骤，这降低了计算成本。文本到图像扩散模型通过预先训练的[语言模型](https://www.zhihu.com/search?q=语言模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3273259612})（如CLIP [65]）将文本输入编码为[潜在向量](https://www.zhihu.com/search?q=潜在向量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3273259612})，从而实现了最先进的图像生成结果。Glide [57]是一个支持图像生成和编辑的文本[引导扩散模型](https://www.zhihu.com/search?q=引导扩散模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3273259612})。Disco Diffusion [5]在[clip](https://www.zhihu.com/search?q=clip&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3273259612})引导下处理文本提示。Stable Diffusion [81]是潜在扩散[71]的大规模实现。Imagen [77]使用金字塔结构直接扩散像素，而不使用潜在图像。商业产品包括DALL-E2[61]和Midjourney[54]。


链接：https://www.zhihu.com/question/614056414/answer/3273259612

###### 优化 或者基于此的工作

https://cloud.tencent.com/developer/article/2328560



## 实践

### webUI 

部署官网步骤

https://www.stable-learn.com/zh/p3-google-colab-sdw/



这里面有一些模型和参考图

https://www.uisdc.com/47-stable-diffusion-models



[

⑥Counterfeit

模型类别：Checkpoint 模型

下载地址：[ https://civitai.com/models/4468/counerfeit-v25](https://link.uisdc.com/?redirect=https%3A%2F%2Fcivitai.com%2Fmodels%2F4468%2Fcounerfeit-v25)

模型说明：高质量二次元、人物、风景模型。需要额外加载 VAE 模型，提升色彩表现。 [https://huggingface.co/gsdf/Couterfeit-V2.5/blob/main/Couterfeit-V2.5.vae.pt](https://link.uisdc.com/?redirect=https%3A%2F%2Fhuggingface.co%2Fgsdf%2FCouterfeit-V2.5%2Fblob%2Fmain%2FCouterfeit-V2.5.vae.pt) （与模型放同一文件夹）

![出图效率倍增！47个高质量的 Stable Diffusion 常用模型推荐](https://image.uisdc.com/wp-content/uploads/2023/05/uisdc-sd-20230521-8.jpg)

⑦MeinaMix  尝试一下这个）

模型类别：Checkpoint 模型

下载地址：[ https://civitai.com/models/7240/meinamix](https://link.uisdc.com/?redirect=https%3A%2F%2Fcivitai.com%2Fmodels%2F7240%2Fmeinamix)

模型说明：高质量二次元、人物模型。

![出图效率倍增！47个高质量的 Stable Diffusion 常用模型推荐](https://image.uisdc.com/wp-content/uploads/2023/05/uisdc-sd-20230521-9.jpg)

⑧DDicon

模型类别：Checkpoint 模型

下载地址： [https://civitai.com/models/38511/ddicon?modelVersionId=44447](https://link.uisdc.com/?redirect=https%3A%2F%2Fcivitai.com%2Fmodels%2F38511%2Fddicon%3FmodelVersionId%3D44447)

模型说明：一款生成 B 端风格元素的模型，需要结合对应版本的 DDicon lora 使用。V1 版本细节多，v2 版本更简洁，使用时搭配 LoRA 模型 DDICON_lora（ [https://civitai.com/models/38558/ddiconlora](https://link.uisdc.com/?redirect=https%3A%2F%2Fcivitai.com%2Fmodels%2F38558%2Fddiconlora) ）。

]



[一个项目体验stable-diffusion、waifu、NovelAI三大模型 - 飞桨AI Studio星河社区 (baidu.com)](https://aistudio.baidu.com/projectdetail/4704437?channelType=0&channel=0)





## 本地sd

https://zhuanlan.zhihu.com/p/626006585



