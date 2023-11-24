# 实验室相关yy

#### 远程连接 jupyter notebook

打开xshell

![image-20231112205412866](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231112205412866.png)

source activate

conda activate chris_torch

jupyter notebook --no-browser --port=8866



打开cmd,输入类：

所以ssh -L x:localhost:y 指的是你需要在本地输入localhost:x ,于是你就连上了服务器的y端口

ssh -L 8890:localhost:8866 yingying@59.77.5.64 -p 10001



然后打开http://localhost:8890/

输入密码 （小写 +2 ）





cd DeepLearningHM/Assignment3





##### jupyter notebook 远程连接

后台挂起

https://zhuanlan.zhihu.com/p/365985728



##### vscode 远程连接

https://zhuanlan.zhihu.com/p/140899377





下载jupyter notebook

https://zhuanlan.zhihu.com/p/136862576

![image-20231114160816193](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231114160816193.png)





![image-20231114161650273](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231114161650273.png)



远程连接

https://blog.csdn.net/weixin_44244168/article/details/125698441

https://blog.csdn.net/wangzhihao1994/article/details/100047558







https://blog.csdn.net/u012228523/article/details/128344285?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-128344285-blog-116720696.235^v38^pc_relevant_sort&spm=1001.2101.3001.4242.2&utm_relevant_index=4





点击运行 ，选择哪个jupyter 服务器啥的

在xshell中输入![image-20231115102708459](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231115102708459.png)

![image-20231114170823035](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231114170823035.png)	

![image-20231114170607170](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231114170607170.png)

复制例如以上信息到那个位置



### 查看并清除进程

xshell中输入`nvtop`

![image-20231115102823969](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231115102823969.png)



可见以上橘色框中的进程占用，（即使已经关闭程序了）

`Fn+F10`退出

输入`kill -9 16305`

被清除

![image-20231115103021535](E:\研究生\科研\实验室相关\实验室相关yy.assets\image-20231115103021535.png)