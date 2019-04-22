---
title: Collaborative Metric Learning(CML)论文总结
copyright: true
mathjax: true
tags:
  - RS
categories: 推荐
abbrlink: ed2225da
date: 2018-11-13 15:23:58
updated:
---
[论文](http://delivery.acm.org/10.1145/3060000/3052639/p193-hsieh.pdf?ip=112.6.124.171&id=3052639&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E33A83A51ACB36C8E%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1542094284_84051fb2edb5faf170d850677a2f0385)
##问题介绍
度量学习算法**通过学习生成什么样的距离度量来帮助解决数据间的重要关系。**
传统的CF(主要以MF,TF为主)的方法只能学习到user-item之间的潜在关系而无法学习到user-user,item-item的Similarity。本文作者将传统的CF与Metric Learning相结合提出CML,学习到了user-item，以及user-user,item-item的similarity。在借助用户隐式评分下产生了state-of-the-art的Top-k推荐结果。
<!--more-->
##简介
这里我们需要学习一种度量,使得给相似的pair一个较小的距离,给不相似的pair一个较大的距离。
这里的Metric需要满足**三角不等式** (triangle inequality):
即任意两边和大于第三边。也就是说如果x和y,z都相似,那么我们的Metric肯定给$dis(x,y),dis(x,z)$指定的距离很小;此时根据三角不等式$dis(x,y) + dis(x,z) >= dis(y,z)$所以说y和z的距离也应该尽可能的小。
在协同过滤中,我们观察确定的pair(user,item)，根据用户给item打的分数去给那些未观察到的pair提供一些信息。比如矩阵因子分解(MF),它利用user/item之间的点积去获取已知的分数,并用户这种点积来预测未知的分数,但是**这种点积本质上是不满足三角不等式的**。
##背景
###Metric Learning
度量学习中的标签以成对约束的形式指定,包括已知相似的集合$ S = \{(x_i,x_j)\}$，以及不相似的集合对$D = \{x_i,x_j\}$。
原始的Metric Learning方法尝试学习马氏距离度量
{% asset_img 1.png %}
###CollaborativeFiltering
传统的CF算法是基于启发式计算用户的相似度,例如余弦相似度。推荐是根据合计该查询用户的K个最相似的用户的打分。
在过去十年,MF成为了最受欢迎的CF方法,因为他超前的表现。原始的MF模型设计通过去映射user和item到一个隐因子空间而建模用户的显示反馈,这样user-item的关系就可以通过他们之间的点积所捕获。
$r_{ij}$表示$user_i$对$item_j$的打分，我们学习到了用户向量$u_i \in R^r$ item向量$v_j \in R^r$,然后用他们的点积去近似$r_{ij}$。然后我们只需要在已知的数据集上去优化均方误差
{%asset_img 2.png %}
####隐式反馈
直接应用传统的MF在隐式反馈上是存在问题的。通过隐式反馈我们只观察到正反馈。我们不能忽视那些未观察到的user-item交互,否则会导致trival但无用的解决方案。同时我们也不能假定这些未观察到的交互都是负反馈,因为我们不知道这些交互没有发生是因为用户不喜欢他们还是说暂时没什么意愿(可能以后会买)。
Weighted regularized matrix factorization(WRMF)将所有的未观察到的user-item交互看成是负样例,然后使用权重$c_{ij}$去减少这些不确定样本的影响。
{%asset_img 3.png %}
####BPR
上面的讨论表明了,通过隐式反馈,rating开始变得不准确了。最近的一些MF模型开始从估计一组明确的评级转为建模item之间的相对偏好。
Bayesian Personalized Ranking 就是这种方法。
{%asset_img 4.png %}
BPR损失函数存在一个问题,对low-rank item惩罚不足。
##CML
给定所有数据和一个数据集$S$，$S$里面是user-item的数据对，并且是已知具有正相关的，通过学习user-item之间度量来将这种关系编码到第三方空间，具体的学习，拉近$S$中正相关item，推远其他的item。由于三角形不等式,我们也聚合了**1)共同喜欢相同物品的用户在一起2)同一用户共同喜欢的item在一起**
最终,任何给定的用户的最近邻将变为:该用户之前喜欢的item，以及与该用户具有相似品味的其他用户所喜欢的item。
也就是说,通过学习遵循已知正关系的metric,我们不仅将这些关系传播到其他user-item，还将其传播到了item-item,user-user。这些都是我们无法直接观察到的关系。
度量使用的是欧几里得距离:{%asset_img 5.png %}
距离损失函数:{%asset_img 6.png %}
其中$S$为数据中具有正相关的集合，即为$item_j$为用户i所喜欢的，$item_k$为用户i所不喜欢的，$[z]+=max(z,0)$为 hinge loss，$w_{ij}$叫做ranking loss weight，后面会提到，$m>$0为安全边界值。
下图表示了在梯度训练时的过程：{%asset_img 7.png %}
可以看到在loss函数的控制下,正相关的item与user的距离沿着梯度缩小,而不相关的则相反。其中impostor表示在安全边界内出现的负样本(与user不相关的item)。
###Approximated Ranking Loss
距离损失函数中的$w_{ij}$采取一种叫做WARP（Weighted Approximate-Rank Pairwise）的loss来建立
{%asset_img 8.png %}
目的是来惩罚排名靠后的正相关item项。其中:
$rank_d(i,j)$为$user_i$的推荐列表中$item_j$的排名,通过以下的策略来近似:
1、对于每个user-item对$(i,j)$，采样$U$个负样本item，并计算距离损失函数。损失函数非0即为impostor。

2、让$M$代表$U$中impostor的个数，则其中$rank_d(i,j)$可近似为：$ \lfloor \frac{J \times M}{U} \rfloor$
其中，$J$为所有的item数。
###特征损失函数
本文借助隐式反馈作为用户喜欢的item的feature,$f(x)$将特征$x$映射到user-item空间的函数(即为本文的**学习结果**),$v_j$为相应的item向量,特征损失函数的目的是把item的特征向量作为item本身的一种高斯先验，来调整$v_j$的位置,即相同feature的item应该离得更近,并且能改善那些很少评分的item的位置。$f(x)$由训练得来,使用MLP + dropout实现。
特征损失函数如下:
{% asset_img 9.png%}
{% asset_img 10.png%}
上面说到$f(x)$是一个特征函数,负责将item的特征映射到user-item Space中。我们在这里将其看成是item本身的先验来后去调整$v_j$的位置,使得相同特征的item离得更近。Paper中作者使用2层MLP+0.5 Dropout,256-D Hidden-layer. 
本来自己不是很理解这个$f(x)$的作用以及怎么去训练的,于是就去看了一下CML的原代码,发现这个特征函数$f(x)$是跟我们最后的目标函数一起训练的。
我们可以理解成有很多item其实有很多是具有相同特征的(如一篇文章的tags等),通过一个NN将提取出的item特征映射到user-item空间中作为item的先验,可以使得具有相同特征的item$v_j$在Space尽可能离得更近(因为具有相同特征的$v_j$随机初始化的vector可能相差很大,而提取出的相同特征经$f(x)$映射到Space位置大体相差不大,所以可以起到调节作用)。另外对于那些和user交互很少的item,他们本身就无法根据三角形不等式使item之间的位置靠的很近,通过特征函数的映射来调整他们之间的相对位置也是一个很不错的办法。
同时这里的$f(x)$还可以解决RS的cold-start问题,就是说因为我们已经训练好了$f(x)$后,它是将item feature 或者user feature映射到user-item Space的,如果我们现在有一个新user他没有任何的implicit feedback,我们可以通过$f$将其映射到user-item space然后预测其Top-n。同理一个新item也可以这样添加进去。
当然,对于user我们也可以使用该方法。
###正则项损失函数
{% asset_img 11.png%}
其中C为协方差矩阵:{% asset_img 12.png%}

其中$y_i^n$为user or item vector,n为batch中的索引,i为向量的维度。 $ \mu _{i} =\frac{1}{N} \Sigma _{n}y_{i}^{n}  $ ，N为batch大的大小。
**协方差正则的引入,主要是解除各维度间的相关性。因为协方差可以看成向量间的冗余,这样能使得整个空间能进行更多信息的表达**
对于正则损失函数是[此Paper提出](https://arxiv.org/abs/1511.06068)的DeCov loss,其主要是用在Depp networks 为了减少过拟合。和Dropout思想类似,DeCov通过减少各隐藏层神经元之间的相关性去减少过拟合。
该文章中也使用了该正则化方法,刚开始我比较迷惑,就去读了原论文。其实我觉得这里的思想和PCA里的降维很类似(也是根据协方差矩阵来减少各维度之间的相关性,但是我们这里不降维)。我们知道这里我们需要去将每一个user,item embedding到user-item空间中,我们假设他是二维的,这样显得更直观一些。
如图(可能这张图并不是很合适。。。):
{%asset_img 14.png%}
如果我们的user和item是这样的,那么我们就可以发现我们造成了维度浪费,因为图中红线就可以表示他们了(也就是说我们选了r-Dimension其实根本不用),同时可以发现通过协方差接触各维度相关性的同时可以使得我们点"相对分散",去利用好我们的整个Space.
###训练过程
目标函数:{%asset_img 13.png %}
这是总的损失函数，下方的限制条件为保证空间可计算性。整个过程为Mini-Batch SGD训练，学习率采用AdaGrad控制。训练步骤如下：
1.从$S$中抽样N个正相关user-item对
2.对于每一对,抽样$U$个负样本item,并计算近似的$rand_d(i,j) = \lfloor \frac{J \times M}{U} \rfloor$
3.对于每一对,保留使得距离损失函数最大的那个负样本item(k),并组成一个batchN。
4.计算梯度并更新参数
5.重复此过程直到收敛
###实验结果
见论文。
###代码实现
[见github](https://github.com/changun/CollMetric)

