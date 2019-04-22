---
title: Metric-Factorization Recommendation beyond Matrix Factorization
copyright: true
mathjax: true
tags: 论文
categories: 推荐
abbrlink: '49246251'
date: 2018-12-25 16:56:58
updated:
---
[原文链接](https://arxiv.org/abs/1802.04606)
[open source](https://github.com/cheungdaven/metricfactorization)
本文主要对度量分解这篇论文做一个翻译+总结.

##Abstract
被我们熟知的基于矩阵分解(MF)的推荐系统得到了广泛的研究,并已成为最受欢迎的个性化推荐技术之一.然而基于內积的矩阵分解不满足不等式属性(CML中我们提到过的三角不等式),这可能限制了midel的表达并且可能导致次优解.
本文中作者提出了度量分解,假定所有的users和items被embedding到了一个低维的空间并采用满足不等式属性的欧几里得距离来衡量他们之间的相似性.设计了两种方法,将Metric Fac分别用在了rating predict 以及个性化推荐中.
<!--more-->
##Introduction
推荐系统RS基于用户的历史交互,例如评分,以前的购买和点击浏览记录,预测用户对item的偏好.主要存在两种类型的推荐任务:rating prediction 和 item ranking.
rating prediction主要目标是通过用户显示的评分去估计未评分物品的真实值,从而找到用户可能还对哪些物品感兴趣.
在很多场合下因为显示评级的匮乏,通过隐式反馈(purchase,log,click,etc.)的rank-based的RS模型更加可取和实用的用来产生个性化推荐列表去实现我们的推荐目标.
###Metric Fac
度量分解适用于Rating prediction和items ranking.主要原理是使用欧式距离代替MF的点积,从位置和距离的角度来考虑推荐问题.**它将交互矩阵分解为user和item的密集嵌入,并根据他们的估计距离来进行推荐**.
**主要贡献:**
1.提出了Metric Factorization,将users和items表示为多维坐标系中的点(嵌入).他们之间的距离通过欧几里得距离来衡量,基于user-item之间的相似性来进行推荐
2.提出了两种度量分解的变体来解决两种经典的推荐任务: rating prediction 和 item ranking
###矩阵分解(MF)的局限性
前面我们说到基于矩阵分解的推荐或者说基于內积的(主要是內积)的推荐,它的表达存在限制,因为他不满足三角不等式(不等式限制),这可能导致我们的模型是次优解.
我们知道內积$a\cdot b = |a||b|cos \theta$,也就是它只关注两个向量的大小和角度,也就是说它在大小和角度的方面来考虑两个向量之间的相似性而不是距离.
下面我们给出一个例子,假设用户的潜在因子是($P_1,P_2,P_3,P_4$),我们通过计算交互矩阵$R$的Jaccard相似度来衡量用户之间的相似度.
{% asset_img 3.png %}
假定一开始我们有3个users,$u_1,u_2,u_3$那么基于交互矩阵$R$的用户之间的相似度为:$S_{23} > S_{12} > S_{13}$,然后我们的$u_1,u_2,u_3$潜在因子$P_1,P_2,P_3$位置可以是如上图中间.现在我们拥有了一个新的user $u_4$,并且可以得到如下jaccard相似度$S_{41} > S_{43} > S_{42}$,那么**如果我们让$P_4$和$P_1$相似,那么无论我们怎么定义$P_4$的位置$S_{43} > S_{42}$都无法满足.**然而如果我们将users看成空间中的点,并且让相似的user之间的距离更近(也就是有$D_{23} < D_{12} < D_{13}$和$D_{41} < D_{43} < D_{42}$,D代表距离),我们很容易满足条件.

##Problem Formulation
假设我们有M个users和N个items,rating/interaction 矩阵表示为$R \in  R^{M \times N}$,$u$和$i$ 分别表示user和item.$R_{ui}$表示$u$对$i$的偏好.在大多数情况下关于矩阵R很多我们是未知的,我们推荐系统的目标就是去预测这些未观察到的的偏好得分.
矩阵$R$的值可以是显示评分[1-5]也可以是隐式反馈.显示评分可以用来对未知评分进行估计,而隐式反馈则可以用来产生个性化排序的推荐列表(OCCF,Top-N).
隐式反馈的定义如下:
{%asset_img 1.png %}
$u$和$i$之间的交互方式可能有很多,1表示user-item pair $(u,i)$被观察到了,**0 并不一定意味着用户不喜欢item,它也可能意味着用户并没有意识到该item的存在**.
如果$R_{ui} = 1$ 我们称之为positive sample,0 为negative sample
{%asset_img 2.png %}
###Metric Factorization
为了解决上面所说的MF存在的问题,提出了度量分解.矩阵分解将users和items看成是新空间中的向量,metric fac将users和items看成是低维坐标系中的点并且通过欧几里得距离表示他们之间的关系.度量分解将距离分解为users和items的位置.
矩阵分解通过分解表示了user和item之间相似度的交互矩阵来学习users和items的潜在因子.但是学习user和item的位置,我们无法直接利用rating/interaction矩阵$R$,因为距离和相似度是两个相反的概念.对于相似度来说值越大我们说相似度越高,而对于距离值越大代表二者相似度越小,是反的.**因此我们首先将$R$转化为距离矩阵$Y$**,如下:
{%asset_img 4.png %}
根据上面的等式,更高的相似度距离为0,这很好的满足了我们上述所说的条件.这个转化可以用在显式评分也可以用在隐式反馈上.
在欧式空间$R^k$中,我们通过欧式距离来表示两个点之间的距离:
{% asset_img 5.png %}
在度量分解中我们使用欧几里得距离来测量user-item之间的距离.对于每个user和item,我们分配给他们一个向量$P_u \in R^k$,$Q_i \in R^k$来表示他们在多维空间中的位置.所以有:
$$
D(u,i) = ||P_u - Q_i||_2^2
$$
在真实世界中只有很少的一些距离矩阵项是已知的,**度量学习的目标是给定部分观察到的距离矩阵求学习users和items在低维坐标系中的位置:**
$$
f(P_u,Q_i|Y)
$$
对于不同的学习任务(rating prediction,item ranking)$f$是不同的.
目前我们可以看的除度量分解和矩阵分解很相似,他们都是从一个大的稀疏的矩阵去学习,但是他们的解释是不同的.
下图展示了metric factorization的整个过程:
{% asset_img 6.png %}
##基于不同推荐任务的度量分解
这一部分将对度量分解应用于rating prediction和item ranking进行详细的解释
###Metric Fac for Rating Prediction
####Basic Model
假设$S$表示可观察的rating数据的集合,首先我们使用下面的公式将评分矩阵$R$转化为距离矩阵:
$$
Y_{ui} = R_{max} - R{ui}
$$
$R_{max}$表示评分矩阵可能的最大值.
我们使用均方损失函数来学习我们的users和items的位置:
{%asset_img 7.png %}
####偏差和置信度
上面的模型只考虑了user和item之间的交互,但是个人的影响对user和item来说也很重要.例如许多item趋向于收到高评分,一些用户倾向于给低评分.这里我们也引入了全局偏差,users偏差和items偏差.因此,user和item之间的距离表示如下:
$$
\hat Y= ||P_u - Q_i||_2^2 + b_u + b_i + \mu
$$
通常，我们可以添加超参数$\tau$来缩放$\mu$，因为平均距离不能总是反映真实的全局偏差.
另一个重要的方面是我们需要考虑rating data的可靠性和稳定性.大部分rating prediction算法忽视了ratings的噪声并且将所有的ratings看成真值.并非所有的观察到的ratings都应该有相同的权重,例如一些用户可能会给同一item不同的评分,如果你在不同的时间要求他们去评分的话.有研究表明极端的评分(如1分,5分)比中间的评分(2,3,4分)更可靠.所以我们引入了置信度$c_{ui}$,因此我们的损失函数如下:
{% asset_img 8.png %}
$c_{ui}$取值如下:
$$
c_{ui} = 1 + \alpha \cdot g(R_{ui} - \frac{R_{max}}{2})
$$
$g(\cdot)$可以是绝对值,平方,甚至是log函数,$\alpha$是一个超参数,来控制置信度的大小.我们保证了极端的评分获得更高的置信度.
####正则化
#####Norm
对于users和items偏差(bias)我们使用L2正则化,由于对于users和items的向量使用L2正则化会使得users和items靠近原点,所以这里我们不采用L2正则化改为Norm.
$$
||P_u||_2^2 \leq l,||Q_i||_2^2 \leq l
$$
#####Dropout
比较新颖的一个地方,这里将Dropout的思想迁移到users和items坐标的各个维度之间,去减少他们之间的相关性和适应性.所以这里我们也随机丢弃一些坐标维度,然后计算剩余维度之间的距离.
{% asset_img 9.png %}
###Metric Factorization for Item Ranking
对于Item Ranking 我们依然需要先将相似度转化为距离.转化公式如下:
$$
Y_{ui} = a \cdot (1 - R_{ui}) + z \cdot R_{ui}
$$
通过上式我们可以得到 
1.$Y_{ui} = a,R_{ui} = 0$
2.$Y_{ui} = z,R_{ui} = 1$
$a,z$两个超参数用来控制user和item之间的距离.但是需要注意的是我们设置超参数时必须保证$a >=z$**保证user和未交互的item之间的距离大于交互过的item.**通常设$z= 0$已经足够了.**因为$R_{ui}$要么为0要么为1,所以所有(user,positive item)和(user,negative item)之间的距离是分别相等的,而不是像rating prediction那样距离取决于评分**.
为了学习users和items在低维坐标系中的位置我们采用pointwise的方法并最小化如下权重均方损失函数:
{% asset_img 10.png %}
这里我们考虑了所有的未观察到的交互,并且去掉了bias.
$c_{ui}$依然表示的是置信度,定于如下:
$$
c_{ui} = 1 + \alpha w_{ui}
$$
$w_{ui}$可以看成是隐式动作的计数,例如user浏览了item 3次那么$w_{ui} = 3$.因为数据的原因这里w = 0 or 1.$\alpha$和上面的意义相同,
关于正则化配置与上面相同.
下图展示了整个该方法的过程:
{% asset_img 11.png %}
可以看到该模型不仅使user和偏好的item更近,也使得不感兴趣的更远.**不像大多数metric learning基础模型,他们强制imposters远离users的领地至少margin的距离,这个方法中的置信度给负items入侵用户的区域提供了可能性,这对推荐任务来说是有利的,因为它可以作为过滤器来选择否定项目进行推荐.**我们模型的另一个重要特征是它可以间接地将共享大量项目的用户聚集在一起。
##Experiments
### Evalutation for rating prediction
####Dataset
主要使用了4种数据集:
[传送门](http://fastml.com/goodbooks-10k)
{% asset_img 12.png %}
这里采用两种评估指标:RMSE(Root Mean Square Error)和MAE(Mean absolute Error)
####Baselines
{% asset_img 13.png %}
结果如下:
{% asset_img 14.png %}
###Evaluation for Item Ranking
####Dataset
{% asset_img 15.png %}
对于Item Ranking 采用了五个评估标准,[Recall,Precision,MAP,MRR,NDCG](https://statusrank.xyz/2018/12/25/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E5%B8%B8%E7%94%A8%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87/)
{% asset_img 16.png %}

##模型参数分析
这一部分主要讨论了维度大小k对两种推荐任务的影响,Clip Value $l$的影响,置信度等级$\alpha$的影响,以及Dropout  rate的影响(这里发现了Dropout对item ranking无太大作用).
详细分析见paper.
