---
title: 贝叶斯个性化排序(BPR)算法小记
copyright: true
mathjax: true
tags:
  - RS
categories: 推荐
abbrlink: 6a0c1b07
date: 2018-11-19 16:31:01
updated:
---
[总结的很好](https://www.cnblogs.com/pinard/p/9128682.html)
[论文翻译](https://blog.csdn.net/qq_20599123/article/details/51315697)
在有些推荐场景中,我们是为了在千万级别的商品中推荐个位数的商品给用户,此时我们更关注的是对用户来说，哪些极少数商品在用户心中有更高的优先级，也就是排序更靠前。说白了,我们需要一个算法，这个算法**可以把每个用户对应的所有商品按喜好排序**。
<!--more-->
###建模思路
在BRR算法中，我们将任意用户u对应的物品进行标记,如果用户u在同时有item i和j时点击了i,那么我们就得到了一个triplet$(u,i,j)$,他表示对用户u来说,i的排序要比j更靠前。如果对于用户u来说我们有m组这样的反馈,那么我们就可以得到m组用户u对应的训练样本。
基于贝叶斯,我们有如下两个假设:
1.每个用户之间的偏好行为相互对立,即用户u在item i和j之间的偏序关系与其他用户无关
2.同一用户在不同商品之间的偏序与其他商品无关。也就是u在item i 和j之间的偏序关系与其他商品无关。
为了便于表述，我们用$>_u$符号表示用户u的偏好，上面的$(u,i,j)$可以表示为：$i>_uj$。
在BPR中，这个排序关系符号$>_u$满足**完全性，反对称性和传递性**，即对于用户集U和物品集I：
{% asset_img 1.png %}
同时BPR也用了类似的矩阵分解模型,这里BPR对于用户集$U$合物品集$I$对于的$U \times I$的预测排序矩阵$ \overline X$,我们期望得到两个分解后的用户矩阵$W(|U| \times k)$和物品矩阵 $H(|I|\times k)$,满足:
$$
\overline{X} = WH^T
$$
** 这里的K也是自己定义的,一般远远小于$|U|,|I|$**
由于BPR是基于用户维度的，所以对于任意一个用户u，对应的任意一个物品i我们期望有：
$$
\overline{x}_{ui} = w_u \bullet h_i = \sum\limits_{f=1}^kw_{uf}h_{if}
$$
最终我们的目标，是希望寻找合适的矩阵$W,H$，让$\overline X$和$X$最相似。
##BPR优化
可能看了上面,大家可能和我一样感觉BPR好像和一般的矩阵分解没什么不同。
**BPR是基于最大后验估计$P(W,H|>_u)$来求解模型参数$W,H$的**,这里我们用$\theta$来表示参数$W,H$,$>_u$表示用户u对应的所有商品的全序关系,则优化目标是$P(\theta|>_u)$.根据贝叶斯公式得:
$$
P(\theta|>_u) = \frac{P(>_u|\theta)P(\theta)}{P(>_u)}
$$
由于我们求解假设了用户的排序和其他用户无关，那么对于任意一个用户u来说，$P(>_u)$对所有的物品一样，所以有：
$$
P(\theta|>_u) \propto P(>_u|\theta)P(\theta)
$$
这个优化目标转化为两部分。第一部分和样本数据集D有关，第二部分和样本数据集D无关。
对于第一部分，由于我们假设每个用户之间的偏好行为相互独立，同一用户对不同物品的偏序相互独立，所以有：
$$
\prod_{u \in U}P(>_u|\theta) = \prod_{(u,i,j) \in (U \times I \times I)}P(i >_u j|\theta)^{\delta((u,i,j) \in D)}(1-P(i >_u j|\theta))^{\delta((u,j,i) \not\in D) }
$$
其中，
$$
\delta(b)= \begin{cases} 1& {if\; b\; is \;true}\\ 0& {else} \end{cases}
$$
根据上面讲到的完整性和反对称性，优化目标的第一部分可以简化为：
$$
\prod_{u \in U}P(>_u|\theta) = \prod_{(u,i,j) \in D}P(i >_u j|\theta)
$$
而对于$ P(i >_u j|\theta)$这个概率，我们可以使用下面这个式子来代替:
$$
P(i >_u j|\theta) = \sigma(\overline{x}_{uij}(\theta))
$$
其中$\sigma(x)$是sigmoid函数。这里你也许会问，为什么可以用这个sigmoid函数来代替呢? 其实这里的代替可以选择其他的函数，不过式子需要满足BPR的完整性，反对称性和传递性。原论文作者这么做除了是满足这三个性质外，另一个原因是为了方便优化计算。
对于$\overline{x}_{uij}(\theta)$这个式子，我们要满足当$i >_u j$时，$\overline{x}_{uij}(\theta) > 0$,反之当$j>_u i$时，$\overline{x}_{uij}(\theta) < 0$最简单的表示这个性质的方法就是
$$
\overline{x}_{uij}(\theta) = \overline{x}_{ui}(\theta) - \overline{x}_{uj}(\theta)
$$
而$ \overline{x}_{ui}(\theta) , \overline{x}_{uj}(\theta)$,就是我们的矩阵$\overline X$对应位置的值。这里为了方便，我们不写$\theta$,这样上式可以表示为:
$$
\overline{x}_{uij} = \overline{x}_{ui} - \overline{x}_{uj}
$$
注意上面的这个式子也不是唯一的，只要可以满足上面提到的当$i >_u j$时，$\overline{x}_{uij}(\theta) > 0$，以及对应的相反条件即可。这里我们仍然按原论文的式子来。
最终，我们的第一部分优化目标转化为：
$$
\prod_{u \in U}P(>_u|\theta) = \prod_{(u,i,j) \in D} \sigma(\overline{x}_{ui} - \overline{x}_{uj})
$$
对于第二部分$P(\theta)$,作者大胆使用了贝叶斯假设,即这个概率分布符合正态分布,且对应的均值是0,协方差矩阵是$ \lambda_{\theta}I$
$$
P(\theta) \sim N(0, \lambda_{\theta}I)
$$
**作者为什么这么假设呢？主要还是为了优化方便，因为后面我们做优化时，需要计算$lnP(\theta)$，而对于上面假设的这个多维正态分布，其对数和$||θ||^2$成正比。即**
$$
lnP(\theta) = \lambda||\theta||^2
$$
最终对于我们的最大对数后验估计函数：
$
ln\;P(\theta|>_u) \propto ln\;P(>_u|\theta)P(\theta) = ln\;\prod\limits_{(u,i,j) \in D} \sigma(\overline{x}_{ui} - \overline{x}_{uj}) + ln P(\theta) = \sum\limits_{(u,i,j) \in D}ln\sigma(\overline{x}_{ui} - \overline{x}_{uj}) + \lambda||\theta||^2\;
$
这个式子可以用梯度上升法或者牛顿法等方法来优化求解模型参数。如果用梯度上升法，对$\theta$求导，我们有：
$$
\frac{\partial ln\;P(\theta|>_u)}{\partial \theta} \propto \sum\limits_{(u,i,j) \in D} \frac{1}{1+e^{\overline{x}_{ui} - \overline{x}_{uj}}}\frac{\partial (\overline{x}_{ui} - \overline{x}_{uj})}{\partial \theta} + \lambda \theta
$$
由于:
$$
\overline{x}_{ui} - \overline{x}_{uj} = \sum\limits_{f=1}^kw_{uf}h_{if} - \sum\limits_{f=1}^kw_{uf}h_{jf}
$$
这样我们可以求出：
$$
\frac{\partial (\overline{x}_{ui} - \overline{x}_{uj})}{\partial \theta} = \begin{cases} (h_{if}-h_{jf})& {if\; \theta = w_{uf}}\\ w_{uf}& {if\;\theta = h_{if}} \\ -w_{uf}& {if\;\theta = h_{jf}}\end{cases}
$$
##BPR算法流程
下面简要总结下BPR的算法训练流程：
输入：训练集$D$三元组，梯度步长$\alpha$， 正则化参数$\lambda$,分解矩阵维度$k$。　
输出：模型参数，矩阵$W,H$
1. 随机初始化矩阵$W,H$
2. 迭代更新模型参数：
$$
w_{uf} =w_{uf} + \alpha(\sum\limits_{(u,i,j) \in D} \frac{1}{1+e^{\overline{x}_{ui} - \overline{x}_{uj}}}(h_{if}-h_{jf}) + \lambda w_{uf})
$$
$$
h_{if} =h_{if} + \alpha(\sum\limits_{(u,i,j) \in D} \frac{1}{1+e^{\overline{x}_{ui} - \overline{x}_{uj}}}w_{uf} + \lambda h_{if})
$$
$$
h_{jf} =h_{jf} + \alpha(\sum\limits_{(u,i,j) \in D} \frac{1}{1+e^{\overline{x}_{ui} - \overline{x}_{uj}}}(-w_{uf}) + \lambda h_{jf})
$$
3. 如果$W,H$收敛，则算法结束，输出$W,H$，否则回到步骤2.
当我们拿到W,H后，就可以计算出每一个用户u对应的任意一个商品的排序分：$\overline{x}_{ui} = w_u \bullet h_i$，最终选择排序分最高的若干商品输出。
##小结
**BPR是基于矩阵分解的一种排序算法，但是它不是做全局优化,而是针对每一个用户自己的喜好评分做优化,这和其他MF方法的迭代思路完全不同。另外,BPR的训练数据集则需要用户对商品喜好排序的三元组做训练集。**

