---
title: Extreme Learning to Rank via Low Rank Assumption论文解读
copyright: true
mathjax: true
tags:
  - SVM
  - RS
categories: 推荐
abbrlink: 37cad335
date: 2018-11-26 15:05:58
updated:
---
在推荐系统和网页搜索中为数十万的用户执行ranking是很常见的。学习单一的ranking function不可能捕捉所有用户的易变性,然而为每个用户学习一个ranking function 是很耗时的,同时也需要来自每个用户的大量数据。
为了解决这个问题,本文作者提出了Factorization RankSVM算法,该算法通过学习k个基础的函数,然后为将这k个ranking function进行线性组合,使得每一个用户有一个ranking function.通过利用low-rank结构,开发了一个更快的算法去减少梯度下降的时间复杂度。同时也证明了他们所提出的方法的泛化误差要好于为每个用户单独执行RankSVN训练一个ranking function.
<!--more-->
##Introduction
###LTR Pairwise method
给你一个具有$x_1,x_2,...x_n$特征和pairwise 比较的训练实例集合,我们的目标是去找到一个最优的决策函数$f(\cdot)$使得如果i的相对偏好大于j则有$f(x_i) > f(x_j)$.
但是在有些情况下,一个单一全局的ranking不足以表示各种各样的用户个人偏好,所以我们考虑**共同学习数十万个ranking function的问题，每个用户一个ranking function**.
我们的目标问题和CR还有BPR的不同,因为他们只是从已知的item中回复ranking,并没有使用item feature,而我们的目标是为了获得ranking function(将item feature 作为输入)，所以可以**推广到看不到的item。**我们这里ranking function的数量T很大(100K),但是可以用来学习每个ranking function的data有限。**朴素的LTR算法是不可以的,因为要学习T个ranking function导致他们的训练次数急剧增加,同时由于训练数据不足会造成过拟合的问题。**

###主要贡献
1.提出了Factorization RankSVM模型在不同的数据集上同时学习大量的不同的ranking function. 通过利用低秩结构,使得梯度的计算变得很高效,同时最后的算法可以扩展到具有大量任务的问题上
2.我们推导出模型的泛化误差界限，表明通过联合训练所有T任务，样本复杂度比在低秩假设下训练单个rankSVM要好得多。
3.我们对现实世界的数据集进行了实验，结果表明，与最先进的方法相比，该算法可以实现更高的准确性和更快的训练时间。
4.我们进一步可视化我们的算法学习的基本排名函数，它具有一些有趣且有意义的模式。
##相关工作比较:
{% asset_img 18.png %}
##Problem Setting
** 我们的目标是一起学习多个ranking function,每个用户一个。**假定一共有T个ranking function需要被学习(每个user可以看成一个task),我们有具有$x_1,x_2,...x_n \in R^d$特征的n个items的pairwise 比较.对于每个task i,pairwise 比较表示为$ \Omega_i = \{\{j,k\}\}$，意思是task i比较了item j和k,并且$y_{ijk} \in \{+1,-1\}$ 表示结果。我们用$\Omega$表示所有$\Omega_i$的集合。
**给定这些pairwise比较,我们的目标是去学习一个线性ranking function集合$w_1,w_2,...w_T \in R^d$**使得:
$$
 sign(w_i^T(x_j-x_k)) \approx y_{ijk}
$$
$$
\forall (j,k) \in \Omega_i ,\forall i = 1,...T
$$
** 我们对T个task做的唯一假设就是每个task中所有的items $ x_j \in  R^d$共享同一个特征空间.**
上面这个模型可以很简单的部署到推荐系统上,每个用户i有一个相对应的ranking function并且items可以是电影,音乐等。我们任务的主要目标是学习这个ranking function，当我们得到$w_i$之后,就可以预测用户i对任意一对$x_j,x_k$的偏好关系,即使他们在训练数据中并没有出现。这是许多协同过滤算法(比如矩阵补全)无法解决的。我们可以预测看不到的items对的偏好主要是因为**我们是基于item feature来学习ranking function的，并不是仅仅通过"seen items"去补全整个评分矩阵**.
###naive method
如图:
{% asset_img 1.png %}
RankSVM是基于单任务的,当然我们也可以假设所有的用户共用一个$w$,这种方法在此论文中叫做 RANKSVM JOINTLY。
(Evgeniou＆Pontil，2004)通过假设每个排名函数为$w_i = w + v_i$来提供变体，其中$w$是集中模型，$v_i$是任务相关方差。这种方法是基于一个很强的假设基础的,Tranking function $\{w_i\}_i^m$都非常接近单一的基础函数$w$,这种假设在实践中并不总是正确的,往往容易造成欠拟合。这种方法在此论文中叫做 RANKSVM JOINTLY。 RankSVM VAR.
我们还可以将每个用户分开来看,这意味着我们可以独立的训练每个$w_i$:
{%asset_img 2.png %}
这种方法叫做 RANKSVM SEPARATELY.很明显，该模型具有更大的自由度来拟合训练数据。但是受限于每个用户能够观察到的pair $\Omega_i$的数量,会导致每个$w_i$由于过拟合而有很差的预测质量.
##Proposed Algorithm
** 我们的低秩个性化ranking猜想就是假设所有T个ranking function都可以近似看成是通过k$(k \lll T)$个基础ranking function线性组合得来的.**
令$\{u_j\}_{j = 1}^k$是basic linear ranking function,我们可以通过线性组合的权重$v_i$来获得每一个ranking function $w_i = \sum_{j=1}^k v_{ij}u_j，\forall i = 1,...T$.
写成矩阵形式就是$W = UV^T$ ，$w_i,u_j$分别是$W,U$的列向量,$V$是系数矩阵,$W \in R^{d \times T}$,$U \in R^{d \times k},V \in R^{T \times k}$.所以$W$是一个秩为k的矩阵,所以我们可以使用核范数来约数矩阵$W$的低秩性:
{%asset_img 3.png %}
<font color = "red">PS:
对于上面所说的矩阵$rank(W) = k$,个人觉得应该是$rank(W)\leq k$，因为这里T和d都是大于k的数,而k是一个几十或者几百(总之比较小的数),作者通过将$W$进行了分解,进而对矩阵$W$做了低秩假设。而又因为矩阵秩是一个非凸问题（NP-hard）不太好优化,所以进而采用核范数去近似。
核范数是矩阵奇异值的和,而矩阵秩就是非0特征值(奇异值)的个数。($R(A^TA) = R(AA^T) = R(A)$,这里就因为$A^TA$一定可以对角化,所以就是非0特征值的个数)
</font>
[关于核范数不错的文章](http://sakigami-yang.me/2017/09/09/norm-regularization-02/)
矩阵的核范数其实就是其奇异值的和,而奇异值又是矩阵$X^TX$的特征值开根号。又根据矩阵的特征值有:$\lambda_1+ \lambda_2,...\lambda_n = x_{11} + x_{22} + ..x_{nn}$,所以综上我们可以得到:
$$
||X||_* = tr (\sqrt{X^TX})
$$
到这里我们仍然不能很好的解决问题,因为我们的矩阵$W$为$d\times T$，它包含$dT$个参数。所以我们利用:
$$
||W||_* = min_{W = UV^T} \frac{1}{2} (||U||_F^2 + ||V||_F^2)
$$
所以我们的得到如下公式:
{% asset_img 4.png %}
经过上面的变化,因为$U \in R^{d \times k},V \in R^{T \times k}$所以这里我们一共有$(d + T)k$个参数。(注意我们的$k \lll T$)
将上述问题转化为无条件限制的问题：
{% asset_img 5.png %}
使用** 交替最小**方法。每次迭代固定$V$更新$U$,然后固定$U$更新$V$。
当固定$V$更新$U$时:
{%asset_img 6.png %}
其梯度可以写成:
{%asset_img 7.png %}
我们发现要计算(4)式时间复杂度为$O(|\Omega|kd)$,因为我们需要两个求和遍历整个数据集(这刚好看成$\Omega$,前面我们假设过它是所有的并集),每次都需要计算$(x_j-x_k)v_i^T$.
于是作者提出如下算法,可以使得更新$U$的梯度时复杂度变为：$O(T\overline nk + dkn + |\Omega|)$，其中$\overline n$是每个用户的平均打分。
算法如下:
{% asset_img 8.png %}
其实该算法就是将上述的:{% asset_img 9.png %}对每一个$(j,k)$进行展开,可以得到:
$$-2Cy_{ijk}max(0,1-y_{ijk}(q_j-q_k))x_jv_i^T + 2Cy_{ijk}max(0,1-y_{ijk}(q_j-q_k))x_kv_i^T
$$
所以我们每次就分开维护一下最后加起来就好了。
对于更新$V$：
{% asset_img 10.png %}
对其求偏导有:
$$
v_i = v_i + \sum_{(j,k) \in \Omega_i}-2Cy_{ijk}max(0,1-y_{ijk}  \overline v_i^TU^T(x_j-x_k))U^T(x_j-x_k)
$$
优化方法同样借鉴上面的,稍微变下形式就好了,比如把$v_i$换成$U^T$。
总结:该算法每一次迭代所需的时间为$O(T\overline nk + dkn + |\Omega|)$，他们相对于$|\Omega|$是很小的,所以该算法近似是线性的,可以扩展到更大的数据集上。
##样本复杂度分析
这一部分真心没看懂...暂时先放一下。
这一部分大体证明的就是,作者这里基于低秩假设的$W$，同时学习T个ranking function所需要的样本数据比将每个task 视为一个RankSVM然后单独去训练(就是我们上文说的RankSVM SEPARATELY)所需要的数据是小的(而且优势较大)。

##Experiment
这里训练时使用了两种数据,一种是合成的(随机初始化),一种是真实数据。
###实验配置
对于每个ranking task,我们随机的将items分为training item或者testing item.
在训练阶段,使用训练item中的所有对来训练模型;在测试阶段,我们评估所有testing-testing items pair 和testing-training items pair 的预测精度.
**准确率定义为:**
$$ 
accuracy = \frac{正确的预测}{总的预测对数}
$$
我们比较了 RankSVM JOINTLY(为所有任务训练一个模型),RankSVM SEPARATELY(为每个任务训练一个模型),RankSVM VAR(the multi-task rankSVM model proposed in (Evgeniou & Pontil, 2004)).
###Synthetic(合成)data
对于合成数据我们假设有1000个任务,10000个items并且每个item 有64 features.
{%asset_img  11.png %}
我们为每个用户采样800对作为训练数据，标签基于$R = (W')^TX $
比较结果如下:
{% asset_img 12.png %}
{% asset_img 13.png %}
### Real world datasets
{% asset_img 14.png %}
选取了三个数据集:
(1) Yahoo! Movies User Ratings and Descriptive Content Information V1_0 
(2) HetRec2011-MovieLens-2K (Cantador et al., 2011). 
(3) MovieLens 20M Dataset (Harper & Konstan, 2016)
(1)使用title 和 摘要结合起来作为feature (2)(3)将每个电影的类型作为特征
具体信息如下:
{%asset_img 15.png %}
结果如下:
{% asset_img 16.png %}
**可以看出该方法在准确率和速度上都优于其他算法,无论是在密集的feature还是稀疏的feature.尤其是对于(3)数据集,task超过了100000,效率仍然很高.**
时间精度图像如下:
{% asset_img 17.png %}
我们需要注意的是RankSVM JOINTLY(也就是为所有user训练一个ranking function),刚开始很快,但是最后它无法收敛到一个好结果。
##Conclusion
本文提出可一个基于RankSVM 和 MF相结合的学习多个ranking function 的算法.因为这里做了低秩假设,所以task数可以很多,而将每个任务独立的去RankSVM就受限于task的个数了。
另外一个比较有意思的idea就是:文中说的$W$是由$k$个基础线性ranking function组合得来,我们可以想一下能否通过非线性ranking function,比如引入神经网络等。


