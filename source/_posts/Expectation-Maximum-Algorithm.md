---
title: EM(Expectation Maximum) 算法总结
copyright: true
mathjax: true
date: 2019-05-13 20:33:04
tags: EM
categories: 机器学习
updated:
---
EM算法，全称为Expectation Maximum Algorithm，是一个基础算法，是很多机器学习领域算法的基础（如HMM，LDA等）。EM算法是在**概率模型中寻找参数最大似然估计或者最大后验估计的算法，其中概率模型依赖于无法观测的隐含变量。**
它经过两个步骤交替进行计算：
 1. 计算期望（E步），基于现有的模型参数（或者随机初始化的模型）对隐含变量的值进行猜测（估计），利用隐含变量现有的估计值，计算其最大似然的估计值。
 2. 最大化（M步），最大化在E步上求得的最大似然值来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。
<!--more-->
# EM解决的问题
   我们经常会从样本观察数据中，找出样本的模型参数。其中最常用的就是最大似然估计。但是在一些情况下，我们观察得到的数据有未观察到的隐含数据，此时我们未知的有隐含数据和模型参数，因此无法直接使用最大似然估计。
EM算法解决这个问题的思路是使用启发式的迭代方法。既然我们无法直接求出模型的参数，那么我们可以先猜想隐含数据——E步，接着基于观察数据和猜测的隐含数据一起来进行最大似然估计，进而求得我们模型分布的参数——M步。由于我们之前的隐藏数据是猜测的，所以此时得到的模型参数并不一定是最好的结果。因此，我们基于当前得到的模型参数来继续猜测隐含数据，然后进行最大似然估计求出模型分布参数。以此类推，不断迭代，直到模型分布参数基本不变化或变化很小，算法收敛停止。
   一个最直观的EM算法是[K-Means聚类算法](https://statusrank.xyz/articles/6a624875.html)。在K-Means聚类时，每个聚类的质心可以看成是隐含数据。我们会假设$K$个初始化质心，即EM算法的E步；然后计算每个样本和$K$个质心之间的距离，并把样本聚类到最近的那个质心类中，即EM算法的M步。重复这个E步和M步质心不在变化为止。
# EM算法的数学基础
## 极大似然估计
### 似然函数
在数理统计学中，似然函数是一种关于统计模型中参数的函数，表示模型参数中的似然性（某种事件发生的可能性）。显然，极大似然就是最大可能性的意思。
**多数情况下我们是根据已知条件来推算结果，而最大似然估计是已知结果来寻求使该结果出现的可能性最大的条件，以此作为估计值。**\
假定已知某个参数$B$时，推测事件$A$发生的概率为:
$$
P(A|B) = \frac{P(A,B)}{P(B)}
$$
由贝叶斯公式，可以得出:
$$
P(B|A) = \frac{P(B)\cdot P(A|B)}{P(A)}
$$
一般来说，事件$A$发生的概率与某一未知参数$B$有关，$B$的取值不同，则事件$A$发生的概率$P(A|B)$也不同，当我们在一次试验中事件$A$发生了，则认为此时的$\hat B$值应该是$B$的所有取值中使得$P(A|B)$达到最大的那一个，极大似然估计法就是要选取这样的$\hat B$值作为参数$B$的估计值，使选取的样本在被选的总体中出现的可能性为最大。
**直观的例子:** 设甲箱子中有$99$个白球，$1$个黑球；乙箱中有$1$个白球，$99$个黑球。现随机取出一箱，再从抽取的一箱中随机取出一球，结果是黑球，这一黑球从乙箱取出的概率比从甲箱取出的概率大的多，这时我们自然更多地相信这个黑球是取自乙箱的。
### 极大似然估计举例
假设我们要调查我们学校的男生和女生的身高分布。假设我们在校园里随机找了男女生分别$100$个，共$200$人。我们将他们按性别划分为两组，然后先统计抽样得到的$100$个男生的身高。假设身高都服从正态分布，但是分布的参数均值$\mu$和方差$\sigma^2$我们不知道。这也正是我们所要估计得参数，记为$\theta = [\mu,\sigma^2]^T$。问题相当于已知样本集$X$，求$P(\theta|X)$
**我们知道样本所服从的概率分布和模型的一些样本，需要求解该模型的参数。**如图：
{%asset_img 1.png %}
下面我们通过使用最大似然估计来估计出我们模型的参数。
设样本集$X = {x_1,x_2,...x_N},N=100，P(x_i|\theta)$为概率密度函数，表示抽到男生$x_i$（身高）的概率。我们一般假设每个样本之间是独立同分布的，所以我同时抽到他们的概率就是他们各自概率的乘积，也就是样本集$X$中各个样本的联合概率:
$$
L(\theta) = L(x_1,x_2,....x_N;\theta) = \prod_{i=1}^N p(x_i|\theta)
$$
通常情况下，因为这里的$L(\theta)$是连乘的，所以我们一般取对数。
$$
L(\theta) = \sum_{i = 1}^N \log p(x_i|\theta)
$$
似然函数$L(\theta)$反映了在概率密度函数的参数是$\theta$时，得到样本集$X$的概率。我们需要找到这样一个参数$\hat\theta$，使抽到$X$这组样本的概率最大，也就是需要其对应的似然函数$L(\theta)$最大。满足这样条件的$\hat\theta$叫做$\theta$的最大似然估计量，记为：
$$
\hat\theta = \arg \max_\theta L(\theta)
$$

**极大似然估计步骤**:
 1.   写出似然函数；
 2. 取对数得到对数似然函数并整理
 3. 求导数，令其导数为$0$得到似然方程
 4. 解似然方程，得到的参数即为所求

## Jensen不等式
设$f$是定义域为实数的函数，如果对于所有的实数$x$，$f(x)$的二次导数大于等于$0$，那么$f$是**凸函数**。当$x$是向量时，如果Hessian矩阵$H$是半正定的（$H>=0$），那么$f$是凸函数。如果$f''(x) > 0,H>0$，那么是严格的凸函数。

**Jensen不等式**：如果$f$是凸函数，$X$是随机变量，那么有$E[f(X)] >= f(E[X])$，也就是**函数的期望大于等于期望的函数**。特别地，如果$f$是严格的凸函数，那么$E[f(x)] = f(E[x])$当且仅当$P(X = E[X]) = 1$时成立，也就是说$X$是常数。
对于**凹函数**，不等号方向反向 $E[f(X)] <= f(E[X])$。
如图:
{% asset_img 2.png %}
图中实现$f$是凸函数，$X$是随机变量，有$0.5$的概率是$a$，有$0.5$的概率是$b$。$X$的期望值就是$a$和$b$的中值了，从上图中我们可以看到$E[f(x)] >= f(E[x])$成立。
# EM算法
## 问题描述
极大似然估计用一句话概括就是：**知道结果$X$，反推模型参数$\theta$**。
对于上面我们举的男女身高的例子，极大似然估计的目标就是根据男女分别$100$个身高样本，来分别估计男女生身高的正态分布的参数$\mu_1,\sigma_1,\mu_2,\sigma_2$。但是极大似然估计面临的概率分布只有一个或者知道结果是通过哪个概率分布实现的，只不过你不知道这个概率分布的参数。现在我们让情况更复杂一些:
我们挑选的这$200$人混在一起了，也就是说我们拥有$200$人的身高数据，但是我们不知道每个人是男生还是女生，此时的男女性别就像是一个**隐变量(latent variable)**。
<font color = "red">通常来说，我们只有精确的知道了男女生身高的正态分布参数，才能更好的知道每一个人是男生和女生哪个概率更大一些。但是反过来，我们必须知道每个人是男生还是女生才能用最大似然估计尽可能准确地估计男女各自身高的正态分布参数。</font>
EM算法就是为了解决这种问题而存在的。
{% asset_img 3.png %}
## EM算法中的隐变量
一般的用$X$表示观测到的随机变量的数据，$Z$表示隐随机变量的数据（因为我们观测不到结果是从哪个概率分布中得出的，所以将这个叫做隐变量，它一般是离散的）。$X$和$Z$连在一起称为完全数据，单独一个$X$我们称为观测数据。
其实我们可以发现，我们所面临的问题其实就是$Z$是未知的，如果隐变量$Z$已知，那么问题就可以用极大似然估计来求解了。因此，EM算法的基本思想就是:
 1. 先给模型参数$\theta$一个随机初始化的初值$\theta_0$
 2. 根据模型参数$\theta$和给定的观测数据，求未观测数据$z$的条件概率分布期望
 3. 利用上一步已经求出的$z$，进行极大似然估计，得到更优的$\theta'$
 4. 不断进行2,3步的迭代，直到收敛
 
## EM算法的另一个例子-抛硬币
{% asset_img 4.jpg %}
如上图，现在我们抛两枚硬币$A$和$B$，如果知道每次抛的是$A$还是$B$，那么就直接可以估计每种硬币的参数$\theta_A,\theta_B$（正面朝上的概率）。如果我们只观测到$5$轮循环，每轮抛硬币$10$次，而并不知道每轮是抛的哪个硬币（这就是隐变量），那么我们就无法直接估计参数$\theta_A,\theta_B$。这时我们可以使用EM算法，先对参数$\theta_A,\theta_B$进行随机初始化，然后根据模型参数去计算隐变量的条件概率。比如对于第一组数据“HTTTHHTHTH”,为$A$的可能性为：$0.6^5 \times (1-0.6)^5$,为$B$的可能性为：$0.5^5\times0.5^5$，二者进行归一化得出隐变量有$0.45$的概率是硬币$A$，有$0.55$的概率是硬币$B$。得到隐变量$z$后我们可以去进行M步计算极大似然估计求得更好的$\theta'$,..以此类推。
## EM算法的推导
对$m$个样本观察数据$x=(x^{(1)},x^{(2)},...x^{(m)})$中，找出样本的模型参数$\theta$,最大化模型分布的对数似然函数如下：
$$
\theta = arg \max \limits_{\theta}\sum\limits_{i=1}^m logP(x^{(i)};\theta)
$$
如果我们得到的观察数据有未观察到的隐含数据$z=(z^{(1)},z^{(2)},...z^{(m)})$,此时我们模型的最大化（极大化）模型分布的对数似然函数如下：
$$
\theta = arg \max \limits_{\theta}\sum\limits_{i=1}^m logP(x^{(i)};\theta) = arg \max \limits_{\theta}\sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}P(x^{(i)}， z^{(i)};\theta)
$$
对于参数估计，我们本质上还是想获得一个使siranhanshu最大化的那个参数$\theta$，现在的上式与之前不同的是我们似然函数之中多了一个未知的变量$z$。也就是说我们的目标是找到合适的$\theta$和$z$使得似然函数最大。那我们也许会想，仅仅是多了个未知变量而已，我们也可以分别对$\theta$和$z$求偏导，再令其为$0$,求出对应的解即可。但是，这里我们看到上式包含了“和”的对数，求导之后的形式非常复杂，所以很难求出对应的解析解。因此EM算法采用了一些技巧，让我们接着往下看。
$$
\begin{aligned} \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}P(x^{(i)}， z^{(i)};\theta)   & = \sum\limits_{i=1}^m log\sum\limits_{z^{(i)}}Q_i(z^{(i)})\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \\ & \geq  \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \end{aligned}
\tag{1}
$$
第一步我们引入了一个未知分布（隐变量$z$的概率分布）$Q_i(z^{(i)})$，其中$\sum_{z^(i)}Q_i(z^{(i)}) = 1$。（这一步其实我们什么都没有做，只是对分子分母进行了缩放）。
第二步我们使用的是上面我们提到过的$Jensen$不等式。因为这里$\log$函数是一个凹函数(二次导师为$\frac{-1}{x^2}$)，所以根据Jensen不等式我们有$E[f(x)] \leq f(E[x])$。至此，我们就将“和”的对数变为了对数的“和”，再求导就很容易了。
上式中的$\sum\limits_{z^{(i)}}Q_i(z^{(i)})\log\frac{P(x_i,z^{(i)};\theta)}{Q_i(z^{(i)})}$，其实就是函数$\log\frac{P(x_i,z^{(i)};\theta)}{Q_i(z^{(i)})}$关于$Q_i(z^{(i)})$的期望。为什么？回想期望公式中的lazy Statistician规则如下:
>设$Y$是随机变量$X$的函数$Y = g(X)$（$g$是连续函数），那么
>1. $X$是离散型随机变量，它的分布律为$P(X = x_k)  = p_k,k = 1,2,\dots$,
>$\sum\limits_{k = 1}^{\infty}g(x_k)p_k$绝对收敛，则有$E(Y) = E(g(X)) = \sum\limits_{k
>= 1}^{\infty}g(x_k)p_k$。
>2. $X$是连续型随机变量，它的概率密度为$f(x)$,若$\int\limits_{
>\infty}^{\infty}g(x)f(x)dx$绝对收敛，则有$E(Y) = E(g(X)) = \int\limits_{
>\infty}^{\infty}g(x)f(x)dx$.

考虑到$E(X) = \sum x\cdot p(x)$，则有$E(f(x)) = \sum f(x)\cdot p(x)$，又因为有$\sum\limits_{z^{(i)}}Q_i(z^{(i)}) = 1$，所以我们可以使用Jensen不等式得到(1)式中的不等式:
$$
f\left(E_{Q_i(z^{(i)})}\left[\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \right]\right) \geq E_{Q_i(z^{(i)})}\left[f\left(\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} \right)\right]
$$
到这里我们就应该很清楚了，但是我们大家可能发现这里是一个**不等号**，而我们是想求得**似然函数的最大值**，那么应该怎么办呢？
其实上式(1)相当于对**似然函数求了下界**，因此我们可以通过调整使得下界不断的变大从而似然函数也不断的变大，以逼近真实值。那么什么时候调整好了呢？当不等式等于等式时说明二者已经等价了。如图:
{% asset_img 5.jpg %}
我们先固定$\theta$,调整$Q(z)$使得下界$J(z,Q)$上升至与$L(\theta)$在此$\theta$处相等（绿色曲线到蓝色曲线），然后固定$Q(z)$,调整$\theta$使当前的下界$J(z,Q)$达到最大值（$\theta_t$到$\theta_{t + 1}$）,然后再固定$\theta_t$，调整$Q(z)$使下界和$L(\theta)$相等，。。。直到收敛到似然函数$L(\theta)$的最大值处$\theta'$。
**那么等式成立的条件是什么呢？**
根据Jensen不等式中等式成立的条件$X$为常数，这里等式成立的条件为：
$$
\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})} =c, c为常数
$$
由于$Q_i(z^{(i)})$是一个未知的分布且有$\sum\limits_{z^{(i)}}Q_i(z^{(i)}) =1$,因此我们可以进行一个变换：
$$
\sum\limits_{z^{(i)}}P(x^{(i)},z^{(i)};\theta) = \sum\limits_{z^{(i)}}Q_i(z^{(i)})c
$$
也就是 :
$$
\sum\limits_{z^{(i)}}P(x^{(i)},z^{(i)};\theta) = c
$$
因此我们可以得到:
$$
\begin{aligned}
Q_i(z^{(i)}) &= \frac{P(x^{(i)},z^{(i)};\theta)}{c}\\
&= \frac{P(x^{(i)},z^{(i)};\theta)}{\sum\limits_{z^{(i)}}P(x^{(i)},z^{(i)};\theta)}\\
&=\frac{P(x^{(i)},z^{(i)};\theta)}{P(x^{(i)};\theta)} \\
&= P(z^{(i)}|x^{(i)};\theta)
\end{aligned}
$$
<font color = "red">至此，我们推出了在固定$\theta$之后，$Q_i(z^{(i)})$如何选择的问题——使下界拉升的$Q_i(z^{(i)})$的计算公式就是条件概率（后验概率）。这一步就是$E$步，固定$\theta$,建立$L(\theta)$的下界，并求得使$L(\theta)$等于下界$J(z,Q)$时等号成立的$Q_i(z^{(i)})$.
接下来是$M$步，就是在$E$步求出$Q_i(z^{(i)})$后，固定$Q_i$,调整 $\theta$，去最大化下界$J(z,Q)$，毕竟在固定$Q_i(z^{(i)})$后，下界还可以更大。</font>
即在$M$步我们需要最大化下式:
$$
arg \max \limits_{\theta} \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log\frac{P(x^{(i)}， z^{(i)};\theta)}{Q_i(z^{(i)})}
$$
由于$M$步我们固定$Q_i$，所以去掉上式中的常数部分，则我们需要极大化的**对数似然下界**为:
$$
arg \max \limits_{\theta} \sum\limits_{i=1}^m \sum\limits_{z^{(i)}}Q_i(z^{(i)})log{P(x^{(i)}, z^{(i)};\theta)}
$$
至此，我们应该理解了EM算法中E步和M步的具体含义。
# EM算法的流程
输入：观察数据$x=(x^{(1)},x^{(2)},...x^{(m)})$,联合分布$p(x,z ;\theta)$,条件分布$p(z|x; \theta)$,最大迭代次数$J$。
 1. 随机的初始化模型参数$\theta$为$\theta_0$
 2. for j from 1 to J开始EM算法迭代：
	 a) E步: 计算联合分布的条件概率期望:
	 $$
	 Q_i(z^{(i)}) = P(z^{(i)}|x^{(i);\theta}) \\
	 L(\theta,\theta^j) = \sum\limits_{i = 1}^m\sum\limits_{z^{(i)}} Q_i(z^{(i)})\log P(x^{(i)},z^{(i)};\theta)
	 $$
	 b) M步: 固定$Q_i$，极大化$L(\theta,\theta^j)$:
	 $$
	 \theta^{j+1} = arg \max \limits_{\theta}L(\theta, \theta^{j})
	 $$
	 c) 如果$\theta^{j + 1}$已收敛，则算法结束。否则继续回到步骤a)进行E步迭代。

输出：模型参数$\theta$
# EM算法收敛性的证明
关于EM算法收敛性的证明，我们觉得B站上shuhuai008给出的两种方法非常直观[https://www.bilibili.com/video/av31906558/?p=2](https://www.bilibili.com/video/av31906558/?p=2)
[https://www.bilibili.com/video/av31906558/?p=3](https://www.bilibili.com/video/av31906558/?p=3)
EM算法的流程并不复杂，但是还是有两个问题需要我们思考：
 1. EM算法能保证收敛吗？
 2. EM算法如果保证收敛，那么保证能收敛到全局最优解吗？
 
 **要证明EM算法的收敛性，我们要证明我们的对数似然函数的值在迭代过程中值在增大。** 即:
 $$
 \sum\limits_{i=1}^m logP(x^{(i)};\theta^{j+1}) \geq \sum\limits_{i=1}^m logP(x^{(i)};\theta^{j})
 $$
也就是说，如果最大似然函数的值一直在增加，那么最终我们会达到最大似然估计得最大值。
由于
$$
L(\theta, \theta^{j}) = \sum\limits_{i=1}^m\sum\limits_{z^{(i)}}P( z^{(i)}|x^{(i)};\theta^{j}))log{P(x^{(i)}， z^{(i)};\theta)}
$$
令：
$$
H(\theta, \theta^{j}) =  \sum\limits_{i=1}^m\sum\limits_{z^{(i)}}P( z^{(i)}|x^{(i)};\theta^{j}))log{P( z^{(i)}|x^{(i)};\theta)}
$$
上面两式相减得到：
$$\sum\limits_{i=1}^m logP(x^{(i)};\theta) = L(\theta, \theta^{j}) - H(\theta, \theta^{j})
$$
在上式中分别取$\theta$为$\theta^j$和$\theta^{j + 1}$并相减得到：
$$
\sum\limits_{i=1}^m logP(x^{(i)};\theta^{j+1})  - \sum\limits_{i=1}^m logP(x^{(i)};\theta^{j}) = [L(\theta^{j+1}, \theta^{j}) - L(\theta^{j}, \theta^{j}) ] -[H(\theta^{j+1}, \theta^{j}) - H(\theta^{j}, \theta^{j}) ]
$$
要证明EM算法的收敛性，我们只需要证明上式的右边是非负的即可。
由于$\theta^{j + 1}$使得$L(\theta, \theta^{j})$极大，因此有:
$$
L(\theta^{j+1}, \theta^{j}) - L(\theta^{j}, \theta^{j})  \geq 0
$$
而对于第二部分，我们有：
$$
\begin{aligned} H(\theta^{j+1}, \theta^{j}) - H(\theta^{j}, \theta^{j})  & = \sum\limits_{i=1}^m\sum\limits_{z^{(i)}}P( z^{(i)}|x^{(i)};\theta^{j})log\frac{P( z^{(i)}|x^{(i)};\theta^{j+1})}{P( z^{(i)}|x^{(i)};\theta^j)} \\ & \leq  \sum\limits_{i=1}^mlog(\sum\limits_{z^{(i)}}P( z^{(i)}|x^{(i)};\theta^{j})\frac{P( z^{(i)}|x^{(i)};\theta^{j+1})}{P( z^{(i)}|x^{(i)};\theta^j)}) \\ & = \sum\limits_{i=1}^mlog(\sum\limits_{z^{(i)}}P( z^{(i)}|x^{(i)};\theta^{j+1})) = 0  \end{aligned}
$$
其中第（4）式用到了Jensen不等式，只不过和第二节的使用相反而已，第（5）式用到了概率分布累积为1的性质。
至此，我们得到了:$\sum\limits_{i=1}^m logP(x^{(i)};\theta^{j+1})  - \sum\limits_{i=1}^m logP(x^{(i)};\theta^{j})  \geq 0$，证明了EM算法的收敛性。
<font color = "red">从上面的推导可以看出，EM算法可以保证收敛到一个稳定点，d但是却不能保证收敛到全局极大值点，因此它是**局部最优的算法。**</font>当然如果我们的优化目标$L(\theta, \theta^{j})$是凸的，则EM算法可以保证收敛到全局最大值，这点和梯度下降法这样的迭代算法相同。
# EM算法的另一种解释
如果我们定义：
$$
J(Q,\theta) = \sum\limits_i\sum\limits_{z^{(i)}}Q_i(z^{(i)})\log \frac{P(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$
从前面的推导我们知道$L(\theta) \geq J(Q,\theta)$,EM可以看作是坐标上升法，**E步固定$\theta$，优化$Q$；M步固定$Q$，优化$\theta$。**
坐标上升法（Coordinate ascent）：
{%asset_img 6.jpg %}
图中的直线式迭代优化的路径，可以看到每一步都会向最优值前进一步，而且前进路线是平行于坐标轴的，因为每一步只优化一个变量。
这犹如在x-y坐标系中找一个曲线的极值，然而曲线函数不能直接求导，因此什么梯度下降方法就不适用了。但固定一个变量后，另外一个可以通过求导得到，因此可以使用坐标上升法，一次固定一个变量，对另外的求极值，最后逐步逼近极值。
对应到EM上，就是：**E步：固定θ，优化Q；M步：固定Q，优化θ；交替将极值推向最大。   **
# Reference
 - [https://www.cnblogs.com/huangyc/p/10123780.html](https://www.cnblogs.com/huangyc/p/10123780.html)
 - [https://www.cnblogs.com/pinard/p/6912636.html](https://www.cnblogs.com/pinard/p/6912636.html)
 - [https://blog.csdn.net/v_july_v/article/details/81708386#commentBox](https://blog.csdn.net/v_july_v/article/details/81708386#commentBox)
 - [https://blog.csdn.net/zhihua_oba/article/details/73776553](https://blog.csdn.net/zhihua_oba/article/details/73776553)
 - [http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html](http://www.cnblogs.com/jerrylead/archive/2011/04/06/2006936.html)
 - [https://www.bilibili.com/video/av31906558?p=3](https://www.bilibili.com/video/av31906558?p=3)

