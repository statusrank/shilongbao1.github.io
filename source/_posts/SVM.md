---
title: 浅谈支持向量机(SVM)
copyright: true
mathjax: true
tags:
  - SVM
  - 机器学习
categories: 机器学习
abbrlink: 9d4ca24c
date: 2018-11-23 21:31:48
updated:
---
本文学习总结自以下,都是很好的资料!
[JULY](https://blog.csdn.net/v_july_v/article/details/7624837)
[数据挖掘十大算法详解](https://wizardforcel.gitbooks.io/dm-algo-top10/content/svm-5.html)
[刘建平](http://www.cnblogs.com/pinard/p/6111471.html)
[刘晓坤](https://www.jiqizhixin.com/articles/2017-10-08)
SVM是一个二元分类算法,线性分类和非线性分类都支持,其学习策略便是**间隔最大化,**最终可以转化为一个凸二次规划问题求解。经过演进,现在也可以支持多元分类,同时经过扩展也能应用于回归问题。
<!--more-->
##初探SVM
###一个简单的例子
下面举个简单的例子。如下图所示，现在有一个二维平面，平面上有两种不同的数据，分别用圈和叉表示。由于这些数据是线性可分的，所以可以用一条直线将这两类数据分开，这条直线就相当于一个超平面，超平面一边的数据点所对应的y全是-1 ，另一边所对应的y全是1。
{%asset_img 1.png %}
这个超平面可以用分类函数$f(x) = w^Tx+b$表示,当$f(x) = 0$,x便是位于超平面上的点,而$f(x) > 0$对应$ y = 1$,$f(x)< 0$对应$y = -1$的点。如图:
{%asset_img 2.png %}
换言之，在进行分类的时候，遇到一个新的数据点x，将x代入$f(x)$ 中，如果$f(x) < 0$则将x的类别赋为-1，如果$f(x) > 0$则将x的类别赋为1。
接下来的问题是如何确定这个超平面呢？{%asset_img 3.png %}
比如对于上图,a和b都可以作为分类的超平面,但最优的超平面只有一个,最优分类平面使间隔最大化。
从直观上来讲,距离样本太近的超平面不是最优的,因为这样的直线对噪声敏感度高,泛化性误差差。因此我们的目标是找到一个超平面,离所有点的距离最远。由此SVM算法的实质是找出一个能够将某个值最大化的超平面,这个值就是超平面距离所有训练样本的最小距离。这个最小间隔SVM术语叫做间隔(margin)。
###函数间隔与几何间隔
在超平面$ w^Tx+b = 0$确定的情况下,$|w^Tx+b|$能够表示点x到超平面距离的远近,而通过观察$w^Tx + b$的富豪与类标记y的符号是否一致可以判断分类是否正确,所以可以用$ y(w^Tx+b)$的正负来判断分类的正确性。与此我们便引出了函数间隔的概念。
定义函数间隔为:
$ \hat r = y(w^Tx + b) = yf(x)$.而超平面(w,b)关于训练集中所有样本点$(x_i,y_i)$的函数间隔的最小值变为超平面关于训练数据集的函数间隔$ \hat r = min  \hat r(i = 1,...n)$
但是这样定义函数间隔有问题,即如果成比例的改变$w,b$的值(如同时扩大2倍)则函数间隔的值$f(x)$却变为2倍(虽然此时超平面没有改变)，所以只有函数间隔远远不够。
事实上,我们可以通过对法向量w加些约束条件,从而引出真正定义点到超平面的距离—**几何距离**。
假定对于一个点 x，令其垂直投影到超平面上的对应点为 $x_0$ ，$w$ 是垂直于超平面的一个向量，$ r$为样本x到超平面的距离，如下图所示：
{%asset_img 4.png %}
根据平面几何知识有:
{%asset_img 5.png %}
 其中$||w||$为$w$的二阶范数（范数是一个类似于模的表示长度的概念），$\frac{w}{||w||}$是单位向量（一个向量除以它的模称之为单位向量）。
 又由于$x_0$ 是超平面上的点，满足 $f(x_0)=0$，代入超平面的方程$w^Tx+b$可得$w^Tx_0 + b= 0$,即$W^Tx_0 = -b$
随即对上式两边同时左乘$w^T$即可求出$r$：{%asset_img 6.png %}
为了得到$r$的绝对值,令其乘上对应类别的绝对值y，我们即可得出几何间隔:
{%asset_img 7.png %}
从上述函数间隔和几何间隔的定义可以看出：几何间隔就是函数间隔除以$||w||$，而且函数间隔y*(wx+b) = y*f(x)实际上就是|f(x)|，只是人为定义的一个间隔度量，而几何间隔$\frac{|f(x)|}{||w||}$才是直观上的点到超平面的距离。
###最大间隔分类器
我们在上面的关于线性分类器的例子当中提出了间隔(margin)的概念。为了使得分类的确信度尽可能高,我们需要让所选择的超平面去最大化这个margin,这个间隔就是下图中的一半。{%asset_img 7.png %}
通过前面的分析可以直观的看出,函数间隔不适合用来最大化间隔,因为在超平面固定后可以等比例的将$w,b$进行缩放,这样可以使得$f(x)$的值任意大,也就是函数间隔值$\hat r$可以在超平面保持不变的前提下任意大。但是几何间隔我们除上了$||w||$使得在对$w,b$进行缩放时几何间隔$\hat r$是不变的,它只随着超平面而变动,因此这更适合做我们的margin。**也就是说这里的最大化间隔其实是最大化我们的几何间隔。**
用数学式子表示为:
$$
max \;\; \gamma = \frac{y(w^Tx + b)}{||w||_2}  \;\; s.t \;\; y_i(w^Tx_i + b) = \gamma^{'(i)} \geq \gamma^{'} (i =1,2,...m)
$$
一般我们都取函数间隔$ \hat r = 1$(这里函数间隔的值对我们的优化问题无影响,且令函数间隔为1有利于后面的优化和推导),这样我们的优化函数为:
$$
max \;\; \frac{1}{||w||_2}  \;\; s.t \;\; y_i(w^Tx_i + b)  \geq 1 (i =1,2,...m)
$$
也就是说我们要在约束条件$y_i(w^Tx_i + b)  \geq 1 (i =1,2,...m)$下,最大化$\frac{1)}{||w||_2}$。
由于$\frac{1)}{||w||_2}$的最大化等同于$\frac{1}{2}||w||_2^2$的最小化。这样SVM的优化函数等价于:
$$

min \;\; \frac{1}{2}||w||_2^2  \;\; s.t \;\; y_i(w^Tx_i + b)  \geq 1 (i =1,2,...m)
$$
因为现在的目标函数是二次的,约束条件是线性的,所以它是一个凸二次规划问题。这个问题可以用现成QP优化包进行求解。一言以蔽之：在一定的约束条件下，目标最优，损失最小。
此外，由于这个问题的特殊结构，还可以通过拉格朗日对偶性（Lagrange Duality）变换到对偶变量 (dual variable) 的优化问题，即通过求解与原问题等价的对偶问题（dual problem）得到原始问题的最优解，这就是线性可分条件下支持向量机的对偶算法，这样做的优点在于：一者对偶问题往往更容易求解；二者可以自然的引入核函数，进而推广到非线性分类问题。
###拉格朗日对偶
前面说过,我们就是要最小化:
$$
min \;\; \frac{1}{2}||w||_2^2  \;\; s.t \;\; y_i(w^Tx_i + b)  \geq 1 (i =1,2,...m)
$$
这是一个拉格朗日优化问题,可以通过拉格朗日乘数法得到最优超平面的权重向量$W$和偏置$b$.
$$
L(w,b,\alpha) = \frac{1}{2}||w||_2^2 - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1] \;
$$
其中$\alpha_i \geq 0$
我们看下面这个式子:
{% asset_img 9.png %}
上面是要求解的目标函数及其条件,我们可以得到拉格朗日公式为:
{%asset_img 10.png %}
在这里$\beta_i$被称为拉格朗日算子。然后分别对$w,\beta$进行求偏导,使得偏导数为0解出$w,\beta$:
{%asset_img 11.png %}
下面我们将要产生一个既有等式又有不等式条件限制的式子，我们可以叫做原始优化问题，这里简单介绍下拉格朗日对偶的原理。如下式子：
{%asset_img 12.png %}
上述式子有两个条件（等式条件和不等式条件）
由此我们定义一般化的拉格朗日公式
{%asset_img 13.png %}
这里$\alpha_i,\beta_i$都是拉格朗日算子。(不要忘了我们要求解的是最小值)。
设如下等式:
{%asset_img 14.png %}
这里的P代表primal。我们设如下约束条件（primal constraints）：
{%asset_img 15.png %}
如果条件不全部满足的话，我们总可以调整$\alpha_i,\beta_i$使最大值出现正无穷,即会出现如下情况:
{%asset_img 16.png %}
因此我们可以得出如下式子：
{%asset_img 17.png %}
这样我们原来要求的min $f(w)$可以转换成求:{%asset_img 18.png %}同时我们设{%asset_img 19.png %}
$\theta_D(\alpha,\beta)$将问题转化为先求拉格朗日关于$w$的最小值,这个时候就把$\alpha,\beta$看成常量,之后再求最大值。
如下:
{%asset_img 20.png %}
** 这个问题就是原问题的对偶问题**。相对于原问题我们只是更换了min和max的顺序,而一般更换顺序的结果是$\max \min X  \leq min \max X$.但是在满足某些情况下二者相等,很幸运在这里我们满足这个条件,所以二者相等。
而我们上面所说的条件就是KTT条件。
{%asset_img 21.png %}
###SVM优化
由拉格朗日对偶,我们的优化目标从
$$
\underbrace{min}_{w,b}\; \underbrace{max}_{\alpha_i \geq 0} L(w,b,\alpha)
$$
变为了:
$$
\underbrace{max}_{\alpha_i \geq 0} \;\underbrace{min}_{w,b}\;  L(w,b,\alpha)
$$
从上式中我们可以先求$w,b$的最小值,接着再求拉格朗日乘子$\alpha$的最大值。
首先我们来求$L(w,b,\alpha)$基于$w,b$的最小值,即$\underbrace{min}_{w,b}\;  L(w,b,\alpha)$.这个极值我们可以通过对$w,b$分别求偏导数得到:
$$
\frac{\partial L}{\partial w} = 0 \;\Rightarrow w = \sum\limits_{i=1}^{m}\alpha_iy_ix_i
$$
$$
\frac{\partial L}{\partial b} = 0 \;\Rightarrow \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
从上两式子可以看出，我们已经求得了$w$和$\alpha$的关系，只要我们后面接着能够求出优化函数极大化对应的$\alpha$，就可以求出我们的$w$了，至于$b$，由于上两式已经没有$b$，所以最后的$b$可以有多个。
我们已经求出了$w$和$\alpha$的关系就可以带入$L(w,b,\alpha)$消去$w$。我们定义:
$$
\psi(\alpha) = \underbrace{min}_{w,b}\;  L(w,b,\alpha)
$$
现在我们来看将$w$替换为$\alpha$后的优化函数$\psi(\alpha)$的表达式:
$$
\begin{align} \psi(\alpha) & =  \frac{1}{2}||w||_2^2 - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1] \\& = \frac{1}{2}w^Tw-\sum\limits_{i=1}^{m}\alpha_iy_iw^Tx_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i -\sum\limits_{i=1}^{m}\alpha_iy_iw^Tx_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i  \\& = - \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i  \\& = - \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}(\sum\limits_{i=1}^{m}\alpha_iy_ix_i)^T(\sum\limits_{i=1}^{m}\alpha_iy_ix_i) - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i  \\& = -\frac{1}{2}\sum\limits_{i=1}^{m}\alpha_iy_ix_i^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}\sum\limits_{i=1}^{m}\alpha_iy_ix_i^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_iy_ix_i^T\alpha_jy_jx_j + \sum\limits_{i=1}^{m}\alpha_i \\& = \sum\limits_{i=1}^{m}\alpha_i  - \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j  \end{align}
$$
{%asset_img 22.png %}
从上面可以看出通过对$w,b$的极小化以后,我们的优化函数仅仅只有$\alpha$向量做参数。只要我们能够极大化$\psi(\alpha)$就可以求出此时对应的$\alpha$进而求出$w,b$.
对$\psi(\alpha)$极大化的数学表达式如下:
$$
\underbrace{max}_{\alpha} -\frac{1}{2}\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{m}\alpha_i\alpha_jy_iy_j(x_i \bullet x_j) + \sum\limits_{i=1}^{m} \alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
\alpha_i \geq 0  \; i=1,2,...m
$$
可以添加符号变为等价的极小化问题,如下:
$$
\underbrace{min}_{\alpha} \frac{1}{2}\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{m}\alpha_i\alpha_jy_iy_j(x_i \bullet x_j) -  \sum\limits_{i=1}^{m} \alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
\alpha_i \geq 0  \; i=1,2,...m
$$
只要我们可以求出上式的极小化时对应的向量$\alpha$向量就可以求出$w,b$了。具体怎么极小化上式的$\alpha$一般要用到SMO算法,这个后面会讲。先假设我们通过SMO算法得到了对应极小值的$\alpha$为$\alpha^{*}$.那么根据我们的:$w = \sum\limits_{i=1}^{m}\alpha_iy_ix_i$就可以求出对应的$w$
$$
w^{*} = \sum\limits_{i=1}^{m}\alpha_i^{*}y_ix_i
$$
求$b$稍微复杂一些,注意到对于任意支持向量$(x_s,y_s)$都有:
$$
y_s(w^Tx_s+b) = y_s(\sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s+b) = 1
$$

假设我们有$S$个支持向量,则对应我们求出S个$b^{'}$，理论上它们都可以作为最终的结果。但是我们一般采用一种更健壮的办法,即求出所有支持向量所对应的$b_s^{'}$，然后将其平均值作为最后的结果。注意到对于严格线性可分的SVM，b的值是有唯一解的，也就是这里求出的所有$b^{'}$都是一样的，这里我们仍然这么写是为了和后面加入软间隔后的SVM的算法描述一致。
根据KTT条件中的对偶互补条件$\alpha_{i}^{'}(y_i(w^Tx_i + b) - 1) = 0$,如果$\alpha_i > 0$则有$y_i(w^Tx_i + b) =1$即点在支持向量上，否则如果$\alpha_i=0$则有$y_i(w^Tx_i + b) \geq 1$，即样本在支持向量上或者已经被正确分类。
**这显示出支持向量机的一个重要性质:训练完成后,大部分的训练样本都不需要保留,最终模型只与支持向量有关。**
###线性可分SVM算法的过程
输入是线性可分的m个样本${(x_1,y_1), (x_2,y_2), ..., (x_m,y_m),}$其中x为n维特征向量，y为二元输出，值为1，或者-1.
输出是分离超平面的参数$w^{'}$和$b^{'}$和分类决策函数。
算法过程如下:
1.构造约束优化问题
$$
\underbrace{min}_{\alpha} \frac{1}{2}\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{m}\alpha_i\alpha_jy_iy_j(x_i \bullet x_j) -  \sum\limits_{i=1}^{m} \alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
\alpha_i \geq 0  \; i=1,2,...m
$$
2.用SMO算法求出上式最小时对应的$\alpha^{'}$向量。
3.计算$w^{'} = \sum\limits_{i=1}^{m}\alpha_i^{'}y_ix_i$
4.找出所有的S个支持向量,即满足$\alpha_s > 0对应的样本(x_s,y_s)$通过$y_s(\sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s+b) = 1$ 计算出每个支持向量$(x_s,y_s)$对应的$b_s^{'}$，计算出这些$b_s^{'} = y_s - \sum\limits_{i=1}^{s}\alpha_iy_ix_i^Tx_s$,所有的$b_s^{'}$对应的平均值即为最终的$b^{'} = \frac{1}{S}\sum\limits_{i=1}^{S}b_s^{'}$这样最终的分类超平面'：$w^{'} \bullet x + b^{'} = 0$，最终的分类决策函数为：$f(x) = sign(w^{'} \bullet x + b^{'})$
##非线性可分SVM
###容错性与软间隔
我们在上一节看到的是一个线性可分数据的简单例子,但现实中的数据通常是很凌乱的,你也很可能遇到一些不能正确线性分类的例子。如图:
{%asset_img 23.png %}
很显然使用一个线性分类器通常都无法完美的将标签分离,但是我们也不想将其完全抛弃不用,毕竟除了几个错点他基本上能很好的解决问题。那么我们如何解决呢？SVM引入了软间隔最大化的方法来解决。
所谓的软间隔是相对于硬间隔来说的,我们可以认为上面的SVM分类方法属于硬间隔最大化。
回顾下硬间隔最大化的条件:
$$
min\;\; \frac{1}{2}||w||_2^2  \;\; s.t \;\; y_i(w^Tx_i + b)  \geq 1 (i =1,2,...m)
$$
接着我们再看如何可以软间隔最大化呢？
SVM对训练集里面的每个样本$(x_I,y_i)$引入了一个松弛变量$\xi_i \geq 0$，使函数间隔加上松弛变量大于等于1,也就是说:
$$
y_i(w\bullet x_i +b) \geq 1- \xi_i
$$
对比硬间隔最大化,可以看到我们对样本到超平面的函数距离要求放松了,之前一定要大于等于1，现在只需要加上一个大于等于0的松弛变量$\xi_i$满足大于等于1就可以了。当然松弛变量不能白加,也是需要有成本的，每一个松弛变量$\xi_i$对应一个代价$\xi_i$，这就得到了我们的软间隔最大化SVM的学习条件:
$$
min\;\; \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i
$$
$$
s.t.  \;\; y_i(w^Tx_i + b)  \geq 1 - \xi_i \;\;(i =1,2,...m)
$$
$$
\xi_i \geq 0 \;\;(i =1,2,...m)
$$
这里$C>0$为乘法参数,可以理解为我们一般回归和分类问题正则化时候的参数。参数C其实表示了我们要在两种结果中权衡:1)拥有很宽的间隔。2）精确分离训练数据。C的值越大,意味着在训练数据中允许的误差越小。这是一个权衡的过程,如果想要更好的分类训练数据,那么代价就是间隔会更宽。一下几个图展示了在不同的C值分类器和间隔的变化:
{%asset_img 24.png %}
注意决策边界随 C 值增大而倾斜的方式。在更大的 C 值中，它尝试将右下角的红点尽可能的分离出来。但也许我们并不希望在测试数据中也这么做。第一张图中 $C=0.01$，看起来更好的抓住了普遍的趋势，虽然跟更大的 C 值相比，它牺牲了精确性。
###线性分类SVM软间隔最大化及函数优化
和线性可分SVM的优化方式类似，我们首先将软间隔最大化的约束问题用拉格朗日函数转化为无约束问题如下：
$$
L(w,b,\xi,\alpha,\mu) = \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}\mu_i\xi_i
$$
其中$\mu_i \geq 0, \alpha_i \geq 0$均为拉格朗日系数。
也就是说，我们现在要优化的目标函数是：
$$
\underbrace{min}_{w,b,\xi}\; \underbrace{max}_{\alpha_i \geq 0, \mu_i \geq 0,} L(w,b,\alpha, \xi,\mu)
$$
这个优化也满足KTT条件,也就是说我们可以通过拉格朗日对偶将我们的优化问题转化为等价的对偶问题来解决:
$$
\underbrace{max}_{\alpha_i \geq 0, \mu_i \geq 0,} \; \underbrace{min}_{w,b,\xi}\; L(w,b,\alpha, \xi,\mu)
$$
我们可以先求优化函数对于$w,b,\xi$的最小值,接着再求拉格朗日乘子$\alpha,\mu$的最大值。
对于最小值我们可以通过求偏导得到:
$$
\frac{\partial L}{\partial w} = 0 \;\Rightarrow w = \sum\limits_{i=1}^{m}\alpha_iy_ix_i
$$
$$
\frac{\partial L}{\partial b} = 0 \;\Rightarrow \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
\frac{\partial L}{\partial \xi} = 0 \;\Rightarrow C- \alpha_i - \mu_i = 0
$$
好了,现在我们可以利用上面的三个石子取消除$w,b$了。
$$
\begin{align} L(w,b,\xi,\alpha,\mu) & = \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}\mu_i\xi_i 　\\&= \frac{1}{2}||w||_2^2 - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] + \sum\limits_{i=1}^{m}\alpha_i\xi_i \\& = \frac{1}{2}||w||_2^2 - \sum\limits_{i=1}^{m}\alpha_i[y_i(w^Tx_i + b) - 1] \\& = \frac{1}{2}w^Tw-\sum\limits_{i=1}^{m}\alpha_iy_iw^Tx_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i -\sum\limits_{i=1}^{m}\alpha_iy_iw^Tx_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = - \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - \sum\limits_{i=1}^{m}\alpha_iy_ib + \sum\limits_{i=1}^{m}\alpha_i \\& = - \frac{1}{2}w^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}(\sum\limits_{i=1}^{m}\alpha_iy_ix_i)^T(\sum\limits_{i=1}^{m}\alpha_iy_ix_i) - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}\sum\limits_{i=1}^{m}\alpha_iy_ix_i^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i - b\sum\limits_{i=1}^{m}\alpha_iy_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}\sum\limits_{i=1}^{m}\alpha_iy_ix_i^T\sum\limits_{i=1}^{m}\alpha_iy_ix_i + \sum\limits_{i=1}^{m}\alpha_i \\& = -\frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_iy_ix_i^T\alpha_jy_jx_j + \sum\limits_{i=1}^{m}\alpha_i \\& = \sum\limits_{i=1}^{m}\alpha_i - \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j \end{align}
$$
{%asset_img 25.png %}
可以发现,这个式子和我们线性可分SVM的一样,唯一不一样的就是约束条件。
$$
\underbrace{ max }_{\alpha} \sum\limits_{i=1}^{m}\alpha_i - \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
C- \alpha_i - \mu_i = 0
$$
$$
\alpha_i \geq 0 \;(i =1,2,...,m)
$$
$$
\mu_i \geq 0 \;(i =1,2,...,m)
$$
对于$C- \alpha_i - \mu_i = 0 ， \alpha_i \geq 0 ，\mu_i \geq 0$可以消除$\mu_i$只留下$\alpha_i$也就是$0 \leq \alpha_i \leq C$，同时将优化目标函数变号，求极小值，如下：
$$
\underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j - \sum\limits_{i=1}^{m}\alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
0 \leq \alpha_i \leq C
$$
这就是软间隔最大化时的线性可分SVM的优化目标形式，和上一篇的硬间隔最大化的线性可分SVM相比，我们仅仅是多了一个约束条件$0 \leq \alpha_i \leq C$我们依然可以通过SMO算法来求上式极小化时对应的$\alpha$向量就可以求出$w$和$b$了。
###软间隔最大化时的支持向量
在硬间隔最大化时支持向量比较简单,就满足$y_i(w^Tx_i + b) -1 =0$即可。根据KTT条件中的对偶互补条件$\alpha_{i}^{'}(y_i(w^Tx_i + b) - 1) = 0$，如果$\alpha_{i}^{'}>0$则有$y_i(w^Tx_i + b) =1$即点在支持向量上,否则$\alpha_{i}^{'}=0$则有$y_i(w^Tx_i + b) \geq 1$即样本在支持向量上或已经被正确分类。
在软间隔最大化时稍微复杂一些,因为我们为每个样本$(x_i,y_i)$引入了松弛变量$\xi_i$.我们从下图来研究软间隔最大化时支持向量的情况，第i个点到对应类别支持向量的距离为$\frac{\xi_i}{||w||_2}$
{%asset_img 26.png %}
根据软间隔最大化时KKT条件中的对偶互补条件

$\alpha_{i}^{'}(y_i(w^Tx_i + b) - 1 + \xi_i^{'}) = 0$我们有:
1)如果$\alpha = 0$那么$y_i(w^Tx_i + b) - 1 \geq 0$，即样本点在间隔边界或者已经被正确分类。
2）如果$0 < \alpha < C$,根据KTT条件 $\mu_i \xi_i = 0$ 以及$C - \alpha_i - \mu_i = 0$ 我们得到$ \mu_i> 0$,则$\xi_i = 0 ,\;\; y_i(w^Tx_i + b) - 1 =  0$,即在间隔边界上是支持向量。
3)如果$\alpha = C$,说明这是一个可能比较异常的点，需要检查此时$\xi_i$
  i)如果$0 \leq \xi_i \leq 1$那么点被正确分类，但是却在超平面和自己类别的间隔边界之间。如图中的样本2和4.
  ii)如果$\xi_i =1$那么点在分离超平面上，无法被正确分类。
  iii)如果$\xi_i >1$那么点在超平面的另一侧，也就是说，这个点不能被正常分类。如图中的样本1和3.
###软间隔最大化的线性可分SVM的算法过程
{%asset_img 27.png %}

###核函数
我们已经介绍过支持向量机如何处理完美或者接近完美线性可分数据，那对于那些明确的非线性可分数据，SVM 又是怎么处理的呢？毕竟有很多现实世界的数据都是这一类型的。当然，寻找一个分离超平面已经行不通了，这反而突出了 SVMs 对这种任务有多擅长。对于非线性可分的情况,SVM处理方法是选择一个核函数m通过将数据映射到高维空间来解决在原始空间线性不可分的问题。
[关于核函数](https://statusrank.xyz/2018/11/02/Kernel/)
 具体来说，在线性不可分的情况下，支持向量机首先在低维空间中完成计算，然后通过核函数将输入空间映射到高维特征空间，最终在高维特征空间中构造出最优分离超平面，从而把平面上本身不好分的非线性数据分开。如图所示，一堆数据在二维空间无法划分，从而映射到三维空间里划分：
 {%asset_img 28.png %}
回顾线性可分SVM的优化目标函数:
$$
\underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i \bullet x_j - \sum\limits_{i=1}^{m}\alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
0 \leq \alpha_i \leq C
$$
注意到上式低维特征仅仅以内积$x_i \bullet x_j$的形式出现,如果我们定义一个低维空间到高维空间的映射,将所有特征映射到一个更高的维度，让数据线性可分,我们就可以继续按照上面的方法来优化目标函数了,并求出分离超平面和决策函数。也就是说现在我们的SVM优化目标函数变成:
$$
\underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_j\phi(x_i) \bullet \phi(x_j) - \sum\limits_{i=1}^{m}\alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
0 \leq \alpha_i \leq C
$$
可以看到，和线性可分SVM的优化目标函数的区别仅仅是将内积$x_i \bullet x_j$替换为$\phi(x_i) \bullet \phi(x_j)$
看起来似乎这样我们就已经完美解决了线性不可分SVM的问题了，但是事实是不是这样呢？我们看看，假如是一个2维特征的数据，我们可以将其映射到5维来做特征的内积，如果原始空间是三维，可以映射到到19维空间，似乎还可以处理。但是如果我们的低维特征是100个维度，1000个维度呢？那么我们要将其映射到超级高的维度来计算特征的内积。这时候映射成的高维维度是爆炸性增长的，这个计算量实在是太大了，而且如果遇到无穷维的情况，就根本无从计算了。
而我们上面的博客有说到,**核函数解决的就是将数据从低维映射到高维的简化内积计算的一种方法**。
假设$\phi$是一个从低维的输入空间$\chi$欧式空间的子集或者离散集合）到高维的希尔伯特空间的$\mathcal{H}$映射。那么如果存在函数$K(x,z)$，对于任意$x, z \in \chi$都有：
$$
K(x, z) = \phi(x_i) \bullet \phi(x_j)
$$
那么我们就称$K(x, z)$为核函数。
仔细观察上式可以发现，$K(x, z)$的计算是在低维特征空间来计算的，它避免了在刚才我们提到了在高维维度空间计算内积的恐怖计算量。也就是说，我们可以好好享受在高维特征空间线性可分的红利，却避免了高维特征空间恐怖的内积计算量。
**我们总结一下非线性可分时核函数的引入过程:**
我们遇到线性不可分的样例时,常用的做法是把样例特征映射到高维空间中取,但是遇到线性不可分的样例一律映射到高维空间,那么这个维度大小是会很高的。此时,核函数就体现出它的价值了，核函数的价值在于它虽然也是将特征进行从低维到高维的转换，但核函数好在它在低维上进行计算，而将实质上的分类效果（利用了内积）表现在了高维上，这样避免了直接在高维空间中的复杂计算，真正解决了SVM线性不可分的问题。

##分类SVM算法总结
引入核函数后,我们的SVM算法才算是比较完整了。现在我们做一个总结,不再区分是否线性可分(因为引入核函数以及软间隔,我们的SVM有很大的容错性了)。
输入是m个样本${(x_1,y_1), (x_2,y_2), ..., (x_m,y_m),}$其中x为n维特征向量。y为二元输出，值为1，或者-1。
输出是分离超平面的参数$w^{*},b^{*}$和分类决策函数。
**算法过程如下：**：
1）选择适当的核函数$K(x,z)$和一个惩罚系数$C > 0$, 构造约束优化问题：
$$
\underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum\limits_{i=1}^{m}\alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
0 \leq \alpha_i \leq C
$$
2)用SMO算法求出上式最小时对应的$\alpha^{'}$
3) 得到$w^{'} = \sum\limits_{i=1}^{m}\alpha_i^{'}y_i\phi(x_i)$ 此处可以不直接显式的计算$w^{'}$
4)找出所有的S个支持向量,即满足$0 < \alpha_s < C$对应的样本$(x_s,y_s)$通过$y_s(\sum\limits_{i=1}^{m}\alpha_iy_iK(x_i,x_s)+b) = 1$计算出每个支持向量$(x_s,y_s)$对应的$b_s^{'}$.计算出这些$b_s^{'} = y_s - \sum\limits_{i=1}^{m}\alpha_iy_iK(x_i,x_s)$，所有的$b_s^{'}$对应的平均值即为最终的$b^{'} = \frac{1}{S}\sum\limits_{i=1}^{S}b_s^{'}$。
这样**最终的分类超平面**为：$\sum\limits_{i=1}^{m}\alpha_i^{'}y_iK(x, x_i)+ b^{'} = 0$,最终的分类决策函数为$f(x) = sign(\sum\limits_{i=1}^{m}\alpha_i^{'}y_iK(x, x_i)+ b^{'})$
至此，我们的分类SVM算是总结完毕.
##SMO算法原理
在前面我们的优化目标函数最终都是一个关于$\alpha$向量的函数。而怎么极小化这个函数求出对应的向量进而分离出我们的超平面呢？
首先我们回顾下我们的优化目标函数:
$$
\underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum\limits_{i=1}^{m}\alpha_i
$$
$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$
$$
0 \leq \alpha_i \leq C
$$
我们的解要满足的KKT条件的对偶互补条件为
$$
\alpha_{i}^{*}(y_i(w^Tx_i + b) - 1 + \xi_i^{*}) = 0
$$
根据这个KKT条件的对偶互补条件，我们有：
$$
\alpha_{i}^{*} = 0 \Rightarrow y_i(w^{*} \bullet \phi(x_i) + b) \geq 1
$$
$$
0 <\alpha_{i}^{*} < C  \Rightarrow y_i(w^{*} \bullet \phi(x_i) + b) = 1
$$
$$
\alpha_{i}^{*}= C \Rightarrow y_i(w^{*} \bullet \phi(x_i) + b) \leq 1
$$
由于$w^{'} = \sum\limits_{j=1}^{m}\alpha_j^{'}y_j\phi(x_j)$，我们令$g(x) = w^{'} \bullet \phi(x) + b =\sum\limits_{j=1}^{m}\alpha_j^{'}y_jK(x, x_j)+ b^{'}$,则有:
$$
\alpha_{i}^{*} = 0 \Rightarrow y_ig(x_i) \geq 1
$$
$$
0 < \alpha_{i}^{*} < C  \Rightarrow y_ig(x_i)  = 1
$$
$$
\alpha_{i}^{*}= C \Rightarrow y_ig(x_i)  \leq 1
$$
###SMO算法的基本思想
上面这个优化式子比较复杂,里面有m个变量组成的向量$\alpha$需要在目标函数极小化的时候求出。直接优化是很难的。SMO算法采用了一种启发式的方法。它每次只优化两个变量,将其他的变量都视为常数。由于$\sum\limits_{i=1}^{m}\alpha_iy_i = 0$。加入将$\alpha_3, \alpha_4, ..., \alpha_m$固定,那么$\alpha_1, \alpha_2$之间的关系也确定了。这样SMO算法将一个复杂的优化算法转化为一个比较简单的两个变量优化问题。
为了后面表示方便，我们定义$K_{ij} = \phi(x_i) \bullet \phi(x_j)$。
由于$\alpha_3, \alpha_4, ..., \alpha_m$都成了常量,虽有的常量我们都从目标函数去除,这样我们上一节的目标优化函数变成:
$$
\;\underbrace{ min }_{\alpha_1, \alpha_1} \frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 +y_1y_2K_{12}\alpha_1 \alpha_2 -(\alpha_1 + \alpha_2) +y_1\alpha_1\sum\limits_{i=3}^{m}y_i\alpha_iK_{i1} + y_2\alpha_2\sum\limits_{i=3}^{m}y_i\alpha_iK_{i2}
$$
$$
s.t. \;\;\alpha_1y_1 +  \alpha_2y_2 = -\sum\limits_{i=3}^{m}y_i\alpha_i = \varsigma
$$
$$
0 \leq \alpha_i \leq C \;\; i =1,2
$$
[关于SMO算法原理](http://www.cnblogs.com/pinard/p/6111471.html)