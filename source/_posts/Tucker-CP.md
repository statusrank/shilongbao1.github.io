---
title: 张量分解——CP分解与Tucker分解详解
copyright: true
mathjax: true
tags: 张量分解
categories: 机器学习
abbrlink: 9b114ae4
date: 2018-11-07 14:38:37
updated:
---
[关于张量分解一些简单的介绍,可以看我的这篇](https://statusrank.xyz/2018/11/06/2008-11-6/)
[转载来源](https://zhuanlan.zhihu.com/p/24798389)
##张量的CP分解模型
一般而言，给定一个大小为$n_1 \times n_2 \times n_3$的张量$\mathcal{X}$，其CP分解可以写成如下形式，即

$\mathcal{X} \approx \sum_{r=1} ^{R} {A(:,r) \otimes B(:,r) \otimes C(:,r)}$

其中，矩阵A,B,C的大小分别为$n_1 \times R$, $n_2 \times R$,$ n_3 \times R$，被称为因子矩阵（factor matrices），符号“$\otimes$ ”表示张量积，对于向量而言，这里的张量积与外积是等价的。张量$\mathcal{X}$任意位置$\left(i,j,k\right)$上的元素估计值为
$$
x_{ijk} \approx \sum _{r=1} ^{R} {a_{ir}b_{jr}c_{kr}}.
$$
<!--more-->
如图: {% asset_img 1.png %}
我在我的前一篇文章也提过Rank-one Tensor的概念,即N阶的张量能以N个张量的外积来表示.可以发现这里CP分解其实就是分解成有限个Rank-one Tensor的和。
若将该逼近问题转化为一个无约束优化问题，且目标函数为残差的平方和，即
$$
f \left(A, B, C\right)=\frac{1}{2} \sum_{i=1}^{n_1} {\sum_{j=1}^{n_2} {\sum_{k=1}^{n_3} {\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right)^2}}}.
$$ 
下面我们就要使用梯度下降来优化我们的损失函数了。我们来看看如何对目标函数$f \left(A,B,C\right)$求偏导，并给出决策变量的梯度下降更新公式。
##"元素级"的梯度下降更新公式
将张量$\mathcal{X}$被观测到的元素相应的索引集合记为$\left(i,j,k\right) \in \Omega$，前面的目标函数可以改写为
$$
f \left(A,B,C\right)=\frac{1}{2} \sum_{\left(i,j,k\right) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right)^2}
$$ 
对目标函数中的决策变量$a_{ir},b_{jr},c_{kr}$求偏导数，则我们能够很容易地得到
$$
\frac{\partial f \left(A,B,C\right)}{\partial a_{ir}}=\sum_{j,k:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-b_{jr}c_{kr}\right)},
$$

$$
\frac{\partial f \left(A,B,C\right)}{\partial b_{jr}}=\sum_{i,k:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-a_{ir}c_{kr}\right)},
$$

$$
\frac{\partial f \left(A,B,C\right)}{\partial c_{kr}}=\sum_{i,j:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-a_{ir}b_{jr}\right)}
$$ 
其中，需要特殊指出的是，求和符号下面的$j,k:(i,j,k) \in \Omega$,$ i,k:(i,j,k) \in \Omega$, $i,j:(i,j,k) \in \Omega$分别表示矩阵$\mathcal{X}(i,:,:)$,$\mathcal{X}(:,j,:$),$\mathcal{X}(:,:,k)$被观测到的元素其索引构成的集合。进一步，决策变量$a_{ir},b_{jr},c_{kr}$的更新公式可以分别写成：
$$
a_{ir} \Leftarrow a_{ir}- \alpha \sum_{j,k:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-b_{jr}c_{kr}\right)},
$$

$$
b_{jr} \Leftarrow b_{jr}- \alpha \sum_{i,k:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-a_{ir}c_{kr}\right)},
$$

$$
c_{kr} \Leftarrow c_{kr} - \alpha \sum_{i,j:(i,j,k) \in \Omega}{\left(x_{ijk}-\sum_{r=1}^{R}{a_{ir}b_{jr}c_{kr}}\right) \left(-a_{ir}b_{jr}\right)}
$$ 
接下来，我们来看看在运算程序时更为有效的一种梯度下降更新策略，即“矩阵化”的梯度下降更新公式。
##"矩阵化"的梯度下降更新公式 
定义一个与张量$\mathcal{X}$大小相同的张量$\mathcal{W}$，当$(i,j,k) \in \Omega$时，$w_{ijk}=1$，否则$w_{ijk}=0$；再令$\mathcal{E}=\mathcal{X}-\sum_{r=1} ^{R} {A(:,r) \otimes B(:,r) \otimes C(:,r)}$，则决策变量A,B,C的更新公式分别为
$$
A \Leftarrow A + \alpha \left(\mathcal{W}\ast \mathcal{E}\right)_{(1)}\left(C \odot B\right),
$$

$$
B \Leftarrow B + \alpha \left(\mathcal{W}\ast \mathcal{E}\right)_{(2)}\left(C \odot A\right),
$$

$$
C \Leftarrow C + \alpha \left(\mathcal{W}\ast \mathcal{E}\right)_{(3)}\left(B \odot A\right).
$$ 
其中，运算符“$\ast $”表示点乘（element-wise multiplication），若大小均为$n_1 \times n_2 \times n_3$的张量$\mathcal{B}和\mathcal{C}$进行点乘，即$\mathcal{A}=\mathcal{B} \ast \mathcal{C}$，则$a_{ijk}=b_{ijk} \cdot c_{ijk}；$运算符“$\odot$ ”表示Khatri-Rao积，其定义可参考前面的文章介绍.
###简单的推导
以因子矩阵A的更新公式示例,令梯度$H=-\left(\mathcal{W} \ast \mathcal{S} \right)_{(1)} \left(C \odot B\right)$由于
$$
\left(\mathcal{W} \ast \mathcal{S} \right)_{(1)}=\left[ \begin{array}{ccccccc}g_{111} & \cdots & g_{1n_21} & \cdots & g_{11n_3} & \cdots & g_{1n_2n_3} \\\vdots & \ddots & \vdots & & \vdots & \ddots & \vdots \\g_{n_111} & \cdots & g_{n_1n_21} & \cdots & g_{n_11n_3} & \cdots & g_{n_1n_2n_3} \\\end{array} \right] 
$$ 
$ g_{ijk}=w_{ijk}e_{ijk},i=1,2,...,n_1,j=1,2,...,n_2,k=1,2,...,n_3.$
故矩阵$ \left(\mathcal{W} \ast \mathcal{S} \right)_{(1)}$ 的第 i 行：
$$
\left[ \begin{array}{ccccccc}w_{i11}e_{i11} & \cdots & w_{in_21}e_{in_21} & \cdots & w_{i1n_3}e_{i1n_3} & \cdots & w_{in_2n_3}e_{in_2n_3} \\\end{array} \right].
$$ 
由于$\left(C \odot B\right)=\left[ \begin{array}{ccc} c_{11}b_{11} & \cdots & c_{1R}b_{1R} \\ \vdots & \ddots & \vdots \\c_{11}b_{n_21} & \cdots & c_{1R}b_{n_2R} \\ \vdots & & \vdots \\ c_{n_31}b_{11} & \cdots & c_{n_3R}b_{1R} \\ \vdots & \ddots & \vdots \\c_{n_31}b_{n_21} & \cdots & c_{n_3R}b_{n_2R} \\\end{array} \right] $，故该矩阵的第$ r $列为
$\left[ \begin{array}{ccccccc}c_{1r}b_{1r} & \cdots & c_{1r}b_{n_2r} & \cdots & c_{n_3r}b_{1r} & \cdots & c_{n_3r}b_{n_2r} \\\end{array} \right]^T$
综上，我们可以计算出矩阵 $H $的第 $i$ 行第 $r$ 列元素为：
$$ 
H(i,r)=-\sum_{(i,j,k) \in \Omega}{w_{ijk}e_{ijk} c_{kr}b_{jr}}=-\sum_{j,k:(i,j,k) \in \Omega}{e_{ijk} b_{jr}c_{kr}} .
$$ 
从而，我们可以发现，上述“矩阵级”的更新公式表达式与“元素级”等价。
有了这些“矩阵化”的更新公式后，我们还需要考虑一个问题，即对于阶数大于3的稀疏张量如何进行CP分解呢？是否存在类似的梯度下降更新公式呢？
##延伸: 更高阶稀疏张量的CP分解
给定一个大小为$n_1 \times n_2 \times ... \times n_d$的张量$\mathcal{X}$，并将其CP分解写成如下形式：
$$
\mathcal{X} \approx \sum_{r=1} ^{R} {A^{(1)}(:,r) \otimes A^{(2)}(:,r) \otimes ... \otimes A^{(d)}(:,r)}
$$
其中，$A^{(1)},A^{(2)},...,A^{(d)}$分别是大小为$n_1 \times R,n_2 \times R,..., n_d \times R$的因子矩阵，采用梯度下降，令$\mathcal{E}=\mathcal{X}-\sum_{r=1} ^{R} {A^{(1)}(:,r) \otimes A^{(2)}(:,r) \otimes ... \otimes A^{(d)}(:,r)}$，则决策变量$A^{(1)},A^{(2)},...,A^{(d)}$的更新公式可以分别写成
$$
A^{(1)} \Leftarrow A^{(1)}+\alpha \left(\mathcal{W} \ast \mathcal{E} \right)_{(1)} \left(A^{(d)} \odot A^{(d-1)} \odot ... \odot A^{(3)}\odot A^{(2)}\right),
$$

$$
A^{(2)} \Leftarrow A^{(2)}+\alpha \left(\mathcal{W} \ast \mathcal{E} \right)_{(2)} \left(A^{(d)} \odot A^{(d-1)} \odot ... \odot A^{(3)} \odot A^{(1)} \right),
$$

$$
... ...
$$

$$
A^{(d)} \Leftarrow A^{(d)}+\alpha \left(\mathcal{W} \ast \mathcal{E} \right)_{(d)} \left(A^{(d-1)} \odot A^{(d-2)} \odot ... \odot A^{(2)} \odot A^{(1)} \right)
$$ 
** 不过，需要额外说明的是，CP分解中的R一般表示张量的秩（tensor rank），由于对于张量的秩的求解往往是一个NP-hard问题，所以读者在设计实验时要预先给定R的值**。
##CP分解的优化
上面我们已经有了CP分解的形式了,那么我们怎么去优化所有参数,使得分解后跟分解前尽可能的相似呢？
我们比较常用的方法就是使用交替最小二乘法(Alternating Least-Square，ALS)去优化.我们还是以上面的三维张量为例:
{%asset_img 2.png %}

我们这里的优化就是:先固定B,C优化A,接着固定A,C优化B,在接着固定A,B优化C，继续迭代知道达到迭代条件为止。
对于参数A，完整的优化过程如下图所示：
{%asset_img 3.png %}
参数B,C同理，然后不断迭代即可。
N维Tensor完整的交替最小二乘法优化CP分解过程:
{%asset_img 4.png %}

##Tucker分解
以第3阶张量为例,假设${\mathcal{X}}$是大小为$n_1\times n_2\times n_3￥的张量，进行Tucker分解后的表达式可以写成(根据前面提到过的高阶奇异值分解)
$$
{\mathcal{X}}\approx {\mathcal{G}}\times _{1}U\times _{2}V\times _{3}W
$$
其中，张量${\mathcal{G}}$的大小为$r_{1}\times r_2\times r_3$，也称为核心张量（core tensor），矩阵U的大小为$n_1 \times r_1$，矩阵V的大小为$n_2 \times r_2$，矩阵W的大小为$n_3 \times r_3$。这条数学表达式可能对很多只熟悉矩阵分解但却没有接触过张量分解的读者来说有点摸不着头脑，但完全不用担心，实际上，这条数学表达式等价于如下这个简洁的表达式，即
$$

x_{ijk}\approx \sum _{m=1}^{r_1}{\sum _{n=1}^{r_2}{\sum_{l=1}^{r_3}}}{\left(g_{mnl}\cdot u_{im}\cdot v_{jn} \cdot w_{kl}\right)}
$$
与上面逼近问题的目标函数类似，我们在这里可以很轻松地写出逼近问题的优化模型：
$$
\min J=\frac{1}{2} \sum_{(i,j,k)\in S}{e_{ijk}^{2}} =\frac{1}{2} \sum_{\left( i,j,k \right)\in S }{\left( x_{ijk}-\sum _{m=1}^{r_1}{\sum _{n=1}^{r_2}{\sum_{l=1}^{r_3}}}{\left(g_{mnl}\cdot u_{im}\cdot v_{jn} \cdot w_{kl}\right)} \right) ^2} 
$$ 
对目标函数J中的$u_{im}、v_{jn}、w_{kl}$和$g_{mnl}$求偏导数，得
$$
\frac {\partial J}{\partial u_{im}}=-\sum_{j,k:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{n=1}^{r_2}{\sum _{l=1}^{r_3}{\left(g_{mnl}\cdot v_{jn}\cdot w_{kl}\right)}}\right)}；
$$

$$
\frac {\partial J}{\partial v_{jn}}=-\sum_{i,k:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{m=1}^{r_1}{\sum _{l=1}^{r_3}{\left(g_{mnl}\cdot u_{im}\cdot w_{kl}\right)}}\right)}；
$$

$$
\frac {\partial J}{\partial w_{kl}}=-\sum_{i,j:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{m=1}^{r_1}{\sum _{n=1}^{r_2}{\left(g_{mnl}\cdot u_{im}\cdot v_{jn}\right)}}\right)}；
$$

$$
\frac {\partial J}{\partial g_{mnl}}=-\sum_{(i,j,k)\in S}{e_{ijk}\cdot u_{im}\cdot v_{jn}\cdot w_{kl}}.
$$ 
再根据梯度下降方法，$u_{im}$、$v_{jn}$、$w_{kl}$和$g_{mnl}$在每次迭代过程中的更新公式为
$$
u_{im}\Leftarrow u_{im}+\alpha \sum_{j,k:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{n=1}^{r_2}{\sum _{l=1}^{r_3}{\left(g_{mnl}\cdot v_{jn}\cdot w_{kl}\right)}}\right)}；
$$
$$
v_{jn}\Leftarrow v_{jn}+\alpha \sum_{i,k:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{m=1}^{r_1}{\sum _{l=1}^{r_3}{\left(g_{mnl}\cdot u_{im}\cdot w_{kl}\right)}}\right)}；
$$

$$
w_{kl}\Leftarrow w_{kl}+\alpha \sum_{i,j:(i,j,k)\in S}{e_{ijk}\cdot \left(\sum_{m=1}^{r_1}{\sum _{n=1}^{r_2}{\left(g_{mnl}\cdot u_{im}\cdot v_{jn}\right)}}\right)}；
$$

$$
g_{mnl}\Leftarrow g_{mnl}+\alpha \sum_{(i,j,k)\in S}{e_{ijk}\cdot u_{im}\cdot v_{jn}\cdot w_{kl}}.
$$ 
其中，更新公式中的求和项下标$j,k:(i,j,k)\in S、i,k:(i,j,k)\in S$和$i,j:(i,j,k)\in S$分别表示矩阵${\mathcal{X}}\left( i,:,: \right)$ 、${\mathcal{X}}\left( :,j,: \right)$ 和${\mathcal{X}}\left( :,:,k \right) $上所有非零元素的位置索引构成的集合。至此，我们已经完成了用梯度下降来进行Tucker张量分解的大部分工作，有了这些更新公式，我们就可以很轻松地编写程序来对稀疏张量进行分解，并最终达到对稀疏张量中缺失数据的填补。
##Tucker分解与CP分解的比较
**CP分解可以认为是Tucker分解的特例**.
我们先写出两种分解的数学表达式:
Tucker分解 ${\mathcal{X}}\approx {\mathcal{G}}\times _{1}U\times _{2}V\times _{3}W$
CP分解：${\mathcal{X}}\approx \sum_{p=1}^{r}{\lambda_{p}F\left(:,p \right) \circ S\left(:,p\right)\circ T\left(:,p\right)}$
其中，张量${\mathcal{X}}$的大小为$n_1\times n_2\times n_3$，在Tucker分解中，核心张量${\mathcal{G}}$的大小为$r_{1}\times r_2\times r_3$，矩阵U、V、W的大小分别是$n_1 \times r_1$、$n_2 \times r_2$、$n_3 \times r_3$；在CP分解中，矩阵$F、S、$T大小分别为$n_1\times r$、$n_2\times r$、$n_3\times r$，运算符号“$\circ $”表示外积（outer product）.如向量$a=\left( 1,2 \right) ^{T}$ ，向量$b=\left( 3,4 \right) ^{T}$ ，则$a\circ b=ab^{T}=\left[ \begin{array}{cc} 3 & 4 \\ 6 & 8 \\ \end{array} \right]$

张量${\mathcal{X}}$在位置索引$\left( i,j,k \right)$ 上对应的元素为

Tucker分解：$x_{ijk}\approx \sum _{m=1}^{r_1}{\sum _{n=1}^{r_2}{\sum_{l=1}^{r_3}}}{\left(g_{mnl}\cdot u_{im}\cdot v_{jn} \cdot w_{kl}\right)}$
CP分解：$x_{ijk}\approx \sum_{p=1}^{r}{\lambda_{p}\cdot f_{ip}\cdot s_{jp}\cdot t_{kp}}$
从这两个数学表达式不难看出，CP分解中$\lambda_{p}$构成的向量替换了Tucker分解中$g_{mnl}$构成的核心张量（如图1所示），即CP分解是Tucker分解的特例，CP分解过程相对于Tucker分解更为简便，但CP分解中r的选取会遇到很多复杂的问题，如张量的秩的求解是一个NP-hard问题等，这里不做深入讨论。
{%asset_img 5.png %}