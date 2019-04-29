---
title: LSTM和GRU的简单总结
copyright: true
mathjax: true
tags: deep learning
categories: 深度学习
abbrlink: ed6ef9a3
date: 2019-04-28 20:02:29
updated:
---
先给出链接

 - [http://www.aboutyun.com/thread-21297-1-1.html](http://www.aboutyun.com/thread-21297-1-1.html)
 - [https://www.youtube.com/watch?v=T8mGfIy9dWM&list=PLqpH5_OnJdN4dIAEL3ih6ZGIrrbEh3xJL&index=25](https://www.youtube.com/watch?v=T8mGfIy9dWM&list=PLqpH5_OnJdN4dIAEL3ih6ZGIrrbEh3xJL&index=25)

 [上一篇](https://statusrank.xyz/articles/5f0e8406.html)关于RNN的文章最后，我们提到过由于梯度消失和梯度爆炸问题，使得RNN很难处理长距离的依赖。本文我们介绍两种改进后的RNN：LSTM(Long Short Term Memory Network)和Gated Recurrent Unit(GRU)。它们成功的解决了原始RNN的缺陷，成为当前比较流行的RNN，并在许多领域中成功应用。
 <!--more-->
 # LSTM
 ## LSTM的结构
 LSTM和RNN其实本质上很类似，只是在隐藏层做了“手脚”。原始的RNN的隐藏层只有一个状态，即h，它对短期的输入非常敏感。于是，LSTM就又添加了一个状态c，让它来保存长期的状态。如图:
 {% asset_img 1.png %}
  新增加的状态c，我们称为单元状态(cell state)。我们把上图按照时间维度展开得到：
  {% asset_img 3.png %}
从图中我们可以看出，在t时刻，LSTM的输入有三个：当前时刻网络的输入值$x_t$，上一时刻LSTM的输出值$h_{t-1}$，上一时刻的单元状态$c_{t-1}$。LSTM的输出有两个：当前时刻输出值$h_t$和当前时刻的单元状态$c_t$。（也可以另加一个输出$y_t$，他是由$h_t$，进行transform+activate function求得）。
 **LSTM的关键，**就是怎样控制长期状态$c$。在这里，LSTM的思路是使用三个控制开关，分别是遗忘门(forget gata),输入门(input gate)和输出门(output gate)。
 {% asset_img 4.png %}
 
## LSTM的计算
前面我们说到的三个门，其实就是三个全连接层。它的输入是一个向量，输出是一个$(0,1)$之间的实数向量(一般采用sigmoid的激活函数)。假设$W$是门的权重向量,$b$是偏置，那么门可以表示为：
$$
g(x) = \sigma(Wx+b)
$$
门的使用，就是用门的输出向量按元素乘以我们需要控制的那个向量。因为门的输出是$(0,1)$之间的实数向量，那么当门的输出为0时，任何响亮与之相乘都会得到0向量（也就是啥都不能通过）；输出为1时，任何向量都不会有任何改变（也就是啥都能通过）。因为$\sigma$通常使用sigmoid函数，其值域为$(0,1)$,所以这些门始终是**半开半闭**的。
### 遗忘门
$$
f_t = \sigma(W_f \cdot[h_{t-1},x_t] + b_f) \tag1
$$
(1)中，$W_f$是遗忘门的权重矩阵,$[h_{t-1},x_t]$表示把两个向量连接成一个更长的向量，$b_f$是遗忘门的偏置，$\sigma$是sigmoid函数。事实上，权重矩阵$W_f$都是两个矩阵拼接而成:一个是$W_{fh}$，它对应着输入项$h_{t-1}$,另一个是$W_{fx}$，它对应着输入项$x_t$。
$$
\left[\begin{matrix}
W_f
\end{matrix}
\right]\cdot 
\left[\begin{matrix}
h_{t-1} \\
x_t
\end{matrix}
\right] = 
\left[\begin{matrix}
W_{fh} & W_{fx}
\end{matrix}\right]
\left[\begin{matrix}
h_{t-1}\\
x_t
\end{matrix}\right]=W_{fh}h_{t-1} + W_{fx}x_t
$$
下图展示了遗忘门的计算：
{% asset_img 5.png %}
### 输入门
$$
\bf{i_t} = \sigma(W_i\cdot[h_{t-1},x_t] + b_i) \tag2
$$
$W_i$和$b_i$分别是输入门的权重和偏置，下图展示了输入门的计算。
{% asset_img 6.png %}
接下来，我们计算用于描述当前输入的单元状态$\hat c_t$,它是根据上一次的输入和本次输入得来:
$$
\hat c_t = tanh(W_c\cdot[h_{t-1},x_t] + b_c) \tag3
$$
下图是$\hat c_t$的计算
{% asset_img 7.png %}
**现在我们计算当前时刻的单元状态$c_t$。**它是由上一时刻的单元状态$c_{t-1}$按元素乘以遗忘门$f_t$，再用当前输入的单元状态$\hat c_t$按元素乘以输入门求和得来。
$$
c_t = f_t \odot c_{t-1} + \bf{i_t} \odot \hat c_t \tag4
$$
$\odot$表示按元素乘，下图展示了计算
{% asset_img 8.png %}
这样，我们就把LSTM关于当前的记忆$\hat c_t$和长期的记忆$c_{t-1}$组合在一起，形成了当前的长期记忆$c_t$。由于遗忘门的控制，它可以保存很久之前的信息，由于输入门的控制，它又可以避免当前无关紧要的内容进入记忆。
### 输出门
**输出门控制了长期记忆对当前输出的影响。**
$$
o_t = \sigma(W_o \cdot [h_{t-1},x_t] + b_o) \tag5
$$
下图表示输入门的计算:
{% asset_img 9.png %}
**LSTM的最终输入，是由输出门和单元状态共同决定的:**
$$
\bf{h}_t = o_t \odot tanh(c_t) \tag6
$$
下图展示了LSTM完整的计算结果
{% asset_img 10.png %}
## LSTM另一种直观结构
 下图以另一种形式直观的展示了LSTM和原始RNN的区别:
 {% asset_img 2.png %}
 长期记忆$c_t$每次的变化很小，而短期记忆$h_t$变化很快。下面我们还是谈一下LSTM中三个门，遗忘门，输入门和输出门。
在某个时刻，我们的输入有：当前的外部输入$x_t$,上一时刻的单元状态$c_{t-1}$,以及$h_{t-1}$。那么我们有如下定义:

 - 当前时刻的即时状态$z = tanh(W\cdot [h_{t-1},x_t])$
 - 遗忘门: $z^f = \sigma(W^f\cdot [h_{t-1},x_t])$
 - 输入门: $z^i = \sigma(W^i\cdot [h_{t-1},x_t])$
 - 输出门:$z^o = \sigma(W^o\cdot [h_{t-1},x_t])$
 **我们知道这四个变量是完全不同的，因为首先激活函数不同，第一个双曲正切，后三个是sigmoid。其次学到的权重矩阵$W$也不同。**
 然后我们就得到了使用这四个状态的LSTM内部结构:
 {% asset_img 1.jpg %}
 $\odot$是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。$\oplus$表示矩阵加法。从上图我们也可以看出，$c_t$确实变化很慢，因为它只是上一个时刻的状态$c_{t-1}$乘以一个遗忘门，再外加当前的即时状态乘以输入门。而$h_t$却变化很快，因为它经历了一系列的操作。另外，经验表明(有对应的文献和实验)tanh的激活函数会有更好的效果，如果将其换成别的会容易过拟合或者效果很差。
LSTM内部主要有三个阶段：

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。

具体来说是通过计算得到的  $z^f$ （f表示forget）来作为忘记门控，来控制上一个状态的  $c^{t-1}$哪些需要留哪些需要忘。

2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入$x^t$ 进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些。当前的输入内容由前面计算得到的  $z$ 表示。而选择的门控信号则是由  $z^i$ （i代表information）来进行控制。

> 将上面两步得到的结果相加，即可得到传输给下一个状态的  $c^t$  。也就是上图中的第一个公式。

3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过  $z^0$  来进行控制的。并且还对上一阶段得到的  $c^o$ 进行了放缩（通过一个tanh激活函数进行变化）。

与普通RNN类似，输出 $y_t$往往最终也是通过  $h_t$变化得到。

## LSTM的训练
请看[这篇文章的推导](http://www.aboutyun.com/thread-21297-1-1.html)。
# GRU
LSTM通过三个门来控制传输状态，展现出了不错的结果。但是也因为引入了很多内容，导致参数变多，也使得训练难度加大了很多。相比LSTM，GRU做了很多简化，能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。
**GRU对LSTM做了两大改动：**

 - 将三个门：输入门，遗忘门，输出门变为两个门：更新门(update gate)$z_t$和重置门$r_t$
 - 将单元状态$c_t$和隐状态$h_t$合并为一个，表示为$h_t$。(也就是说，从外部看,GRU的结构和原始的RNN是一样的，只是内部实现更高级)。
 ## GRU的两个门控信号
 ### reset gate
 $$
 r = \sigma(W^r\cdot[h_{t-1},x_t])
 $$
 ### update gate
 $$
 z = \sigma(W^z\cdot[h_{t-1},x_t])
 $$
有了这两个门，下面我们先直接给出GRU的网络结构:
{% asset_img 11.png %}
对于$h_{t-1}$,我们先让其通过重置门来得到"重置"之后的数据${h_{t-1}}' = h_{t-1} \odot r$。将得到的${h_{t-1}}'$与$x_t$进行拼接之后，乘上一个权重矩阵$W$,再通过激活函数tanh将数据缩放到$(-1,1)$范围内，即得到上图所说的$h'$:
$$
h' = tanh(W\cdot[{h_{t-1}}',x_t])
$$
这里的$h'$主要是包含了当前输入的$x_t$数据.有针对性地对$h'$添加到当前的隐藏状态相当于“记忆了当前时刻的状态”。(类似于LSTM的选择记忆阶段)。
图2-3中的 $\odot$ 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 $\oplus$则代表进行矩阵加法操作。
**那么记忆如何更新呢？**(也就是当前时刻的$h^t$怎么得到呢？)
$$
h_t = z \odot h_{t-1} + (1 -z )\odot h'
$$
这里的门控门的取值范围和LSTM中的一样，也是$(0,1)$。门控信号越接近$1$，代表“记忆”下来的数据越多；而越接近$0$，代表“遗忘”的越多。
**GRU相比LSTM简化的主要部分就在于此，**使用了一个update gata 来同时进行遗忘和记忆。
 - $z \odot h_{t-1}$: 表示对原本隐藏状态的选择性“遗忘”。这里的$z$就可以想象为LSTM中的遗忘门。
 - $(1-z) \dot h'$: 表示对包含当前结点信息的$h'$进行选择性"记忆"。$1-z$可以类比LSTM的输入门
 - $h_t$的更新公式和LSTM中$c_t$的更新公式极其相似，$c_t = z^f \odot c_{t-1} + z^i \odot \hat c_t$。<font color = "red">只不过这里遗忘$z$和选择$1-z$是联动的。也就是说，对于传递进来的隐信息$h_{t-1}$,我们会用$z$选择性遗忘，被遗忘的多少权重，我们会使用当前输入包含的信息$h'$用权重$1-z$对相应的部分进行弥补，以保持一种"恒定".</font>

