---
title: TensorFlow入门
tags: Deep Learning
categories: TensorFlow
mathjax: true
copyright: true
abbrlink: 31693f5f
date: 2018-08-16 12:37:21
updated:
---
[本文部分转载自该大佬](https://blog.csdn.net/geyunfei_/article/details/78782804)
[TensorFlow中文文档](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-st6f2ez1.html)
<h3>张量(Tensor)</h3>
Tensorflow内部的计算都是基于张量的,因此我们有必要先对张量有个认识。张量是在我们熟悉的标量、向量之上定义的,详细的定义很复杂,我们可以先简单的将其理解为多维数组。
<!--more-->
```
3                                       # 这个 0 阶张量就是标量，shape=[]
[1., 2., 3.]                            # 这个 1 阶张量就是向量，shape=[3]
[[1., 2., 3.], [4., 5., 6.]]            # 这个 2 阶张量就是二维数组，shape=[2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]        # 这个 3 阶张量就是三维数组，shape=[2, 1, 3]
```

TensorFlow内部使用tf.Tensor类的实例来表示张量,每个tf.Tensor具有两个属性:
dtype Tensor存储的数据类型,可以为tf.float32,tf.int32,tf.string。。。
shape Tensor存储的多维数组中每个维度的数组元素的个数.
我们现在可以敲几行代码看一下 Tensor 。在命令终端输入 python 或者 python3 启动一个 Python 会话，然后输入下面的代码：
```

# 引入 tensorflow 模块
import tensorflow as tf

# 创建一个整型常量，即 0 阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)

# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])

# 打印上面创建的几个 Tensor
print(t0)
print(t1)
print(t2)
print(t3)
```

注意下面代码输出的shape类型:
```
>>> print(t0)
Tensor("Const:0", shape=(), dtype=int32)
>>> print(t1)
Tensor("Const_1:0", shape=(3,), dtype=float32)
>>> print(t2)
Tensor("Const_2:0", shape=(2, 2), dtype=string)
>>> print(t3)
Tensor("Const_3:0", shape=(2, 3, 1), dtype=int32)
```
<font color = "red">print一个Tensor只能打印出它的属性定义,并不能打印出他的值,要想查看一个Tensor中的值还需要经过Session运行一下</font>

```
>>> sess = tf.Session()
>>> print(sess.run(t0))
3
>>> print(sess.run(t1))
[ 3.          4.0999999   5.19999981]
>>> print(sess.run(t2))
[[b'Apple' b'Orange']
 [b'Potato' b'Tomato']]
>>> print(sess.run(t3))
[[[5]
  [6]
  [7]]

 [[4]
  [3]
  [2]]]
>>> 
```
<h3>数据流图(Dataflow Graph)</h3>
数据流是一种常见的并行计算编程模型,数据流图是由结点(nodes)和线(edges)构成的有向图:
节点(nodes) 表示计算单元，也可以是输入的起点或者输出的终点
线(edges) 表示节点之间的输入/输出关系.
在TensorFlow中每个结点都是用tf.Tensor的实例来表示的,即每个节点的输入、输出都是Tensor,如图,Tensor在Graph中流动,形象的展示TensorFlow的由来
{% asset_img 1.gif%}
<h3>Session</h3>
我们在Python中需要做一些计算操作时一般会使用NumPy，NumPy在做矩阵操作等复杂的计算的时候会使用其他语言(C/C++)来实现这些计算逻辑，来保证计算的高效性。但是频繁的在多个编程语言间切换也会有一定的耗时，如果只是单机操作这些耗时可能会忽略不计，但是如果在分布式并行计算中，计算操作可能分布在不同的CPU、GPU甚至不同的机器中，这些耗时可能会比较严重。 
TensorFlow 底层是使用C++实现，这样可以保证计算效率，并使用 tf.Session类来连接客户端程序与C++运行时。上层的Python、Java等代码用来设计、定义模型，构建的Graph，最后通过tf.Session.run()方法传递给底层执行。

<h3>构建计算图</h3>
Tensor 可以表示输入、输出的端点,还可以表示计算单元,如下的代码创建了对两个Tensor执行+操作:
```
import tensorflow as tf
# 创建两个常量节点
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
sess = tf.Session()
print(sess.run(adder))
```
输出为:
```
Tensor("add:0", shape=(), dtype=float32)
8.0
```

上面使用的tf.constant()创建的Tensor都是常量,一旦创建后其中的值就不能在改变了.但是有时我们还会需要从外部输入数据,这时可以用tf.placeholder创建占位的Tensor,占位Tensor的值可以在运行的时候输入。例如:
```
import tensorflow as tf
# 创建两个占位 Tensor 节点
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder_node = a + b
# 打印三个节点
print(a)
print(b)
print(adder)
# 运行一下，后面的 dict 参数是为占位 Tensor 提供输入数据
sess = tf.Session()
print(sess.run(adder, {a: 3, b: 4.5}))
print(sess.run(adder, {a: [1, 3], b: [2, 4]}))

```
运行结果:
```
Tensor("Placeholder:0", dtype=float32)
Tensor("Placeholder_1:0", dtype=float32)
Tensor("add:0", dtype=float32)
7.5
[ 3.  7.]
```
<h2>TensorFlow 应用实例
下面通过一个例子来了解下TensorFlow。
<h3>建立模型</h3>
如下为我们进行某项实验获得的一些实验数据：

输入	输出
1	4.8
2	8.5
3	10.4
6	21
8	25.3
我们将这些数据放到一个二维图上可以看的更直观一些，如下，这些数据在图中表现为一些离散的点：
{% asset_img 2.png %}
我们需要根据现有的这些数据归纳出一个通用模型，通过这个模型我们可以预测其他的输入值产生的输出值。假设我们现在选择一个线性模型来拟合这些数据。如果用 x 表示输入， y 表示输出，线性模型可以用下面的方程表示： $ y = Wx + b$
即使我们选择了直线模型，可以选择的模型也会有很多，如下图的三条直线都像是一种比较合理的模型，只是W和b参数不同。这时我们需要设计一个损失模型(loss model)，来评估一下哪个模型更合理一些，并找到一个最准确的模型。 
 {% asset_img 3.png %}
如下图每条黄线代表线性模型计算出来的值与实际输出值之间的差值: 
{% asset_img 4.png %}
我们用y′表示实验得到的实际输出，用下面的方程表示我们的损失模型：
$ loss = \sum_{n = 1}^N (y_n - y_n')^2 $
显然，损失模型里得到的loss越小，说明我们的线性模型越准确。
<h3>使用Tensorflow实现模型</h3>
在我们设计的模型$ y = Wx + b$中 x是我们输入的值,所以可以使用tf.placeholder 来实现。输出y可以用线性模型的输出表示,我们需要不断的改变W和b的值,来找到一个使loss最小的值。这里W和b可以用变量tf.Variable()来表示。

```
import tensorflow as tf
# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W*x + b
# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))

# 创建 Session 用来计算模型
sess = tf.Session()

```
通过tf.Variable()创建变量Tensor时需要设置一个初始值,这个初始值不能立即使用必须通过init过程进行初始化。

```
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)
```
这之后再使用print(sess.run(W))打印就可以看到我们之前赋的初始值 
```
[0.1]
```
变量初始化完之后，我们可以先用上面对W和b设置的初始值0.1和-0.1运行一下我们的线性模型看看结果：
```
print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
```
输出结果为：
```
[ 0.          0.1         0.20000002  0.5         0.69999999]
```
<h3>使用Tensorflow训练模型</h3>
Tensorflow提供了很多优化算法来帮助我们训练模型,最简单的就是Gradient Descent.
```
#创建一个梯度下降优化器,学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
# 训练10000次
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

 打印一下训练后的结果
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train , y: y_train})))
```
打印出来的训练结果如下，可以看到损失值已经很小了：
```
W: [ 2.98236108] b: [ 2.07054377] loss: 2.12941
```
### TensorFlow基本用法
#### tf.get_variable ###
获取具有这些参数的现有变量或者创建一个新的变量
```
tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE
)
  常用:
   w = tf.get_variable("w",shape = [],initializer = "")
```
####tf.nn.conv2d ###
```python
tf.nn.conv2d(
    input,#需要卷积的图像,它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape
    filter,#卷积核它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape
    strides, #步长,这是一个一维的向量，长度4
    padding,#只能是"SAME","VALID"其中之一
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```
tf.nn.conv2d(X,W,stride=[1,s,s,1],padding = 'SAME')
结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
对于给定的输入张量[batch, in_height, in_width, in_channels]和一个卷积核[filter_height, filter_width, in_channels, out_channels],它们的计算方式如下:
1.先将卷积核编程二维的[fliter_height*fliter_width*in_channels,output_channels].
2.从输入张量中提取图像块组成一个虚拟张量[batch,out_height,out_width,fliter_height*fliter_width*in_channels]
3.对于每个块,右乘filter
####tf.nn.max_pool
```python
tf.nn.max_pool(
    value,
    ksize,#每一维的窗口大小4-D
    strides,#4-D 每一维步长
    padding,# padding方式
    data_format='NHWC',
    name=None
)
#最大值池化操作,返回一个池化后的Tensor
```
tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'): given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.
### tf.nn.relu
tf.nn.relu(Z1): computes the elementwise ReLU of Z1 (which can be any shape)
###tf.contrib.layers.flatten
```language
tf.contrib.layers.flatten(
    inputs,# [batch_size,....]
    outputs_collections=None,
    scope=None
)
#Return a tensor [batch_size,k]
```
将一个输入,变为2-D的Tensor
#### tf.contrib.layers.fully_connected
```language
tf.contrib.layers.fully_connected(
    inputs,#Tensor [bathc_size,image_pixels]
    num_outputs,# 输出(神经元)的个数 [batch_size,num_outputs]
    activation_fn=tf.nn.relu, #采用的激活函数,不需要请指定为None
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
tf.contrib.layers.fully_connected(P2,6,activation_fn = None)
#输入为P2,输出为6个神经元,不需要激活函数
```
<font color = "red">tf.contrib.layers.fully_connected(),在计算图中会自动的初始化参数w,并且会训练这些参数,所以不需要你来初始化他们</font>
#### tf.nn.softmax_cross_entropy_with_logits
```language
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,#[batch_size,num_classes],神经网络的期望输出
    logits=None,#神经网络的最后一层输出
    dim=-1,#分类的个数,默认为tensor的最后一维
    name=None
)
#这个函数内部自动计算softmax函数,然后在计算代价损失,所以logits必须是未经softmax处理过的函数,否则会造成错误.
```
####tf.reduce_mean
```language
tf.reduce_mean(
    input_tensor,
    axis=None,#在哪一维求平均,默认对所有数求平均
    keepdims=None,#是否保留原来维度
    name=None,
    reduction_indices=None,
    keep_dims=None
)
```
####tf.boolean_mask
基本形式:
```python
tf.boolean_mask(tensor,mask,name='boolean_mask',axis=None)
```
参数：tensor是N维度的tensor，mask是K维度的，注意K小于等于N，name可选项也就是这个操作的名字，axis是一个0维度的int型tensor，表示的是从参数tensor的哪个axis开始mask，默认的情况下，axis=0表示从第一维度进行mask，因此K+axis小于等于N。

返回的是N-K+1维度的tensor，也就是mask为True的地方保存下来。
```language
一般来说，0<dim(mask)=K<=dim(tensor),mask的大小必须匹配参数tensor的shape的前K维度。
```
{%asset_img 5.png %}
####tf.image.non_max_suppression()
非最大值抑制函数
基本形式:
```python
nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold= 0.5)
```
boxes 是不同box的坐标,scores是不同boxes预测的分数,max_boxes是保留最大box的个数,iou_threshold是一个阈值
####tf.reduce_sum
```python
reduce_sum(
    input_tensor, #表示输入,要求和的Tensor
    axis=None, #在哪个维度上进行操作,如果缺省,默认对所有维度求和
    keep_dims=False, #是否保留原始数据的维度
    name=None,
    reduction_indices=None
)
```
####tf.subtract
```python
tf.subtract(
    x,
    y,
    name=None
)#返回x-y,支持广播
"""
x：一个 Tensor，必须是下列类型之一：half，bfloat16，float32，float64，uint8，int8，uint16，int16，int32，int64，complex64，complex128。
y：一个 Tensor，必须与 x 具有相同的类型。
name：操作的名称（可选）。
tf.subtract 返回一个Tensor 与x具有相同的类型
"""
```
####tf.transpose
```python
tf.transpose(
    a,
    perm=None,#perm参数用于指定交换后的张量的轴是原先张量的轴的位置
    name='transpose',
    conjugate=False
)
#将a进行转置，并且根据perm参数重新排列输出维度。这是对数据的维度的进行操作的形式。
```
tf.transpose() 是TensorFlow中用于矩阵，张量转置用的，除了可以交换二维矩阵的行和列之外，还可以交换张量的不同轴之间的顺序
例子如：
```python
import tensorflow as tf

raw = tf.Variable(tf.random_normal(shape=(4, 3, 2)))
transed_1 = tf.transpose(raw, perm=[1, 0, 2])
transed_2 = tf.transpose(raw, perm=[2, 0, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(raw.eval())
    print('-----------------------------')
    print(sess.run(transed_1))
    print('-----------------------------')
    print(sess.run(transed_2))
--------------------- 
```
输出如
```python
[[[-1.41665018 -1.47619307]
  [-0.26559591 -0.33706442]
  [-0.70152593  0.93126309]]

 [[ 0.67472041 -0.81008494]
  [ 1.74384487  1.2157737 ]
  [ 0.94848555  0.12342481]]

 [[-1.14412427  0.00174908]
  [ 1.09407389 -0.67949998]
  [-0.40438497  0.5193854 ]]

 [[ 0.79596692 -0.58678174]
  [ 1.16309321  0.42068651]
  [ 1.03116786 -0.69529283]]]
-----------------------------
[[[-1.41665018 -1.47619307]
  [ 0.67472041 -0.81008494]
  [-1.14412427  0.00174908]
  [ 0.79596692 -0.58678174]]

 [[-0.26559591 -0.33706442]
  [ 1.74384487  1.2157737 ]
  [ 1.09407389 -0.67949998]
  [ 1.16309321  0.42068651]]

 [[-0.70152593  0.93126309]
  [ 0.94848555  0.12342481]
  [-0.40438497  0.5193854 ]
  [ 1.03116786 -0.69529283]]]
-----------------------------
[[[-1.41665018 -0.26559591 -0.70152593]
  [ 0.67472041  1.74384487  0.94848555]
  [-1.14412427  1.09407389 -0.40438497]
  [ 0.79596692  1.16309321  1.03116786]]

 [[-1.47619307 -0.33706442  0.93126309]
  [-0.81008494  1.2157737   0.12342481]
  [ 0.00174908 -0.67949998  0.5193854 ]
  [-0.58678174  0.42068651 -0.69529283]]]
--------------------- 
```
####tf.InteractiveSession()
tf.InteractiveSession() 相较于tf.Session()使用Tensor.eval()和Operation.run() 方法代替 Session.run()
TensorFlow的前后端的连接依靠于session，使用TensorFlow程序的流程构建计算图完成之后，在session中启动运行。
InteractiveSession（）：先构建一个session，然后再构建计算图；也就是说，InteractiveSession（）能够在运行图时，插入一些计算图，比较方便。

Session（）：先构建整个计算图，然后构建session，并在session中启动已经构建好的计算图；也就是说，在会话构建之前，构建好计算图。

版权声明：本文为博主原创文章，转载请附上博文链接！