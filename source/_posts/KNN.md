---
title: 深度学习之图像分类-----------K最邻近算法（KNN）
tags: Deep Learning
copyright: true
mathjax: true
categories: 深度学习
abbrlink: b04eb2da
date: 2018-06-14 23:13:06
updated:
---
##邻近算法,或者说是K最邻近算法,是一个相对简单的多分类算法，其基本工作原理为:
首先我们存在一个训练集,训练集中的每个图片都存在标签（已知图片属于哪一类）.对于我们输入的没有标签的数据，我们将新数据中的每个特征与样本集合中的数据的对应特征进行比较,计算出二者之间的距离，然后记录下与新数据距离最近的K个样本，最后选择K个数据当中类别最多的那一类作为新数据的类别。

下面通过一个简单的例子说明一下：如下图，绿色圆要被决定赋予哪个类，是红色三角形还是蓝色四方形？如果K=3，由于红色三角形所占比例为2/3，绿色圆将被赋予红色三角形那个类，如果K=5，由于蓝色四方形比例为3/5，因此绿色圆被赋予蓝色四方形类。
<!-- more-->

![该图片也说明了KNN算法的结果很大程度取决于K的选择](https://img-blog.csdn.net/20180609213238608watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hvd2FyZEVtaWx5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

上面说到要计算二者之间的距离,那么距离怎么计算呢？这里距离一般使用欧式距离，或者曼哈顿距离。

欧氏距离: $d(x,y) = \sqrt{ \sum_{k=1}^n  (x_k - y_k)^2 } $

曼哈顿距离: $ d(x,y) = \sum_{i=1}^n |x_i - y_i| $

<font color = "red">那么在进行图像分类的时候,两个图片之间的距离就是每个像素点距离的和。
</font>

<h3>KNN算法的流程</h3>

(1)计算测试数据与训练数据之间的距离
(2)按照距离的递增关系进行排序
(3)找出其中距离最近的K个数据
(4)确定K个点哪个类别出现的次数最多(概率最大)
(5)返回这K个点中类别出现频率最高的作为预测数据的类别


<h3>说明:</h3>


1)KNN算法不存在训练数据的问题，他是一种懒惰学习,仅仅是把所有的数据保存下来，训练时间开销为0
2)KNN算法的优点: 简单、有效。计算时间和空间线性于训练集的规模（在一些场合不算太大）
3)缺点:计算复杂度高，空间复杂度高。存储开销大,需要把所有数据都保存起来
当样本不平衡时，如一个类的样本容量很大，而其他类样本容量很小时，有可能导致当输入一个新样本时，该样本的K个邻居中大容量类的样本占多数。


<font size = 10>代码实现：</font>

```
import numpy as np

class KNN(object):
    def __init__(self):
        pass
    
    def train(self,X,Y):
        """
        Train the classifier .For k_nearest_neighbor this is just memorizing the train data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
              consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.Y_train = Y
    def predict(self,X,k = 1,num_loops = 0):
        """
      Predict labels for test data using this classifier.
      基于该分类器，预测测试数据的标签分类。
      Inputs:
      - X: A numpy array of shape (num_test, D) containing test data consisting
           of num_test samples each of dimension D.测试数据集
      - k: The number of nearest neighbors that vote for the predicted labels.
      - num_loops: Determines which implementation to use to compute distances
        between training points and testing points.选择距离算法的实现方法
  
      Returns:
      - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].  
      """
        if num_loops == 0:
            dis = self.compute_no_loop(X)
        elif num_loops == 1:
            dis = self.compute_one_loop(X)
        elif num_loops == 2:
            dis = self.compute_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" %num_loops)
        return self.predict_label(dis,k=k)
    def compute_two_loops(self,X):
        """
        Compute the distance between each test point in X and each training point in self.X_train using a nested
        loop over both trainging and the test data.(两层for 计算距离,这里计算欧几里得距离)
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        # dis[i,j] is the Euclidean distance between the ith test data and jth training data.
        dis = np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                test_row = X[i,:] # choose this test row.
                train_row = self.X_train[j,:]
                dis[i,j] = np.sqrt(np.sum((test_row - train_row)**2))
        return dis
    def compute_one_loop(self,X):
        """
        Compute the distance between each test point in X and each trainging point in X_train using a single loop 
        over the test data.
        dis: the same as compute_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train[0]
        dis = np.zeros((num_test,num_train))
        for i in range(num_test):
            test_row = X[i,:]
            # numpy 的broadcasting 机制
            dis[i,:] = np.sqrt(np.sum(np.square(test_row - self.X_train),axis = 1))#axis=1 计算每一行的和,返回为一个列表,正好存在第i行
        return dis
    def compute_no_loop(self,X):
        """
        Compute the distance between each test point in X and each training point in X_train without using any 
        loops.        
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dis = np.zeros((num_test,num_train))
        X_square = np.square(X).sum(axis = 1) # 1 * 500
        X_train_square = np.square(self.X_train).sum(axis = 1) # 1 * 5000
        dis = np.sqrt(-2*np.dot(X,self.X_train.T) + np.matrix(X_square).T + X_train_square) # 500 * 5000
        dis = np.array(dis)
        return dis
    def predict_label(self,dis,k = 1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        """
        num_test = dis.shape[0]
        # y_pred Return the label of ith test data.
        y_pred = np.zeros(num_test) 
        for i in range(num_test):
            closest_y = []
            closest_y = self.Y_train[np.argsort(dis[i,:])[:k]] 
            # np.argsort 函数返回的是数组值从小到大的索引值(按从小到大的顺序返回每个索引)
            y_pred[i] = np.argmax(np.bincount(closest_y))
            # np.argmax 返回最大值,np.bincount()统计每一个元素出现次数的
        return y_pred
    
```

PS:上面计算两个图片之间的距离用到了三种方法,一是两层循环，一个是一层循环,还有一个是不用循环。最后两个主要是用到了numpy的广播机制。

<font size = 5>numpy的广播机制</font>
广播(broadcasting)指的是不同形状的数组之间的算数运算的执行方式。
1.数组与标量值的乘法：

```
import numpy as np
arr = np.arange(5)
arr #-> array([0, 1, 2, 3, 4])
arr * 4 #-> array([ 0,  4,  8, 12, 16])
```
在这个乘法运算中，标量值4被广播到了其他所有元素上。
2.通过减去列平均值的方式对数组每一列进行距平化处理

```
arr = np.random.randn(4,3)
arr #-> array([[ 1.83518156,  0.86096695,  0.18681254],
    #       [ 1.32276051,  0.97987486,  0.27828887],
    #       [ 0.65269467,  0.91924574, -0.71780692],
    #       [-0.05431312,  0.58711748, -1.21710134]])
arr.mean(axis=0) #-> array([ 0.93908091,  0.83680126, -0.36745171])
```
关于mean中的axis参数，个人是这么理解的： 
在numpy中，axis = 0为行轴(竖直方向),axis = 1为列轴（水平方向），指定axis表示该操作沿axis进行，得到结果将是一个shape为除去该axis的array。 
在上例中,arr.mean(axis=0)表示对arr沿着轴0(竖直方向)求均值，即求列均值。而arr含有3列,所以结果含有3个元素,这与上面的结论相符。

```
demeaned = arr - arr.mean(axis=0)
demeaned
> array([[ 0.89610065,  0.02416569,  0.55426426],
           [ 0.3836796 ,  0.1430736 ,  0.64574058],
           [-0.28638623,  0.08244448, -0.35035521],
           [-0.99339402, -0.24968378, -0.84964963]])
demeaned.mean(axis=0)
> array([ -5.55111512e-17,  -5.55111512e-17,   0.00000000e+00])
```

<font size = 5 color = "red">广播原则:

如果两个数组的后缘维度(从末尾开始算起的维度)的轴长度相符或其中一方的长度为1，则认为它们是广播兼容的。广播会在缺失维度和(或)轴长度为1的维度上进行。</font>
3.各行减去行均值

```
row_means = arr.mean(axis=1)
row_means.shape
> (4,)
arr - row_means
> ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-3d1314c7e700> in <module>()
    ----> 1 arr - row_means


    ValueError: operands could not be broadcast together with shapes (4,3) (4,) 
```
直接相减，报错，无法进行广播。回顾上面的原则，要么满足后缘维度轴长度相等，要么满足其中一方长度为1。在这个例子中，两者均不满足，所以报错。根据广播原则,较小数组的广播维必须为1。 
解决方案是为较小的数组添加一个长度为1的新轴。 
numpy提供了一种通过索引机制插入轴的特殊语法。通过特殊的np.newaxis属性以及“全”切片来插入新轴。
上面的例子中，我们希望实现二维数组各行减去行均值，我们需要你将行均值沿着水平方向进行广播，广播轴为axis=1，对arr.mean(1)添加一个新轴axis=1

```
row_means[:,np.newaxis].shape
> (4, 1)
arr - row_means[:,np.newaxis]
> array([[ 0.87419454, -0.10002007, -0.77417447],
           [ 0.46245243,  0.11956678, -0.58201921],
           [ 0.36798351,  0.63453458, -1.00251808],
           [ 0.17378588,  0.81521647, -0.98900235]])
```


