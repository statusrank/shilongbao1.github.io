---
title: Python 装饰器总结
copyright: true
mathjax: true
tags:
  - 装饰器
categories:
  - 各种基础知识
abbrlink: 54e65a77
date: 2018-12-11 12:23:35
updated:
---
[学习来源](https://blog.csdn.net/u010358168/article/details/77773199)
[学习来源](https://www.cnblogs.com/huchong/p/7725564.html)
python 装饰器(decorator)是在程序开发中经常用到的功能,合理利用装饰器能让我们的程序如虎添翼
<!--more-->
##作用域
在python中,作用域分为两种: 全局作用域和局部作用域。
**全局作用域**是定义在文件级别的变量,函数名。
**局部作用域**是定义在函数内部。
**关于作用域我们要理解两点:**
1.在全局不能访问到局部定义的变量
2.在局部能够访问到全局定义的变量,但是不能修改全局定义的变量(当然有方法可以改)
##高级函数
**我们知道函数名其实就是指向一段内存空间的地址,既然是地址,那么我们就可以利用这种特性:**
1.函数名可以作为一个值
2.函数名可以作为返回值
3.函数名可以作为一个参数
满足以上三个条件中的一个,我们称之为高级函数.
##闭包函数
**闭包函数必须满足两个条件:**
1.函数内部定义的函数
2.包含对外部作用域而非全局作用域的引用
我们先来看一个闭包函数的例子:
```python
def outer():
    x = 1
    def inner():
        print("x=%s" %x)
        print("inner func excuted")
    inner()
    print("outer func excuted")

outer()

#####输出结果#########
x=1
inner func excuted
outer func excuted
```
##装饰器
对闭包函数有一定的了解,我们就可以来看看什么是装饰器了。
<font color = "red">装饰器:外部函数传入被装饰函数名,内部函数返回装饰函数名
特点: 1.不修改被装饰函数的调用方式
2.不修改被装饰函数的源代码</font>
###装饰器问题引入
假如现在在一个公司，有A B C三个业务部门，还有S一个基础服务部门，目前呢，S部门提供了两个函数，供其他部门调用，函数如下：
```python
def f1(): 
	print('f1 called') 
def f2(): 
	print('f2 called')
```
在初期，其他部门这样调用是没有问题的，随着公司业务的发展，现在S部门需要对函数调用假如权限验证，如果有权限的话，才能进行调用，否则调用失败。考虑一下，如果是我们，该怎么做呢？

一种比较完美的解决方案就是使用装饰器函数:
```python
def w1(func):
    def inner():
        print('...验证权限...')
        func()

    return inner


@w1
def f1():
    print('f1 called')


@w1
def f2():
    print('f2 called')


f1()
f2()

```
输出结果为:
```python
...验证权限... 
f1 called
 ...验证权限... 
 f2 called
```
我们可以看到,在调用函数f1,f2时成功的先进行了权限验证,然后才进行调用。这里就是用到了装饰器,通过定义一个闭包函数w1.在我们调用函数上通过@w1，就完成了对函数的装饰。
##装饰器的原理
让我们来看一下我们的装饰器函数w1,首先它的参数是接受一个函数func(上面我们说过函数名其实就是指向一段内存空间的地址),w1内部我们又定义了一个函数inner,在inner函数内部我们先进行权限验证,并在权限验证结束后调用func，w1的返回值为inner.

我们再来看一下@w1这句话是什么意思呢?档python解释器执行到这句话的时候,会去调用w1函数,同时将被装饰函数名作为参数传入w1。根据上面的分析w1是将内部inner函数作为返回值,并赋值给f1,此时**f1已经不是未加装饰的原函数了,而是w1.inner()的函数指向的内存地址**,其实也就是说在函数f1上加@w1，就等价于执行 f1 = w1(f1).
这样在接下来调用f1的时候,其实执行的就是w1.inner函数,就会先进行权限验证再调用f1了。这就完成了对一个函数的装饰。
###多个装饰器执行流程和装饰结果
当有两个或两个以上装饰器装饰一个函数时,执行过程是什么样的呢？
```python
def makeBold(fun):
    print('----a----')

    def inner():
        print('----1----')
        return '<b>' + fun() + '</b>'

    return inner


def makeItalic(fun):
    print('----b----')

    def inner():
        print('----2----')
        return '<i>' + fun() + '</i>'

    return inner


@makeBold
@makeItalic
def test():
    print('----c----')
    print('----3----')
    return 'hello python decorator'


ret = test()
print(ret)
```

输出结果:
```python
----b----
----a----  
----1----  
----2----  
----c----  
----3---- 
<b><i>hello  python  decorator</i></b>
```
**通过上面这个例子我们可以发现python的装饰器是从下往上进行装饰,但是执行时却是从上往下执行**
我们可以来分析下:
1.当python的解释器执行到@makeBold时,解释器知道是要对下面的函数进行装饰,但是它发现它下面紧跟的还是一个装饰器@makeItalic，这时候@makeBold就先暂停执行装饰,而是接着执行@makeItalic的装饰,把test传入装饰器函数,打印出'b',此时test函数名的地址指向的是返回inner的地址。@makeItalic装饰完成之后,再由@makeBold来执行装饰,因此打印了'a'。
2.当我们调用test执行时,此时test指向的makeBold.inner函数的地址了,所以先执行此函数,打印出1,然后调用func,根据1的分析结果可知此时调用的makeItalic.inner,所以打印出2,然后再调用func就是真正的test函数了,打印出c,3.最后一层层返回。

###对有参数的函数进行装饰
在使用中，有的函数可能会带有参数，那么这种如何处理呢？利用python的可变参数轻松实现装饰带参数的函数。
```python
def w_add(func):
    def inner(*args, **kwargs):
        print('add inner called')
        func(*args, **kwargs)

    return inner


@w_add
def add(a, b):
    print('%d + %d = %d' % (a, b, a + b))


@w_add
def add2(a, b, c):
    print('%d + %d + %d = %d' % (a, b, c, a + b + c))


add(2, 4)
add2(2, 4, 6)
```
输出:
```python
add inner called 
2 + 4 = 6  
add inner called 
2 + 4 + 6 = 12
```

