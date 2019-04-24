---
title: 简单总结下python中的数据结构
copyright: true
mathjax: true
tags: python
categories: 各种基础知识
abbrlink: 1eeb7190
date: 2019-04-24 16:44:57
updated:
---
[来源](https://docs.python.org/zh-cn/3/tutorial/datastructures.html#dictionaries)
# 序列
序列式python中最基本的数据结构。序列中的每个元素都分配一个数字-表示它的位置或索引，第一个索引是0，第二个索引是1，依此类推。python有6个序列的内置类型，最常见的就是列表，元组合和字符串三个。
序列都可以进行的操作包括**索引，切片，加，乘，检查成员。**
<!--more-->
## 列表
这里简单介绍下列表的基本操作：

 - list.append(x) , **在列表的末尾添加一个元素**
 - list.extend(iterable), **使用可迭代对象中的所有元素来扩展列表**
 - list.insert(i,x), **在给定的位置i插入一个元素, i是要插入元素的位置索引**
 - list.remove(x), **移除列表中第一个值为x的元素, 如果没有这样的元素抛出[`ValueError`](https://docs.python.org/zh-cn/3/library/exceptions.html#ValueError "ValueError") 异常。**
 - list.pop([i]), **删除列表中给定位置的元素并返回它, 如果没有给定位置, pop() 将会删除列表中的最后一个元素**。
 - list.clear(),  **删除列表中的所有元素**
 - list.index(x[,start[,end]]), **返回列表中第一个值为x的元素, 从0开始索引,如果没有这样的元素将会抛出 [`ValueError`](https://docs.python.org/zh-cn/3/library/exceptions.html#ValueError "ValueError") 异常。**
 - list.count(x), **返回元素x在列表中出现的次数**
 - list.sort(key = None, reverse = False), **对列表中的元素进行排序（参数可用于自定义排序，解释请参见 [`sorted()`](https://docs.python.org/zh-cn/3/library/functions.html#sorted "sorted")）。**
 - list.reverse(), **反转列表中的元素**
 - list.copy(), **返回一个列表的浅拷贝**
 ```python
 >>> fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
>>> fruits.count('apple')
2
>>> fruits.count('tangerine')
0
>>> fruits.index('banana')
3
>>> fruits.index('banana', 4)  # Find next banana starting a position 4
6
>>> fruits.reverse()
>>> fruits
['banana', 'apple', 'kiwi', 'banana', 'pear', 'apple', 'orange']
>>> fruits.append('grape')
>>> fruits
['banana', 'apple', 'kiwi', 'banana', 'pear', 'apple', 'orange', 'grape']
>>> fruits.sort()
>>> fruits
['apple', 'apple', 'banana', 'banana', 'grape', 'kiwi', 'orange', 'pear']
>>> fruits.pop()
'pear'
 ```
 **你可能已经注意到，像insert, remove或者sort方法，只修改列表，没有打印出返回值——它们返回默认的值None.**
### 列表作为栈使用
列表方法使得列表作为堆栈非常容易，“先进后出”。要添加一个元素到堆栈顶端，只需要list.append(x)，要从顶端取元素只需要list.pop()，不指定索引。
```python
>>> stack = [3, 4, 5]
>>> stack.append(6)
>>> stack.append(7)
>>> stack
[3, 4, 5, 6, 7]
>>> stack.pop()
7
>>> stack
[3, 4, 5, 6]
>>> stack.pop()
6
>>> stack.pop()
5
>>> stack
[3, 4]
```
### 列表作为队列使用
列表也可以作队列——先进先出。<font color = "red">然而列表作为队列却非常低效，因为从列表尾部弹出或插入元素非常高效，而从头部弹出或插入元素却需要将所有的元素前移或后移一个。</font> 若要实现一个队列，**collections.deque**, 实现了双端队列，可以快速地从两端操作。
```python
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])
```
**基本操作**:
 - append()，右边添加一个元素
 - appendleft()，左边添加一个元素
 - clear()，清空队列
 - copy()，浅拷贝
 - count()，返回指定元素出现的次数
 - index()，查找某个元素的索引位置
 - insert()，在指定位置插入元素
 - pop()，获得最右边一个元素并删除
 - popleft()，获得最左边一个元素并删除
 - remove()，删除某个元素
 - reverse()，列表元素反转


## del 语句
del 可以从列表按照给定的索引而不是值来移除一个元素，它不同于会返回一个值得pop() 方法。**del语句也可以用来从列表中移除切片或者清空整个列表**
```python
>>> a = [-1, 1, 66.25, 333, 333, 1234.5]
>>> del a[0]
>>> a
[1, 66.25, 333, 333, 1234.5]
>>> del a[2:4]
>>> a
[1, 66.25, 1234.5]
>>> del a[:]
>>> a
[]
```
del 也可以用来删除整个变量
```python
>>> del a
```
此后再引用 `a` 时会报错（直到另一个值被赋给它）。
## 元组
一个元组由几个被逗号隔开的值组成，
```python
>>> t = 12345, 54321, 'hello!'
>>> t[0]
12345
>>> t
(12345, 54321, 'hello!')
>>> # Tuples may be nested:
... u = t, (1, 2, 3, 4, 5)
>>> u
((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
>>> # Tuples are immutable:
... t[0] = 88888
Traceback (most recent call last): File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> # but they can contain mutable objects:
... v = ([1, 2, 3], [3, 2, 1])
>>> v
([1, 2, 3], [3, 2, 1])
```
如你所见，元组在输出时总是被圆括号包围的，以便正确表示嵌套元组。输入时圆括号可有可无，不过经常会是必须的（如果这个元组是一个更大的表达式的一部分）。给元组中的一个单独的元素赋值是不允许的，当然你可以创建包含可变对象的元组，例如列表。
**虽然元组看起来可能与列表很像，但他们通常是在不同的场景被使用，并且有着不同的用途。元组是不可变的(immutable)，其序列通常包含不同种类的元素。列表是可变的(mutable)，并且列表中的元素一般是同种类型的。**
**一个特殊的问题是构造包含0个或1个元素的元组。**空元祖可以直接被一对空圆括号创建，含有一个元素的元组可以通过在这个元素后添加一个逗号来构建（圆括号里只有一个值的话不够明确）。
```python
>>> empty = ()
>>> singleton = 'hello',    # <-- note trailing comma
>>> len(empty)
0
>>> len(singleton)
1
>>> singleton
('hello',)
```
语句 `t  =  12345,  54321,  'hello!'` 是 _元组打包_ 的一个例子：值 `12345`, `54321` 和 `'hello!'` 被打包进元组。其逆操作也是允许的
```python
>>> x, y, z = t
```
这被称为<font color = "red">序列解包</font>，因为解包操作的等号右侧可以是任何序列。序列解包要求等号左侧的变量数与右侧序列里包含的元素数相同。
# 优先队列
### queue.PriorityQueue的基本操作
 - Q.put（），入队列
 - Q.pop（），出队列
 - Q.queue（），查看队列中的元素
 - Q.empty（），判断队列是否为空
 #### Q.put（）入队
 put（）入队列的元素一定要是**元组**，默认情况下根据元组的第一个元素进行排序，越小优先级越低，如果第一个元素想等再根据第二个，依次类推。
 <font color = "red">入队列时可能犯的错误:
 我们自定义一个类A，将两个优先级相同的不同A的实例加入队列中。
 ```python
from queue import PriorityQueue as PQ
pq = PQ()
class A:
    def __init__(self, val):
        self.val = val
pq.put((1, A(1)))
pq.put((1, A(2))) # error:  '<' not supported between instances of 'A' and 'A'
 ```
 原因是A的实例是不可比较的，上面说过当第一个元素优先级相同时会依次往后比较，所以**当你可能向元组中存入不可比较的值，或者当你希望通过入队的顺序控制出队的顺序，你需要为元组第二个位置增加一个辅助变量，通常是一个自增变量。**
 ```python
from queue import PriorityQueue as PQ
pq = PQ()
class A:
    def __init__(self, val):
        self.val = val
index = 0
pq.put((1, index, A(1)))
index += 1
pq.put((1, index, A(2)))
```

# 集合
python也包含有集合类型，**集合是由不重复元素组成的无序的集，它的基本用法包括成员监测和消除重复元素。**集合对象也支持像联合，交集，差集，对称差分的数学运算。
花括号或 [`set()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#set "set") 函数可以用来创建集合。**注意：要创建一个空集合你只能用 `set()` 而不能用 `{}`，因为后者是创建一个空字典。**
## 常用操作

 - add（），添加元素
 - update（），也可以添加元素，且参数可以是列表，元组，字典等
 - remove（x），将元素x从集合移除，**若不存在或抛出异常**
 - discard（x），将元素x从集合移除，**不存在不会发生错误**
 - pop（），随机删除集合的一个元素
 - x in s，判断元素是否在集合中存在
```python
>>> basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
>>> print(basket)                      # show that duplicates have been removed
{'orange', 'banana', 'pear', 'apple'}
>>> 'orange' in basket                 # fast membership testing
True
>>> 'crabgrass' in basket
False

>>> # Demonstrate set operations on unique letters from two words
...
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a                                  # unique letters in a
{'a', 'r', 'b', 'c', 'd'}
>>> a - b                              # letters in a but not in b
{'r', 'd', 'b'}
>>> a | b                              # letters in a or b or both
{'a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'}
>>> a & b                              # letters in both a and b
{'a', 'c'}
>>> a ^ b                              # letters in a or b but not both
{'r', 'd', 'b', 'm', 'z', 'l'}
```
# 字典
## 基本概念
另一个非常有用的python内置数据类型是**字典**。与连续整数位索引的序列不同，字典是**关键字**为索引的，关键字可以使任意不可变类型，通常是字符串或数字。**如果一个元组只包含字符串、数字或者元组，那么这个元组也可以作为关键字。**如果元组直接或间接的包括了[可变对象]([https://statusrank.xyz/articles/e59929ab.html](https://statusrank.xyz/articles/e59929ab.html))那么就不能用作关键字。列表不能用作关键字，因为列表可以通过索引、切片或 `append()` 和 `extend()` 之类的方法来改变。
理解字典的最好方式，就是将它看做是一个键-值对的集合，**键必须是唯一的。**一对花括号可以创建一个空字典：`{}` 。另一种初始化字典的方式是在一对花括号里放置一些以逗号分隔的键值对，而这也是字典输出的方式。
字典主要的操作是使用关键字存储和解析值。也可以用 `del` 来删除一个键值对。如果你使用了一个已经存在的关键字来存储值，那么之前与这个关键字关联的值就会被遗忘。用一个不存在的键来取值则会报错。
对一个字典执行 `list(d)` 将返回包含该字典中所有键的列表，按插入次序排列 (如需其他排序，则要使用 `sorted(d)`)。要检查字典中是否存在一个特定键，可使用 [`in`](https://docs.python.org/zh-cn/3/reference/expressions.html#in) 关键字。
以下是使用字典的一些简单示例
```python
>>> tel = {'jack': 4098, 'sape': 4139}
>>> tel['guido'] = 4127
>>> tel
{'jack': 4098, 'sape': 4139, 'guido': 4127}
>>> tel['jack']
4098
>>> del tel['sape']
>>> tel['irv'] = 4127
>>> tel
{'jack': 4098, 'guido': 4127, 'irv': 4127}
>>> list(tel)
['jack', 'guido', 'irv']
>>> sorted(tel)
['guido', 'irv', 'jack']
>>> 'guido' in tel
True
>>> 'jack' not in tel
False
```
## 访问字典里的值

1. 在字典中循环时，用items（）方法可将关键字和对应的值同时取出
 ```python
 >>> knights = {'gallahad': 'the pure', 'robin': 'the brave'}
>>> for k, v in knights.items():
...     print(k, v)
...
gallahad the pure
robin the brave
 ``` 
2. 遍历key值
```python
>>> a  
{'a': '1', 'b': '2', 'c': '3'}  
>>> for key in a:  
print(key+':'+a[key])
a:1  
b:2  
c:3  
>>> for key in a.keys():  
print(key+':'+a[key])
a:1  
b:2  
c:3
```
在使用上，for key in a和 for key in a.keys():完全等价。
3. 遍历value值——dict.values（）
```python
>>> for value in a.values():  
print(value)
```

