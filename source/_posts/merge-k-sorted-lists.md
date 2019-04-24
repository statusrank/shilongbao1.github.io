---
title: LeetCode 23. Merge k Sorted Lists
copyright: true
mathjax: true
abbrlink: '1e66386'
date: 2019-04-24 16:13:11
tags: 
categories: LeetCode
updated:
---
[传送门](https://leetcode.com/problems/merge-k-sorted-lists/)
## 题意
将k个有序的列表合并成一个有序的列表，并分析时间复杂度和空间复杂度。
<!--more-->
## 思路
### 法一
首先我们想到的就是先两个列表合并为一个，然后再挑一个和该列表进行合并，以此类推，直到最后合并为一个列表。
```python
class Solution:
    def mergeKLists(self, lists):
        head = cur = ListNode(0)
        for l in lists:
            if l:
                if head.next is None:         
                    cur.next = l
                    cur = cur.next
                else:
                    pre, cur = head, head.next
                    while cur and l:
                        while cur and l.val >= cur.val:
                            pre = cur
                            cur = cur.next
                        temp = l.next
                        l.next = cur
                        pre.next = l
                        pre = pre.next
                        l = temp
                    if cur is None and l:
                        pre.next = l
        return head.next
```
**复杂度分析：**
时间复杂度:
假设有$k$个列表，总长度为$N$，那么平均每个列表长度为$\frac{N}{k}$，k个列表合并为一个共需要$k-1$合并。第一次将两个合并一个时间复杂度为$O(2\cdot \frac{N}{k})$，此时列表长度为$\frac{2N}{k}$，再和第三个合并，复杂度为$O(3\cdot\frac{N}{k})$，以此类推，总的时间复杂度为$O(n) = \sum_{i = 1}^{k-1} (i + 1)\cdot \frac{N}{k} = O(kN)$。
空间复杂度: $O(1)$
### 法二
我们其实可以发现这里的序列合并问题，其实完全可以采用分治的思想。或者可以说，我们采用类似于归并排序的思想，即: 先将k个元素分为两两合并，然后将合并获得的序列再进行两两合并。我们用一个图来描述这个过程:
{% asset_img 1.png %}
```python
class Solution:
    def mergeKLists(self, lists):
        numb = len(lists)
        iter = 1
        while iter < numb:
            for i in range(0, numb - iter, iter * 2):
                lists[i] = self.merge2Lists(lists[i],lists[i + iter])
            iter *= 2
        return lists[0] if numb > 0 else []
    def merge2Lists(self, l1, l2):
        if l1 == None: return l2
        if l2 == None: return l1
        h1,h2,cur = l1,l2,ListNode(-1)
        head = cur
        while h1 and h2:
            if h1.val > h2.val:
                cur.next = h2
                cur = h2
                h2 = h2.next
            else:
                cur.next = h1
                cur = h1
                h1 = h1.next
        if h1:
            cur.next = h1
        if h2:
            cur.next = h2
        return head.next
```
**复杂度分析:**
时间复杂度:
将k个列表，每次二分每次二分，直到最后两两列表进行合并，该二叉树的深度最多$\log_2k$.在每一层，都相当于将总长度为$N$的列表合并了一次，所以总的复杂度为$O(N \log k)$。
空间复杂度: $O(1)$，代码中未使用递归
### 法三
最简单的方法就是先将k个列表不管顺序的合并为长度为$N$的列表，再用快排等高效的排序方法进行排序即可。
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists):
        _list = []
        head = cur = ListNode(0)
        for l in lists:
            while l:
                _list.append(l.val)
                l = l.next
        for l in sorted(_list):
            cur.next = ListNode(l)
            cur = cur.next
        return head.next

```
**复杂度分析:**
时间复杂度:
合并为整个列表，复杂度为$O(N)$，进行快排的复杂度为$O(N\log N)$，最后再遍历为一个链表，复杂度为$O(N)$，所以时间复杂度为$O(N\log N)$。
空间复杂度为: $O(N)$，链表$O(N)$，列表$O(N)$
### 法四
我们还可以从每个序列的头开始，每次比较这k个结点哪个小，把最小的合并到序列中，指针后移，再进行比较。。
{% asset_img 2.png %}
{% asset_img 3.png %}
```python
class Solution:
    def mergeKLists(self, lists):
        head = cur = ListNode(0)
        _list = []
        for l in lists:
            if l:
                _list.append((l.val,l))
        while len(_list):
            _id, _min = 0, _list[0][0]
            for i in range(len(_list)):
                if _list[i][0] < _min:
                    _min = _list[i][0]
                    _id = i
            cur.next = ListNode(_min)
            cur = cur.next
            if _list[_id][1].next is not None:
                _list.append((_list[_id][1].next.val, _list[_id][1].next))
            del _list[_id]
        return head.next
```
**复杂度分析:**
时间复杂度:
每找到一个最小的结点合并到序列中，就需要比较$k-1$次，一共有$N$个结点，所以复杂度为$O(kN)$.
空间复杂度: 创建一个新链表——$O(N)$

### 法五
在方法四中，每次比较k个结点的结点的大小关系有点过于费时，我们可以维护一个大小为$k$的优先队列(或堆，[关于优先队列](https://statusrank.xyz/articles/1eeb7190.html)），这样每次就可以快速地找出当前最小的结点。
```python
from queue import PriorityQueue
class Solution:
    def mergeKLists(self, lists):
        head = cur = ListNode(0)
        que = PriorityQueue()
        index = 0
        for l in lists:
            if l:
                que.put((l.val, index, l))
                index += 1
        while not que.empty():
            val, _, node = que.get()
            cur.next = ListNode(val)
            cur = cur.next
            if node.next:
                que.put((node.next.val, index, node.next))
                index += 1
        return head.next
```
**复杂度分析**：
时间复杂度:
大小为k的优先队列，每次插入和比较需要$O(\log k)$，但是找到最小元素是$O(1)$的，一共有$N$个结点，总的时间复杂度为$O(N\log k)$。
空间复杂度: 
创建一个新链表，复杂度为$O(N)$；创建一个大小为$k$的优先队列，复杂度为$O(k)$，总的复杂度为$O(N)$。

