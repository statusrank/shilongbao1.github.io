---
title: Remove-Nth-Node-From-End-of-List(two points)
copyright: true
mathjax: true
tags: two points
categories: LeetCode
abbrlink: 3946e2a4
date: 2019-04-19 09:44:26
updated:
---
[传送门](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
## 题意:
给你一个链表,删除链表的倒数第n个结点。（要求只遍历一次链表）
<!--more-->
## 思路:
如果可以遍历链表两次，那就很简单了 第一遍确立链表的长度$L$，然后删除倒数倒数第n个等价于删除第$L-n+1$个结点。

只遍历一次链表,我们可以使用双指针first和second,然后始终维护first和second的之间的gap为$n+1$ ,当first指向None时,那么second指向的下一个结点就是要删除的结点。
```python
# Definition for singly-linked list.

# class ListNode:

# def __init__(self, x):

# self.val = x

# self.next = None

  

class  Solution:

def  removeNthFromEnd(self, head, n):

root = ListNode(0)

root.next = head

first = root

second = root

gap =  0

while first !=  None:

while gap <= n and first !=  None:

first = first.next

gap +=  1

if first !=  None:

second = second.next

first = first.next

second.next = second.next.next

return root.next
```

