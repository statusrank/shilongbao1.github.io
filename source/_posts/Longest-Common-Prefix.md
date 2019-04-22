---
title: LeetCode Longest Common Prefix
copyright: true
mathjax: true
tags:
  - 字典树
  - 二分
  - 分治
categories: LeetCode
abbrlink: 52563db7
date: 2019-01-14 19:19:03
updated:
---
## [传送门](https://leetcode.com/problems/longest-common-prefix/)
## 题意:
题意很简单,就是要你找出n个串的最长公共前缀
<!--more-->
## 思路:
这个题其实是因为这个题有很多种解法,而自己一时间并没有想到那么多(虽然复杂度基本没多大差别,但是从多个不同角度思考还是不错的)
### 第一种
最简单的就是我们直接枚举一个串的前缀,然后判断剩下n-1个串是否满足.
```python
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
        	return ""
        _str,_max,flag = strs[0],0,0
        for i in range(1,len(_str) + 1):
        	for s in strs:
        		if len(s) < i or s[:i] != _str[:i]:
        			flag = 1
        			break
        	if flag == 1:
        		break
        	_max = max(_max,i)
        return _str[:_max]
```
可以看出这种方法最坏的时间复杂度是我们有n个串他们完全相同,设每个串长度为m,那么为$O(m*n)$
### 第二种
其次我们还可以用分治来做.
我们可以把判断$LCP(s_1,s_2,s_j,s_{j + 1},...s_n)$变为 $LCP(LCP(s_1,s_2,s_j),LCP(s_{j + 1},...s_n))$以此类推.
```python
class Solution:
	def longestCommon(self,strs,l,r):
		if l == r:
			return strs[l]
		mid = (l + r) // 2
		leftstrs = self.longestCommon(strs,l,mid)
		rightstrs = self.longestCommon(strs,mid + 1,r)
		return self.Commonleftright(leftstrs,rightstrs)
	def Commonleftright(self,leftstrs,rightstrs):
		_l = min(len(leftstrs),len(rightstrs))
		_max = -1
		for i in range(_l):
			if leftstrs[i] != rightstrs[i]:
				break
			_max = i
		return leftstrs[0:_max + 1]
	def longestCommonPrefix(self, strs):
		if len(strs) == 0:
			return ""
		return self.longestCommon(strs,0,len(strs) - 1)
```
但是因为分治采用递归会使得我们的空间复杂度变为$O(mlog(n))$

### 第三种
还有一种方法就是二分.我们去二分最长公共前缀的长度,然后带入验证.(显然最长公共前缀的长度满足二分的性质).
这种方法的时间复杂度就变为$O(Slog(m)),S = nm$
```python
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
        	return ""
        _str,_max,flag = strs[0],0,0
        l,r = 1,len(_str) + 1
        while l <= r:
        	flag = 0
        	i = (l + r) // 2
        	for s in strs:
        		if len(s) < i or s[:i] != _str[:i]:
        			flag = 1
        			break
        	if flag == 1:
        		r = i - 1
        	else:
        		_max = max(_max,i)
        		l = i + 1
        return _str[:_max]
```
### 第四种
字典树,这是字典树解决多个字符串的公共前缀,或者有多少个不重复的单词,是否有其他单词以某个单词为前缀的基本问题.
**[字典树](https://blog.csdn.net/HowardEmily/article/details/76278332)**
时间复杂度: 建树$O(S)$,查询$O(m)$
空间复杂度$O(S)$
```python
class Trie:
	"""docstring for Trie"""
	def __init__(self):
		self.root = {}
		self.end = -1
	def insert(self,strs):
		Node = self.root
		for c in strs:
			if c not in Node:
				Node[c] = {}
			Node = Node[c]
		Node[self.end] = True
	def search(self):
		_len,Node = -1,self.root
		while True:
			if len(Node) == 1 and self.end not in Node:
				_len += 1
				Node = list(Node.values())[0]
			else:
				return _len
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
        	return ""
        T = Trie()
        for i in strs:
        	T.insert(i)
        return strs[0][:T.search() + 1]
```
