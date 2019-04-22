---
title: Longest Palindromic Substring(马拉车)
copyright: true
mathjax: true
tags:
  - Dp
  - Manacher
categories: LeetCode
abbrlink: c253f0f2
date: 2019-01-04 20:52:21
updated:
---
[传送门](https://leetcode.com/problems/longest-palindromic-substring/)
##题意:
给你一个字符串s,长度不超过1000,找出最长的回文串.
<!--more-->
##思路:
###暴力
最简单的想法就是我们可以直接$O(n^2)$直接找出所有的子串,然后$O(n)$去判断是否回文,同时维护最大回文串即可.
###枚举中心
一种比较好的方法就是我们可以以某个字符为中心向两边扩展所能达到的回文串的长度,但是需要注意的是回文串分为奇数和偶数,所以我们需要分别枚举奇数长度和偶数长度的.时间复杂度为$O(n^2)$.
```python
class Solution:
	def findPalindrome(self,s,i,j):
		while i >= 0 and j < len(s) and s[i] == s[j]:
			i -= 1
			j += 1
		return s[i + 1:j]
	def longestPalindrome(self, s):
		_max = ""
		for i in range(len(s)):
			_str = self.findPalindrome(s,i,i)
			if len(_str) > len(_max):
				_max = _str
		for i in range(len(s) - 1):
			_str = self.findPalindrome(s,i,i + 1)
			if len(_str) > len(_max):
				_max = _str 
		return _max
```
###动态规划
我们需要发现的是对于字符串"ababa",如果我们已经知道"bab"是回文串了,那么"ababa"也是回文串了,所以我们不需要再重新计算一遍了.
设$dp[i][j]$表示从字符串$i$到字符串$j$的回文串长度,如果为$0$则表示不是回文串.
那么有: 
$dp[i][i] = 1$
如果$s[i] == s[i + 1],dp[i][i + 1] = 2$
如果$s[i] == s[j]$,则 $dp[i][j] = dp[i + 1][j - 1] + 2$
复杂度也是$O(n^2)$
```python
class Solution:
	def longestPalindrome(self, s):
		_len = len(s)
		l,r = 0,0
		dp = [[0 for i in range(_len)] for j in range(_len)]
		for i in range(_len):
			for j in range(i):
				if s[i] == s[j]:
					if (i - j) == 1:
						dp[j][i] = 2
					elif dp[j + 1][i - 1]:
						dp[j][i] = dp[j + 1][i - 1] + 2
				if dp[j][i] > (r - l + 1):
					l,r = j,i
			dp[i][i] = 1
		return s[l:r + 1]
```

###Manacher算法
Manacher算法可以在$O(n)$时间内求出所有的最长回文串.
首先为了避免上述第二种方法我们所说的存在回文串奇数和偶数的问题,我们在每个字符串相邻位置添加未出现过的字符比如'#',这样新的字符串长度就是$2n-1$,这样我们就只需要考虑奇数的问题.如图:
{% asset_img 1.png %}
下面介绍马拉车算法的主要思想:
$p[i]$表示以字符s[i]为中心的回文串的半径(也可以理解为以s[i]为中心的回文串到最右端和最左端的距离).对于上图中的例子,我们可以得到如下:
{% asset_img 2.png %}
可以观察到$p[i]-1$即为该回文串在原串s中的长度.
根据上图我们可以发现p数组的一部分其实是对称的,比如上面的"#a#a#a#"他们的p就是关于中间'a'对称的,这样对于这些重复的部分我们就没必要重复计算了.
这就是manacher算法的核心,关于$p[i]$数组的计算:
首先我们从左往右依次计算$p[i]$,当计算$p[i]$时,$p[j]$(0 <= j < i)已经计算完毕.设$P$为之前回文串的右端点的最大值,$Po$为最大时的中心点($P = Po + p[Po]$),那么有:
(1)对于当前i,如果有$i < P$,我们可以找到i关于$Po$的对称点$j = 2Po - i$,得到$p[j]$,如果$p[j] < P - i$ 我们可以得到$p[i] = p[j]$如图:
{% asset_img 3.png %}
这个图就说明了如果**我们已经找到了以$Po$为中心的长度为$2p[Po] - 1$的回文串,如果$p[j] < P - i$就说明以j为中心的回文串一定在Po的内部,那么以其对称点i的同理,就不需要计算了.**
如果$p[j] >= P - i$,说明以i为中心的回文串可能会延伸到$P$之外,我们只知道当前的长度为$p[j]$,能否继续变大未知,需要我们匹配了.即从$P+1$开始,并对$Po$和$P$进行更新.
{% asset_img 4.png %}
(2)如果$i >= P$这时候只能老老实实的匹配了,并更新$Po$和$P$.
{% asset_img 5.png %}
**时间复杂度:**
可以发现我们是否需要进行匹配是取决于$i$和$P$的关系的,对于已经小于P的点我们是不需要匹配的,也就是说每个字符只会被匹配一次,即使我们将字符串变成了长度无$2n-1$但复杂度依然是线性的$O(n)$.
代码如下:(通过replace删除'#'变换为原串)
```python
class Solution:
	def longestPalindrome(self, s):
		_str = '$#' + '#'.join(s) + '#@'
		_id,mx = 0,0
		p = [0 for i in range(len(_str) + 1)]
		for i in range(1,len(_str) - 1):
			if i < mx:
				p[i] = min(mx - i,p[2 * _id - i])
			else:
				p[i] = 1
			while _str[i - p[i]] == _str[i + p[i]]:
				p[i] += 1
			if mx < i + p[i]:
				_id = i
				mx = i + p[i]
		_id,mx = 0,0
		for i in range(len(_str)):
			if mx < p[i]:
				_id = i
				mx = p[i]
		return _str[_id - p[_id] + 1:_id + p[_id]].replace('#','')
```
