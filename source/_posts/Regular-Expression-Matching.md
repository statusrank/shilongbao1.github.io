---
title: LeetCode Regular Expression Matching(动态规划)
copyright: true
mathjax: true
tags: Dp
categories: LeetCode
abbrlink: 9ba9f2d4
date: 2019-01-13 13:04:38
updated:
---
[传送门](https://leetcode.com/problems/regular-expression-matching/)
##题意:
给你两个字符串s和p,让你判断两个字符串是否可以完全匹配.
匹配采用正则化匹配的方式,'.'可以匹配任意字符,'*'表示前面的一个字符匹配0次或多次.
<!--more-->
##思路:
比较好想的一种方法就是递归.
首先p中若没有'.'和'*',那么只需要看s和p是否完全一样即可.
其次若p中有'.'那么只需要跳过s中和p的'.'对应的字符,往后继续判断即可
最后若p中有'*'那么它可以使它前面的字符匹配0次或者多次,这种情况我们比较好用递归来解决,也就是让其匹配0次和1次进行递归,就会出现匹配多次的情况.
```python
class Solution:
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not p:
        	return not s
        first_match = bool(s) and p[0] in {s[0],'.'}
        if len(p) >= 2 and p[1] == '*':
        	return (self.isMatch(s,p[2:])) or first_match and self.isMatch(s[1:],p)
        else:
        	return first_match and self.isMatch(s[1:],p[1:])

```


另一种方法就是动态规划(dp),我们发现按照上面我们递归的方法,我们的当前状态都是和后一步的是否匹配状态有关的.所以我们可以设dp(i,j)表示s[i:]和p[j:]是否匹配,这样就会减少很多不必要的递归.
```python
class Solution:
    def isMatch(self, s, p):
    	mem = {}
    	def dp(i,j):
    		if (i,j) not in mem:
	    		if j == len(p):
	    			ans = i == len(s)
	    		else:
	    			match = i < len(s) and p[j] in {s[i],'.'}
		    		if j + 1 < len(p) and p[j + 1] == "*":
		    			ans = dp(i,j + 2) or (match and dp(i + 1,j))
		    		else:
		    			ans =  match and dp(i + 1,j + 1)
    			mem[i,j] = ans
    		return mem[i,j]
    	return dp(0,0)
```
动态规划的另一种写法.这个方法还是按照动态规划的思路,这里dp(i,j)我们表示s[:i]和p[:j]是否匹配,则:
if p[j-1]为'.'或a-z:
$dp[i][j] = dp[i-1]][j-1]$如果$s[i - 1] == p[j - 1]$
else:
if $dp[i][j] = dp[i ][j - 2]$ 如果 $p[j - 2] != s[i - 1]$
else
$dp[i][j] == dp[i][j-2]$ 这个代表匹配0次
$dp[i][j] == dp[i][j-1]$ 这个代表匹配1次
$dp[i][j] = dp[i-1][j]$ 这个代表匹配多次

```python
class Solution:
    def isMatch(self, s, p):
    	dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    	dp[0][0] = True
    	for i in range(1,len(p) + 1):
    		if p[i - 1] == "*":
    			dp[0][i] = dp[0][i - 2]
    	for i in range(1,len(s) + 1):
    		for j in range(1,len(p) + 1):
    			match = p[j - 1] in {s[i - 1],'.'}
    			if match:
    				dp[i][j] = dp[i- 1][j - 1]
    			elif j != 1 and p[j - 1] == '*':
    				if p[j - 2] not in {s[i - 1],'.'}:
    					dp[i][j] = dp[i][j] or dp[i][j - 2]
    				else:
    					dp[i][j] = dp[i][j] or dp[i - 1][j] or dp[i][j - 2] or dp[i][j - 1]
    	return dp[len(s)][len(p)]
```
