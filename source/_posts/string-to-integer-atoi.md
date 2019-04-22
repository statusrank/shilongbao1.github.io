---
title: LeetCode String to Integer (atoi) (正则表达式)
copyright: true
mathjax: true
tags:
  - 正则表达式
  - python
categories: LeetCode
abbrlink: 194c9918
date: 2019-01-09 19:45:39
updated:
---
[传送门](https://leetcode.com/problems/string-to-integer-atoi/)
##题意:
很easy的一道题啊,就是实现一些c语言atoi()将字符转化为integer的函数.
<!--more-->
##思路:
按照题目要求,然后注意下细节其实就可以了.
```python
class Solution:
    def myAtoi(self, _str):
        _str = _str.strip()
        if _str == "":
        	return 0
        _len,flag = 0,0
        if _str[0] == '-' or _str[0] == '+':
        	flag = 1
        for i in _str[flag:]:
        	if i >= '0' and i <= '9':
        		_len += 1
        	else:
        		break
        if _len == 0:
        	return 0
        ans = int(_str[:_len + flag])
        if ans < -2**31:
        	return -2**31
        elif ans > 2**31 - 1:
        	return 2**31 - 1
        else:
        	return ans
```
但是记录下来 主要是该题可以使用python中的正则表达式(regular expression)来简单的完成.
首先我们先总结下py中的正则表达式.
##正则表达式
[部分转载来源](https://www.cnblogs.com/huxi/archive/2010/07/04/1771073.html)
正则表达式并不是python本身的一部分,它是一个极其强大的处理字符串的工具,被集成在re模块中.
下图列出了python支持的正则表达式元字符和语法:
{% asset_img 1.png %}
###贪婪模式与非贪婪模式
正则表达式通常用于在文本中查找匹配的字符串.python里数量词默认是贪婪的,总是尝试尽可能匹配更多的字符,非贪婪的则相反,它总是尝试匹配尽可能少的字符.例如：正则表达式"ab*"如果用于查找"abbbc"，将找到"abbb"。而如果使用非贪婪的数量词"ab*?"，将找到"a"。
###re模块
[见](https://www.cnblogs.com/tina-python/p/5508402.html)
本题中我们需要使用re中的search和match object的group即可.
正则表达式为:r"^ *[-+]?[0-9]+"
```python
class Solution:
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        match = re.search(r"^ *[-+]?[0-9]+", str)
        if match:
            match = int(match.group())
            if match < -2**31:
                return -2**31
            elif match > 2**31-1:
                return 2**31-1
            return match
        
        return 0
```

