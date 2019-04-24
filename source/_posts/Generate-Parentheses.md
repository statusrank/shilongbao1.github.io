---
title: 'Generate-Parentheses(Dfs,Dp)'
copyright: true
mathjax: true
tags:
  - Dp
  - dfs
  - 卡特兰数
categories: LeetCode
abbrlink: 1574a532
date: 2019-04-22 15:13:15
updated:
---
[传送门](https://leetcode.com/problems/generate-parentheses/)
## 题意
给你n对括号，让你生成所有满足条件的括号。
<!--more-->
## 思路
### 方法一：暴力
每一个满足条件的括号序列长度为$2n$,每个位置有两种可能得情况,要么为'(',要么为')'。很明显我们可以直接暴力产生所有的括号序列,然后$O(n)$判断是否是合理的序列即可。
```python
class Solution:
    def generateParenthesis(self, n):
        def generateAll(_str = []):
            if len(_str) == 2 * n:
                if check(_str):
                    ans.append("".join(_str))
                    return
            else:
                _str.append('(')
                generateAll(_str)
                _str.pop()
                _str.append(')')
                generateAll(_str)
                _str.pop()
        def check(_str = ""):
            flag = 0
            for s in _str:
                if s == '(': flag += 1
                else: flag -= 1
                if flag < 0: return False
            return flag == 0
        ans = []
        generateAll()
        return ans
```
### 方法二: dfs
给定n个左括号和n个右括号,很明显要想保证最后排列出的括号是匹配的,那么任意时刻我们都要保证已经放的左括号的个数要大于等于右括号。(本质上这是一个卡特兰数的问题)所以我们可以通过这个条件进行dfs并剪枝。
```python
class Solution:
    def generateParenthesis(self, n):
        ans = []
        def generateSolution(left = 0,right = 0,_str = ""):
            if left == n and right == n:
                ans.append(_str)
                return 
            if left < n:
                generateSolution(left + 1,right, _str + '(')
            if right < n and right < left:
                generateSolution(left,right + 1, _str + ')')

        generateSolution()
        return ans
```

### 方法三: 动态规划
我们要生成满足条件的n对括号排列,我们可以发现其可以由所有满足条件的$1,2,\dots,n-1$对的组合而来:对于新添加的一对 "( )",可以有$x$对满足条件的括号在新添加的里面,$y$对括号在新添加的后面。这样既可产生全部n对满足条件的排列。
```python
class Solution:
    def generateParenthesis(self, n):
        dp = [[] for i in range(n + 1)]
        dp[0].append('')
        for i in range(n + 1):
            for j in range(i):
                dp[i] += ['(' + x + ')' + y for x in dp[j] for y in dp[i - 1 - j]]
        return dp[n]
```

## 扩展
在上面第二种方法的时候我们提到过,满足条件的括号序列的个数本质上是**卡特兰数**。下面简单复习下卡特兰数: [详细介绍](https://blog.csdn.net/HowardEmily/article/details/60880084)
卡特兰数的前几项为:$1,1,2,5,14,42,132,429,\dots$
令$h(0) = h(1) = 1,$catalan数满足递推式:

 -  $h(n) = h(0)*h(n-1) + h(1)*h(n-2) + \dots + h(n - 1)*h(0) (n >= 2)$
 - $h(n) =  \frac{h(n-1) * (4n-2)}{n + 1}$
 - $h(n) = \frac{C(2n,n)}{n + 1},n = 0,1,2,..\dots$
 - $h (n) = C(2n,n) - C(2n,n - 1),n = 0,1,2,..\dots$
常见的catalan数问题:
 - 括号化问题
 - 出栈次序问题
 - 将多边行划分为三角形问题。将一个凸多边形区域分成三角形区域(划分线不交叉)的方法数
 - 在圆上选择2n个点,将这些点成对连接起来使得所得到的n条线段不相交的方法数?
 - 给定N个节点，能构成多少种形状不同的二叉树？

 

