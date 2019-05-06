---
title: LeetCode 32. Longest Valid Parentheses
copyright: true
mathjax: true
tags:
  - 思维
  - Dp
  - Stack
categories: LeetCode
abbrlink: 6f6d730d
date: 2019-05-05 19:24:30
updated:
---
[传送门](https://leetcode.com/problems/longest-valid-parentheses/)
# 题意:
给你一个只包含'('和')'的字符，让你找出最长的满足括号匹配条件的子串，返回其长度。
<!--more-->
# 思路:
## Brute Force
最好想的方法就是$O(n^2)$枚举出所有的子串，然后$O(n)$来判断是否满足条件。
 - 时间复杂度: $O(n^3)$
 - 空间复杂度: 取决于用什么方法来判断该字符串是否满足条件。若用stack则为$O(n)$，若使用两个变量线性判断,空间复杂度为$O(1)$
 
 代码略。
 
## Dp
$dp[i]$表示以$i$结尾的的满足匹配条件的最长子串的长度，我们可以发现满足条件的$i$，一定有$s[i] ==$'）'，否则$dp[i] = 0$。因此，我们可以推导出如下的转移
 - $s[i] ==$'）' 并且$s[i - 1] ==$'（'，那么有: $dp[i] = dp[i - 2] + 2$
 - $s[i] ==$'）'并且$s[i-1]==$'）' ，这时候说明有可能是'（（（）））'这种匹配的形式。所以我们还需要判断, 若$s[i - dp[i-1]-1] ==$‘（’，那么有$dp[i] = dp[i-1] + dp[i - dp[i-1] - 2] + 2$
 
```python
class Solution:
    def longestValidParentheses(self, s):
        if s == "": return 0
        dp = [0 for i in range(len(s))]
        for i in range(1,len(s)):
            if s[i] == ')':
                if s[i - 1] == '(':
                    dp[i] = (dp[i - 2] if i - 2 >= 0 else 0) + 2 
                if i - 1 - dp[i - 1] >= 0 and s[i - 1] == ')' and s[i - 1 - dp[i - 1]] == '(':
                    dp[i] = dp[i - 1] + (dp[i - 2 - dp[i - 1]] if i - 2 - dp[i - 1] >= 0 else 0) + 2
        return max(dp)
 ```
 
 >时间复杂度$O(n)$，空间复杂度$O(n)$

 ## Stack
 这里使用栈我们也有两种思想类似的方法。
 ### One
 如果是让你判断给定的字符串是否满足括号匹配，那么这就是一个经典的括号匹配问题。**可是对于寻找最长的子串，如果我们还是使用传统的方法来求解，那么主要难点就是在于如何判断或者标记是否连续的问题。**
 为了解决这个问题，我们可以使用一个数组，如果当前的栈顶字符满足匹配并出栈，我们可以将**当前的匹配长度记录在下一个栈顶上**。考虑到栈的“后进先出”的性质，每个栈顶便会记录其后面的与其相邻的括号匹配长度，当该栈顶出栈时，只需要拿其记录的长度+2，并累加到下一个即可。具体见代码，详细体会。
 ```python
 class Solution:
    def longestValidParentheses(self, s):
        stack = [0]
        longest = 0  
        for c in s:
            if c == "(":
                stack.append(0)
            else:
                if len(stack) > 1:
                    val = stack.pop()
                    stack[-1] += val + 2
                    longest = max(longest, stack[-1])
                else:
                    stack = [0]
        return longest
 ```
 >时间复杂度$O(n)$，空间复杂度$O(n)$
### Another
同样地，我们还是要解决上述的问题，即如果当前栈顶的括号和后面的来的括号可以**匹配**，那么我们该如何判断和我相邻的是否匹配了？匹配了多少呢？
初始时，stack中有一个$-1$（因为下标从$0$开始，其前一个就是$-1$）。
 - 如果我们遇到'（'则直接将其入栈
 - 如果是‘）’，说明可以匹配，我们直接弹出栈顶元素。并用下一个栈顶元素的index和当前的index计算已经匹配的序列长度。
 - 需要注意的是，如果当前是‘）’但是栈为空，也需要将其index入栈（本质上是起到了和$-1$一样的作用，因为我们知道如果‘（’比‘）’对栈没有影响，即每次都可以pop出值；但是如果'）'比‘（’多，那么栈有可能为空）
 ```python
class Solution:
    def longestValidParentheses(self, s):
        sta = [-1]
        _max = 0
        for i in range(len(s)):
            if s[i] == '(' or len(sta) == 0:
                sta.append(i)
            else:
                sta.pop()
                if not len(sta):
                    sta.append(i)
                else:
                    _max = max(_max,i - sta[-1])
        return _max
 ```
 >时间复杂地$O(n)$，空间复杂度$O(n)$
 
 ## Using two variables
 >这是一个时间复杂度$O(n)$，空间复杂度$O(1)$的方法

我们可以设两个变量$left$和$right$，当遇到一个'（’时，$left++$，否则$right++$。那么很显然，如果$left == right$则说明当前的长度可以匹配，并且长度为$2\times left$。如果$left < right$ 说明该段字符串无法匹配，将$left$ 和$right$置$0$。但是需要注意的是，如果遇到的字符串为'（）（（）（）'，由于第三个位置的‘（’的存在，使得后面的‘（）（）’无法匹配。所以这里我们需要**从左往右**按此方法计算一次最优解，再**从右往左**计算一次，两次取最优解
```python
class Solution:
    def longestValidParentheses(self, s):
        l,r = 0,0
        _max = 0
        for i in s:
            if i == '(':
                l += 1
            else:
                r += 1
            if l == r:
                _max = max(_max,2 * l)
            elif r > l:
                l,r = 0,0
        l,r = 0,0
        for i in s[::-1]:
            if i == '(':
                l += 1
            else:
                r += 1
            if l == r:
                _max = max(_max,2 * l)
            elif r < l:
                l,r = 0,0
        return _max
```

