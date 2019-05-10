---
title: 总结下LeetCode上 combination sum problem(未完结)
copyright: true
mathjax: true
tags:
  - Dp
  - Backtrack
categories: LeetCode
top: 100
abbrlink: 49bf382f
date: 2019-05-08 11:26:23
updated:
---
# LeetCode 39. Combination Sum
[传送门](https://leetcode.com/problems/combination-sum/)
## 题意
给你一个候选集合(全都是正数)和target number，请你从候选集合中找出所有的加和等于target的组合。
<!--more-->
## 思路
### One 
这种问题最简单的就是backtracking方法，外加一点小小的剪枝
```python
class Solution:
    def combinationSum(self, candidates, target):
        ans = []
        candidates.sort(reverse = True)
        def dfs(target,cur,path):
            if target == 0:
                ans.append(path)
                return 
            for i in range(cur,len(candidates)):
                if target - candidates[i] >= 0:
                    dfs(target - candidates[i],i,path + [candidates[i]])
        dfs(target,0,[])
        return ans
```
### Another
可以想一下：每个商品可以无限用，同时又要凑出target，那么显然我们可以将该问题想成是一个完全背包的满包问题+记录路径的问题。[01背包+记录路径问题](https://blog.csdn.net/HowardEmily/article/details/77175117)
这个和上面的问题稍有偏差。在这里我们不存在最优解，而是有满足条件的所有的解，所以我们需要每次对子结构中满足条件的解进行组合。（由于这里我们对每件商品一次性放完，所以不会存在重复解的情况）
```python
class Solution:
    def combinationSum(self, candidates, target):
        dp = collections.defaultdict(list)
        for val in candidates:
            for j in range(val,target + 1):
                if j - val == 0:
                    dp[j].append([val])
                else:
                    tmp = dp[j - val]
                    for _list in tmp:
                        l = _list + [val]
                        dp[j].append(l)
        return dp[target]
```


