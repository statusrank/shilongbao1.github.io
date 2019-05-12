---
title: LeetCode 48. Rotate Image
copyright: true
mathjax: true
tags: 思维
categories: LeetCode
abbrlink: ed5ef735
date: 2019-05-12 11:50:42
updated:
---
[传送门](https://leetcode.com/problems/rotate-image/)
# 题意：
给你一个 $n \times n$的二维矩阵，让你在使用$O(1)$的空间复杂度的情况下，将该矩阵进行顺时针旋转(clockwise)。
<!--more-->
# 思路：
## clockwise
一个简单找规律。先将矩阵倒置，在按对角线对称即可。
```python
"""
 * clockwise rotate
 * first reverse up to down, then swap the symmetry 
 * 1 2 3     7 8 9     7 4 1
 * 4 5 6  => 4 5 6  => 8 5 2
 * 7 8 9     1 2 3     9 6 3
"""
class Solution:
    def rotate(self, matrix):
        if not matrix:
            return 
        matrix.reverse()
        for i in range(len(matrix)):
            for j in range(i + 1):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
```

## anticlockwise
```python
"""
 * anticlockwise rotate
 * first reverse left to right, then swap the symmetry
 * 1 2 3     3 2 1     3 6 9
 * 4 5 6  => 6 5 4  => 2 5 8
 * 7 8 9     9 8 7     1 4 7
"""
```
## 一行代码搞定矩阵旋转
但是好像这个空间复杂度不是$O(1)$
```python
        def rotate(self, matrix):
            matrix[:] = map(list,zip(*matrix[::-1]))
```
[Zip解析](https://www.runoob.com/python/python-func-zip.html)
[Map解析](https://my.oschina.net/zyzzy/blog/115096)
