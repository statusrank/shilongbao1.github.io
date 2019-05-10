---
title: LeetCode 42. Trapping Rain Water
copyright: true
mathjax: true
tags:
  - Dp
  - Stack
  - 思维，TwoPointer
categories: LeetCode
abbrlink: 3c5ba713
date: 2019-05-10 17:58:29
updated:
---
[传送门](https://leetcode.com/problems/trapping-rain-water/)
# 题意
给你$n$个非负整数，表示围栏的高度（宽度都为1），问你如果下雨一共能储多少水？如图：
{% asset_img 1.png %}
> **Input:** [0,1,0,2,1,0,1,3,2,1,2,1]
**Output:** 6
<!--more-->
# 思路
## Dp
可以发现对于每一个$i$来说，它能储水的高度显然取决于它左面和右面最高的围栏高度之间的最小值。所以我们可以简单的想到brute force，对每个$i$,分别找它左面和右面的最大值，然后答案就是$min(max_l[i],max_r[i]) - height[i]$。显然这种方法的时间复杂度为$O(n^2)$,所以我们可以通过$Dp$的思想将时间复杂度优化到$O(n)$级别。
{% asset_img 2.png %}
```python
class Solution:
    def trap(self, height):
        if not height:
            return 0
        _len = len(height)
        max_l = [height[i] for i in range(_len)]
        max_r = [height[i] for i in range(_len)]
        for i in range(1,_len):
            max_l[i] = max(max_l[i - 1],height[i])
        ans = 0
        for i in range(_len - 2,-1,-1):
            max_r[i] = max(max_r[i + 1],height[i])
        for i in range(_len):
            ans += min(max_l[i],max_r[i]) - height[i]
        return ans 

```
## Two Pointer
上面的Dp的方法虽然是$O(n)$,但是仍然遍历了很多次，我们可以使用Two pointer来进一步优化。
我们可以发现对于每个$i$来说，如果有$maxleft[i] < maxright[i]$,那么$i$处的储水量和$maxright[i]$就没什么关系了。因此，我们可以得出:

 - 如果有$maxleft[i] < maxright[i]$,我们在这个过程中维护$maxleft$即可，如果$height[i] > maxleft$ 则更新，答案为$ans += maxleft - height[i]$,$l += 1$ 继续从左往右
 - 如果有$maxleft[i] < maxright[i]$，那么应该从右往左了，并维护$maxright$,答案为$ans += maxright-height[i],r -= 1$
```python
class Solution:
    def trap(self, height):
        if not height:
            return 0
        _len = len(height)
        max_l,max_r = 0,0
        l,r = 0,_len - 1
        ans = 0
        while l <= r:
            if max_l < max_r:
                max_l = max(max_l,height[l])
                ans += max_l - height[l]
                l += 1
            else:
                max_r = max(max_r,height[r])
                ans += max_r - height[r]
                r -= 1
        return ans
```
## Stack
以上两个方法都是对每个$i$找到左右最深能储多少水，这里其实我们还可以换个思想。对于每一个$i$,只要左右有比他高的，那么他肯定能储一部分水，如果有再高的又能多储一部分——所以这里我们的思想是可以一层层的解决储水的问题。
那么这里我们就需要维护一个**递减栈**，栈中保存下标。
如果有$height[i] > height[stack[top]]$,那么说明stack[top]处可以储水，离它左右最近的一层就是当前的$i$和栈中的下一个(栈顶)。那么有:
 - 这一层的长度为$distance = i - stack[top]-1$
 - 最大高度为$h = min(height[i],height[stack[-1]]$
 - 则储水为$ans += h \times distance$
 ```python
 class Solution:
    def trap(self, height):
        if not height:
            return 0
        ans = 0
        _len = len(height)
        stack = []
        for i in range(_len):
            while len(stack) and height[i] > height[stack[-1]]:
                top = stack.pop()
                if not len(stack):
                    break
                dis = i - stack[-1] - 1
                ans += dis * (min(height[i],height[stack[-1]]) - height[top])
            stack.append(i)
        return ans 
 ```

