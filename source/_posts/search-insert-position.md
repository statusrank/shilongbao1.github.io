---
title: LeetCode 33. Search in Rotated Sorted Array
copyright: true
mathjax: true
tags: 二分
categories: LeetCode
abbrlink: 7c97f959
date: 2019-05-06 13:09:51
updated:
---
[传送门](https://leetcode.com/problems/search-in-rotated-sorted-array/)
# 题意
假设一个按从小到大排序的数组按照某个位置的中心点进行了循环(比如，$[1,2,3,4]$变为$[3,4,1,2]$)。给你一个target，你需要它在这个数组中的index，若无则返回$-1$。**要求时间复杂度为$O(\log n)$级别，数组中元素无重复**
<!--more-->
# 思路
看到时间复杂度要求为$O(\log n)$，而且数组还是有序的，那么我们首先应该想到的就是二分查找了。但是这个数组又发生了循环，使得新的数组并不是一个完全有序的。所以这里我们难点就是如何在$O(\log n)$时间内找到发生循环的$pivot$,显然这个部分也可以通过**binary search**来解决。
$pivot$处一定是有$nums[pivot] < nums[pivot-1]$的，所以我们可以通过二分找到pivot的位置（具体见代码）。
```python
"""
class Solution:
    def search(self, nums, target):
        return nums.index(target) if target in nums else -1
"""
class Solution:
    def search(self, nums, target):
        if not nums:
            return -1
        _len = len(nums)
        l,r = 0,_len - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        pivot, l, r, ans = l, 0, _len - 1, -1
        while l <= r:
            mid = (l + r) // 2
            now = nums[(mid + pivot) % _len]
            if target == now:
                ans = mid
                break
            elif target > now:
                l = mid + 1
            else:
                r = mid - 1
        return (ans + pivot) % _len if ans != -1 else ans

```
