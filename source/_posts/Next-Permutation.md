---
title: LeetCode 31. Next Permutation(思维)
copyright: true
mathjax: true
tags: 思维
categories: LeetCode
abbrlink: 4a6fe0a2
date: 2019-04-29 20:50:41
updated:
---
[传送门](https://leetcode.com/problems/next-permutation/)
## 题意
假设我们按照字典序来给一个升序排序的列表生成全排列，那么现在给出某一个排列，让你给出它的下一个排列(字典序顺序下的下一个，如果已经是最后一个，给出第一个)
要求空间复杂度$O(1)$
<!--more-->
## 思路
最最最简单的方法就是，生成所有的全排列然后找出对应的下一个。这样时间复杂度为$O(n!)$，空间复杂度为$O(n)$

其实我们很容易就能想到，给定一个当前状态的排列，比如"4321"，对于这个降序序列已经是最大字典序的排列了，那么他的下一个就只能是"1234"。
再给出另一个例子"3421"，"421"这个降序序列也是最后一个了，那么对于"3"我们应该找谁替换呢？应该找右边降序序列中大于它的最小的那个"4"，这样我们就得到了它的下一个是"4123"。
由此，我们可以得到我们的方法了。给定一个当前状态，我们应该从后往前找，找到那个降序序列被破坏的位置，也就是$nums[pos] > nums[pos - 1]$，然后将$pos-1$处的值和右边大于等于它的最小的那个交换，再将右边的升序排序即可。
**注意到，右边的已经是降序，所以找替换$pos-1$的值，我们可以使用二分**。另外，二者交换后，并不影响我们序列降序的性质，所以我们无需排序，只需直接reverse列表即可。
{% asset_img 1.gif%}
```python
class Solution:
    def nextPermutation(self, nums):
        pos = len(nums) - 1
        while pos > 0:
            if nums[pos] > nums[pos - 1]:
                break
            pos -= 1
        if pos == 0:
            nums.reverse()
        else:
            l,r,mid,ans = pos,len(nums) -1,0,0
            while l <= r:
                mid = (l + r) // 2
                if nums[mid] > nums[pos - 1]:
                    ans = mid
                    l += 1
                else:
                    r -= 1
            nums[pos - 1],nums[ans] = nums[ans],nums[pos - 1]
            nums[pos:] = list(reversed(nums[pos:]))
        return None
        
```
