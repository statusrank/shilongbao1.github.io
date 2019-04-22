---
title: Median of Two Sorted Arrays(二分)
copyright: true
mathjax: true
abbrlink: 7db58ea8
date: 2019-01-03 15:15:30
tags: [思维,二分]
categories: [LeetCode]
updated:
---
[传送门](https://leetcode.com/problems/median-of-two-sorted-arrays/)
##题意:
给你两个已经有序的序列,找到将这两个序列合并后的中位数.
**要求: 复杂度应为O(log min(m,n))**
<!--more-->
##思路:
一道非常好的题目.
找中位数我们最最普遍的做法就是将这两个有序序列合并,然后直接找中间那个数即可.**复杂度是O(n + m).**
但是题目中要求的复杂度是log级别的,这里我们首先想到的就是二分了.难点在于如何同时在两个序列之中进行二分?
首先我们需要中位数的具体意义,一个序列的中位数将整个序列两部分设为left和right,它满足$num[left] <= num[right]$ 且 $len(left) == len(right)$ 或者 $len(left) - 1 = len(right)$.
那么对于序列A 长度为m,我们可能的划分点有$m + 1$个,i的范围$[0,m]$
如图:{% asset_img 1.png %}
同理对于序列B长度为n,我们也有如下:
{% asset_img 2.png %}
如果我们将A和B的left看成一部分,right看成一部分,即可得到下图:
{% asset_img 3.png %}
此时我们根据两部分的长度可以得到:
$i + j = m - i + n - j(or \ \ i + j = m - i + n - j + 1)$
因此$j = \frac{m + n + 1}{2} - i$
同时我们有$max(left) \leq min(right)$即:对于两个分割点i,j我们有:
$nums1[i -1] \leq nums2[j]$和$nums2[j - 1] \leq nums1[i]$
PS:当然我们需要特判i,j = 0和=m的情况.
由此我们得到了二分应该满足的条件,我们只需要在长度最小的那个序列中(设为nums1)对i进行二分,范围$[0,m]$然后根据公式得到$j$,再判断是否满足$nums1[i -1] \leq nums2[j]$和$nums2[j - 1] \leq nums1[i]$这两个条件.
如果$nums1[i -1] > nums2[j]$说明i太大应减小.
如果$nums2[j - 1] > nums1[i]$说明i太小应增大.
最后根据$m+n$的奇偶判断中位数的值即可.
```python
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        m,n = len(nums1),len(nums2)
        if m > n:
        	m,n,nums1,nums2 = n,m,nums2,nums1
        l,r,lmax,rmin  = 0,m,0,0
        while l <= r:
        		i = (l + r) // 2
        		j = (m + n + 1) // 2 - i
        		if i < m and nums2[j - 1] > nums1[i]:
        			l = i + 1
        		elif i > 0 and nums1[i - 1] > nums2[j]:
        			r = i - 1
        		else:
        			if i == 0:
        				lmax = nums2[j - 1]
        			elif j == 0:
        				lmax = nums1[i - 1]
        			else:
        				lmax = max(nums1[i - 1],nums2[j - 1])
        			if (m + n) % 2 == 1:
        				return lmax
        			if i == m:
        				rmin = nums2[j]
        			elif j == n:
        				rmin = nums1[i]
        			else:
        				rmin = min(nums1[i],nums2[j])
        			return (lmax + rmin) / 2
```

