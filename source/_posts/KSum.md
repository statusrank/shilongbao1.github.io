---
title: 总结一下LeetCode上的KSum问题(实时更新)
copyright: true
mathjax: true
tags:
  - KSum
  - TwoPointer
categories: LeetCode
abbrlink: 78db25ba
date: 2019-01-19 16:06:23
updated:
---
这里总结一下LeetCode上经典的几个问题,其实思路真的都很简单,所有的变体大部分都是依据2Sum和3Sum得来.
另外以前都是主用C++,最近开始用py,写这篇总结主要是为了记录下思路,然后练习下py.
<!--more-->
# Tow Sum
[传送门](https://leetcode.com/problems/two-sum/)
## 题意:
在nums中找到两个数使得nums1 + nums2 == target
## 思路:
最简单的方法就是暴力,$O(n^2)$,其次我们可以采用Hashtable(python里就是字典),然后遍历其中一个数num,得到另一个数target-num,在这个过程中我们可以来维护hash表,然后找到target-num的位置即可.时间复杂度,空间复杂度都是$O(n)$
```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res = {}
        for Index,Num in enumerate(nums):
        	if (target - Num) in res:
        		if res[target - Num] != Index:
        			return [Index,res[target - Num]]
        	res[Num] = Index
```
我们还可以采用双指针,但是需要排序(因为使用双指针需要使得我们的问题存在一个单调性),这里又需要我们返回下标,所以我们可以通过Hashtable来记录下原来的位置再排序即可.时间复杂度和空间复杂度也是$O(n)$
```python
class Solution:
    def twoSum(self, nums, target):
        if len(nums) < 2: return []
        _index = {}
        for i in range(len(nums)):
        	if nums[i] not in _index:
        		_index[nums[i]] = [i]
        	else:
        		_index[nums[i]].append(i)
        nums.sort()
        l,r = 0,len(nums) - 1
        while l < r:
        	s = nums[l] + nums[r]
        	if s < target:
        		l += 1
        	elif s > target:
        		r -= 1
        	else:
        		if nums[l] != nums[r]:
        			return [_index[nums[l]][0],_index[nums[r]][0]]
        		else:
        			return _index[nums[l]]
```

# 3Sum
[传送门](https://leetcode.com/problems/3sum/)
## 题意:
2Sum的变体,这里其实就是要你找出所有满足num1 + num2 + num3 = target的组合
## 思路:
依然非常简单,一开始我想的是$O(n^2)$处理任意两个数相加,然后$O(n)$去枚举第三个数,从而得到,另外两个数的和为target - num3.这样对原来的$O(n^2)$的数组排序,二分看是否有满足的.总体的时间复杂度就是$O(n^2) + O(nlogn^2) + O(n^2logn^2)$,因为排序占的复杂度最大的,所以还是一个$O(n^2logn^2)$的算法,不是很优啊.
[关于python内置排序list.sort](https://www.jiqizhixin.com/articles/2018-11-20-3)
PS: **更新**. 这里也可以将任意两个数的和用Hashtable存一下,这样$O(n)$遍历第三个数只需要$O(1)$在Hashtable中查询,但是最终的复杂度也还是$O(n^2)$的了.

后来想到了我们可以$O(n)$枚举一个数,然后得到另外两个数的和,余下的数列(当然需要提前排序)就成了2Sum的问题,再用双指针,那么也是$O(n^2)$,所以最后复杂度就是$O(n^2)$,而空间复杂度是$O(1)$.Good!
```python
class Solution:
    def threeSum(self, nums):
        ans = set()
        nums.sort()
        for i in range(len(nums) - 2):
        	l,r = i + 1,len(nums) - 1
        	while l < r:
        		_sum = nums[i] + nums[l] + nums[r]
        		if _sum == 0:
        			ans.add((nums[i],nums[l],nums[r]))
        			l += 1;r -= 1
        		elif _sum < 0:
        			l += 1
        		elif _sum > 0:
        			r -= 1
        return list(ans)
```

# 3Sum Closest
[传送门](https://leetcode.com/problems/3sum-closest/)
## 题意:
这是3Sum问题的一个变种,即我们不要求你找到三个数相加正好等于target,而是找到加和最接近target的三个数.
## 思路:
这里首先考虑**3Sum**是一个$O(n^2)$时间内我们可以直接解决的问题.那么我们也一定可以在$O(n^2)$时间内解决该问题,且一定是最优了.考虑在求解3Sum的过程中来维护最接近的即可,发现该问题的单调性和3Sum问题的单调性是一致的.
```python
class Solution:
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) < 3:
        	return []
        nums.sort()
        _closestsum,_closesdis = 0,0xFFFFFFFF
        for i in range(len(nums) - 2):
        	if i > 0 and nums[i] == nums[i - 1]:
        		continue
        	l,r = i + 1,len(nums) - 1
        	while l < r:
        		s = nums[i] + nums[l] + nums[r]
        		if s == target:
        			return target
        		elif s < target:
        			l += 1
        		else:
        			r -= 1
        		_currentdis = abs(s - target)
        		if _currentdis < _closesdis:
        			_closestsum = s
        			_closesdis = _currentdis
        return _closestsum
```

# 4Sum
[传送门](https://leetcode.com/problems/4sum/)
## 题意:
在nums中找到所有a + b + c + d = target
## 思路:
解决了2Sum,3Sum可以发现对于这个问题我们会有很多想法了.
首先最基本的一个想法就是将4Sum转化为3Sum,也就是我们先$O(n)$枚举一个,然后就转化为3Sum,而3Sum是$O(n^2)$,所以我们可以$O(n^3)$解决4Sum.
那么是否有更快的方法呢??
当然有.那就是还是Hashtable,利用空间来换时间.也就是我们可以先$O(n^2)$预处理两个数的和,然后在$O(n^2)$来枚举两个数的和,然后target - a - b,然后在hashtable中找是否存在这样的两个数的和即可.由于这里要保证4个数不相同,所以我们需要在Hashtable中记录好对应数的下标,我们就直接放在dictionary中处理就好.
时间复杂度空间复杂度都是$O(n^2)$
```python
class Solution:
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        _len,_dict,ans = len(nums),{},set()
        nums.sort()
        #this if sentence can reduce about 80 ms.
        if len(nums) < 4 or 4*nums[0] > target or 4*nums[_len-1] < target:
        	return []
        for i in range(_len):
        	for j in range(i + 1,_len):
        		_sum = nums[i] + nums[j]
        		if _sum not in _dict:
        			_dict[_sum] = [(i,j)]
        		else:
        			_dict[_sum].append((i,j))
        for i in range(_len):
        	for j in range(i + 1,_len):
        		_cha = target - (nums[i] + nums[j])
        		if _cha in _dict:
        			for k in _dict[_cha]:
        				if k[0] > j:
        					ans.add((nums[i],nums[j],nums[k[0]],nums[k[1]]))
        return list(ans)

```
