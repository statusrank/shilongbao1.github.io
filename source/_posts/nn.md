---
title: c语言中移位运算符
tags: 各种基础知识
abbrlink: 9943af19
date: 2018-06-14 11:38:39
---

###本文说一下c语言中的移位运算符###
有如下代码:  

```
#include<bits/stdc++.h>

using namespace std;

int main()
{
   int a = 1,b = 32;
   printf("%d %d\n",a << b,1 << 32);
}
```
  
<!-- more -->  

输出结果:
	1 0

我们可以发现变量的移位运算和常量的移位运算是不一样的,常量的左移的运算后面补0,所以超过31为整个32位全是0,最后结果也是0.而<font color = "red">对于变量的移位如果b超过了其size(int 这里32位),就先会用b = b % size ，然后在进行移位,所以是1</font>.
