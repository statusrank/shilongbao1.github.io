---
title: 小米oj 米兔的胡萝卜(ST表水题)
copyright: true
mathjax: true
tags:
  - ST表
  - 区间最大值
  - ACM
categories: ACM
abbrlink: b58b8411
date: 2018-09-10 20:56:03
updated:
---

[传送门](https://code.mi.com/problem/list/view?id=111)

##题意:

N 个米兔排成一排，每个米兔拥有的胡萝卜数可能为 0 ~ 9 中的任一整数，拥有胡萝卜最多的米兔可以获得由米司令奖励的品罗小怪兽料理机。M个询问,查询区间最大值

<!--more-->

##思路:

区间最大值也可以线段树的,但是这题数据量有点大,而且不带修改的,所以可以直接上ST表。

###关于ST表

ST表就是一个来解决区间最值(Rmq)问题的算法,但是他不支持在线修改的.针对ST表,预处理是$O(nlogn)$但是查询$O(1)$的复杂度。这就是他的优良特性。

###如何预处理ST表

ST表就是根据DP思想来求的,我们设$dp[i][j]$表示$[i,i+2^j-1]$的区间最大值.那么可知$ dp[i][0]=a[i]$.状态转移:将区间分为两段,一段为 $[i,i + 2^{j-1}-1]$,一段为

$[i + 2^{j-1},i + 2^{j-1} + 2^{j-1} - 1] $

即转移方程为 $dp[i][j] = max(dp[i][j-1],dp[i+(1<<(j-1))][j-1])$

###如何查询

需要查询的区间为[i,j],则需要找到两个覆盖这个区间的最小幂区间。这两个区间可以重复,因为区间重复对最值是没有影响的

因为区间的长度为j-i+1，所以可以取k=log2(j-i+1)。

则$RMQ(A,i,j)=max(dp[i][k],dp[j-2^k+1][k])$。{% asset_img 1.png %}

###本题代码

PS:这个垃圾oj的读入对C++很不友好,所以这里采用stringstream来进行转化。另外cin.geiline注意s的长度,wa了一晚上因为这个东西。。

```C++

#include<bits/stdc++.h>

using namespace std;

const int maxn = 1e5 + 10;

char s[10*maxn];

int dp[maxn][30];

int lg[maxn];

int N,M,R;

void Replace()

{

    int len = strlen(s);

    for(int i = 0;i < len;++i)

    {

        if(s[i] == ';') s[i] = ' ';

    }

}

void ST()

{

    for(int j = 1;(1 << j)<= N;++j)

        for(int i = 1;i + (1 << j) - 1 <= N;++i)

            dp[i][j] = max(dp[i][j - 1],dp[i + (1 << (j-1)) ][j - 1]);

    return ;

}

int query(int l,int r)

{

    l = max(1,l),r = min(r,N);

    int k = lg[r-l+1];

    return max(dp[l][k],dp[r-(1<<k) + 1][k]);

}

int main()

{

    lg[1] = 0;

    for(int i = 2;i < maxn;++i)

        lg[i] = lg[i / 2] + 1;

    while (cin.getline(s, 9*maxn)) {

        Replace();

        stringstream ss(s);

        memset(dp,-1,sizeof dp);

        ss >> N >> M >> R;

        //cout << N << ' ' << M << ' ' << R << endl;

        for(int i = 1;i <= N;++i)

            ss >> dp[i][0];

        ST();

        for(int i = 0;i < M;++i)

        {

            int q;

            ss >> q;

            printf("%d%c",query(q-R,q+R),i == M - 1?'\n':' ');

        }

    }

    return 0;

}



```