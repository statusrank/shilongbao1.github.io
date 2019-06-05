---
title: 分解机(Factorization Machine，FM)算法总结
copyright: true
mathjax: true
tags: FM
categories: 推荐
abbrlink: 1ed9075e
date: 2019-06-05 21:06:11
updated:
---
[转载来源](https://www.cnblogs.com/pinard/p/6370127.html)，原文链接失效。
FM(Factorization Machine)主要是为了解决数据稀疏的情况下，特征怎样组合的问题。此算法的主要作用是可以把所有特征进行高阶组合，减少人工参与特征组合的工作。**FM只需要线性时间复杂度，可以应用于大规模机器学习**。
<!--more-->
## 预测任务
{% asset_img 1.png %}
{% asset_img 2.png %}
{% asset_img 3.png %}
## 模型方程
{% asset_img 4.png %}
{% asset_img 5.png %}
{% asset_img 6.png %}
## 回归和分类
{% asset_img 7.png %}
## 学习算法
{% asset_img 8.png %}
<font color = "red">注：上面最后一句话应该是"而$g_{\theta}(x)$则利用$\widehat{y}(x) - \theta h_{\theta}(x)$来计算"</font>
{% asset_img 9.png %}
{% asset_img 10.png %}
{% asset_img 11.png %}
{% asset_img 12.png %}
{% asset_img 13.png %}
{% asset_img 14.png %}
{% asset_img 15.png %}
## 参考文献
{% asset_img 16.png %}
