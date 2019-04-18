---
title: 热门文章
comments: false
keywords: top,文章阅读量排行榜
description: 博客文章阅读量排行榜
---
<div id="top"></div>  
<script src="//cdn1.lncld.net/static/js/3.0.4/av-min.js"></script>  
<script>AV.initialize("QCHnVkYIrh5qoSOSSISETUBC-gzGzoHsz", "nPH2aXy1ll77oLHYcbJr1xpX");</script>  
<script type="text/javascript">  
 var time=0  
 var title=""  
 var url=""  
 var query = new AV.Query('Counter');  
 query.notEqualTo('id',0);  
 query.descending('time');  
 query.limit(100);  
 query.find().then(function (todo) {  
 for (var i=0;i<1000;i++){  
 var result=todo[i].attributes;  
 time=result.time;  
 title=result.title;  
 url=result.url;
var content="<a href='"+"https://statusrank.xyz"+url+"'>"+title+"</a>"+"<br />"+"<font color='#555'>"+"阅读次数："+time+"</font>"+"<br /><br />"; 
 document.getElementById("top").innerHTML+=content  
 }  
 }, function (error) {  
 console.log("error");  
 });  
</script>  
  

