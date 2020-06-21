## 赛题链接
https://tianchi.aliyun.com/competition/entrance/231785/introduction      

## 运行环境
- python 3.6
- tensorflow 1.12

## 运行入口
- ./run.sh


## 数据集
https://pan.baidu.com/s/1bf5lpP0yqebXF4yHhHv0Lw 提取码：zf58 

**解压密码**
```  
7c2d2b8a636cbd790ff12a007907b2ba underexpose_train_click-1  
ea0ec486b76ae41ed836a8059726aa85 underexpose_train_click-2  
65255c3677a40bf4d341b0c739ad6dff underexpose_train_click-3  
c8376f1c4ed07b901f7fe5c60362ad7b underexpose_train_click-4  
63b326dc07d39c9afc65ed81002ff2ab underexpose_train_click-5  
f611f3e477b458b718223248fd0d1b55 underexpose_train_click-6  
ec191ea68e0acc367da067133869dd60 underexpose_train_click-7  
90129a980cb0a4ba3879fb9a4b177cd2 underexpose_train_click-8  
f4ff091ab62d849ba1e6ea6f7c4fb717 underexpose_train_click-9  

96d071a532e801423be614e9e8414992 underexpose_test_click-1  
503bf7a5882d3fac5ca9884d9010078c underexpose_test_click-2  
dd3de82d0b3a7fe9c55e0b260027f50f underexpose_test_click-3  
04e966e4f6c7b48f1272a53d8f9ade5d underexpose_test_click-4  
13a14563bf5528121b8aaccfa7a0dd73 underexpose_test_click-5  
dee22d5e4a7b1e3c409ea0719aa0a715 underexpose_test_click-6  
69416eedf810b56f8a01439e2061e26d underexpose_test_click-7  
55588c1cddab2fa5c63abe5c4bf020e5 underexpose_test_click-8  
caacb2c58d01757f018d6b9fee0c8095 underexpose_test_click-9 
```

**数据目录**
```
|-- data
  |-- underexpose_train
    |-- underexpose_user_feat.csv
    |-- underexpose_item_feat.csv
    |-- underexpose_train_click-0.csv
    |-- underexpose_train_click-1.csv
    |-- ...
    |-- underexpose_train_click-9.csv
  |-- underexpose_test
    |-- underexpose_test_click-0
      |-- underexpose_test_qtime-0.csv
      |-- underexpose_test_click-0.csv
    |-- underexpose_test_click-1
      |-- underexpose_test_qtime-1.csv
      |-- underexpose_test_click-1.csv
    |-- ...
    |-- underexpose_test_click-9
      |-- underexpose_test_qtime-9.csv
      |-- underexpose_test_click-9.csv
```

## 模型
- 主要模型框架：
![image](https://github.com/Autumn945/KDDCUP20-Debiasing-Top5/blob/master/image.png)
