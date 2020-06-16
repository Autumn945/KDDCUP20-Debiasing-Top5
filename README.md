## 赛题链接
https://tianchi.aliyun.com/competition/entrance/231785/introduction      

# 运行环境
- python 3.6
- tensorflow 1.12

# 运行入口
- ./run.sh

# 数据文件格式
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
