# 知识分析与处理-Kmeans
## 实验数据
* 本次使用document.txt文件中实体间关系类型为/location/location/contains、/people/person/nationality、/people/person/place_lived、/business/person/company（共四个类型）的数据
* 使用stoprwords.txt作为文本清洗工具
* 输出结果放在target.csv中
* 验证数据为人工识别的约1700条数据
## 实现功能
* 数据读取与清洗
* 词向量生成
* 计算余弦相似度与分簇
* 更新质心
* 数据分析与检验
## 算法分析
```python
初始化4个随机质心
对于所有样本
    计算每个样本余弦相似度
    分簇
    更新质心
```
## 实验结果评测
* 与已标注数据集结果进行对比：准确率 = 准确数目/ 1700 =
   取5次试验比较结果：
   0.64937 0.58497 0.6145 0.7049

* 四个聚类中心相互的余弦相似度
  |  | 1 |2|3|4 |
  | ------ | ------ | ------ |------ |------ |
  | 1 |1 | 0.0123 | 0.0928 | 0.00127 |
  | 2 | 0.0123 | 1 |0.0306 |0.006 |
  | 3 | 0.0928 | 0.0306 |1 |0.004|
  | 4 | 0.0127 | 0.006 |0.004 |1|

* 取三千条为例观察收敛特性
  ![收敛趋势](1.jpg)
## 总结
* 由于测试数据集中1类占了绝大多数，2,3,4数目太少，导致数据结果并不能全面反映算法质量。
* 虽然已知K = 4，但Kmeans对噪音和异常点非常敏感，为了减少离群点和孤立点对聚类的效果。可以采用求点的中位数这种方式来改进效果
