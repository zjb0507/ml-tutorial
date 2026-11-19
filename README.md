# 机器学习入门练习

基于 **scikit-learn** 的机器学习入门代码合集，覆盖 KNN、线性回归、逻辑回归、感知机、决策树等经典算法，包含课堂练习与小型实战案例（心脏病预测、广告销量分析、手写数字识别）。

## 目录结构

```
ml_tutorial/
├── ch01_02_机器学习基础/     # Matplotlib 绘图、数据集生成、分类评估报告
├── ch03_knn/                 # KNN 分类与回归、归一化/标准化、心脏病预测案例
├── ch04_线性回归/            # 线性回归、梯度下降、SGD、广告销量案例
├── ch05_逻辑回归/            # 逻辑回归 API、心脏病检测、手写数字识别
├── ch06_感知机/              # 感知机实现逻辑门电路（与门/或门/与非门）
├── ch07_决策树/              # 信息熵计算
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+
- 依赖见 `requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

脚本通过**相对路径**读取同目录下的 CSV 数据，因此请先进入对应章节目录再运行，例如：

```bash
cd ch03_knn
python 4_案例测试.py
```

Notebook 文件（`.ipynb`）可直接用 Jupyter Notebook 或 PyCharm 打开运行。

## 章节内容

### ch01_02_机器学习基础
- `机器学习练习.py`：NumPy 与 Matplotlib 基础绘图（sin/cos 曲线）
- `classification_report.py`：使用 `make_classification` 生成数据，训练逻辑回归并输出分类报告

### ch03_knn
- `1_knn分类测试.py`：KNN 分类器基本用法
- `2_knn回归测试.py`：KNN 回归器基本用法
- `3_归一化.py`：MinMaxScaler 归一化与 StandardScaler 标准化
- `4_案例测试.py`：心脏病数据集预处理、ColumnTransformer、GridSearchCV 调参
- `test.py`：心脏病预测流程复现
- 数据：`heart_disease.csv`

### ch04_线性回归
- `1_lr_test.ipynb`：线性回归原理与实验
- `2_梯度下降.py`：手写损失函数与梯度下降
- `3_sgd回归_test.ipynb`：SGDRegressor 实验
- `4_广告案例分析.py`：广告投入与销量预测（LinearRegression vs SGDRegressor）
- 数据：`advertising.csv`

### ch05_逻辑回归
- `1_api_test.ipynb`：逻辑回归 API 练习
- `2_心脏病检测.py`：逻辑回归实现心脏病二分类
- `3_数字识别.py`：逻辑回归手写数字识别（MNIST 子集）
- 数据：`heart_disease.csv`、`train.csv`

### ch06_感知机
- `1_逻辑门电路.py`：用感知机实现与门、或门、与非门

### ch07_决策树
- `1_信息熵测试.ipynb`：信息熵与信息增益计算

## 说明

- 仓库仅用于学习记录，代码以理解算法流程为主。
- `train.csv` 为手写数字识别数据集（约 73 MB）。

## 相关仓库

- Python 学习笔记（基础语法、numpy/pandas）：https://github.com/zjb0507/python-notes
