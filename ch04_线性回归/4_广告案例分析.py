import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   #从模型选择模块引入数据集划分函数
from sklearn.preprocessing import StandardScaler   #从数据预处理模块引入标准化器
from sklearn.linear_model import LinearRegression,SGDRegressor  #从线性模型模块引入两种回归模型
from sklearn.metrics import mean_squared_error   #模型评估指标模块引入均方误差
#1.读取数据集
df=pd.read_csv('advertising.csv')
df=df.drop(columns=df.columns[0])  #columns指定删列
x=df.drop(columns='Sales')
y=df['Sales']
#2.划分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
linear=LinearRegression()
linear.fit(x_train,y_train)
#3.特征工程:标准化
scaler=StandardScaler()  #定义一个标准化转换器
x_train=scaler.fit_transform(x_train)  #返回标准化的数组，需要赋值保存
x_test=scaler.transform(x_test)  #标准化训练集与测试集
#4.创建模型并训练
#4.1.正规方程法
model_lr=LinearRegression()
model_lr.fit(x_train,y_train)  #训练模型
print('lr斜率：',model_lr.coef_)
print('lr截距：',model_lr.intercept_)
#4.2.随机梯度下降法
model_sgd=SGDRegressor()
model_sgd.fit(x_train,y_train)
print('sgd斜率：',model_sgd.coef_)
print('sgd截距：',model_sgd.intercept_)
#5.预测
y_pred1=model_lr.predict(x_test)
y_pred2=model_sgd.predict(x_test)
error1=mean_squared_error(y_test,y_pred1)
error2=mean_squared_error(y_test,y_pred2)
print('lr模型误差：',error1)
print('sgd模型误差：',error2)