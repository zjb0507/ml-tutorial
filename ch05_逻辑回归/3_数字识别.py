import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler   #引入归一化缩放器
from sklearn.linear_model import LogisticRegression  #引入逻辑回归

#1.加载数据集
df=pd.read_csv('train.csv')
print(df.head())

#测试图像
# df=df.iloc[10,1:].values  #把第十一行的像素点信息读取出来，并从dataframe转为numpy数组
# df=df.reshape(28,28)
# plt.imshow(df,cmap='gray')   #imshow用来展示二维数组图像，camp指定用灰度色彩进行映射
# plt.show()

#2.划分数据集
x=df.drop(columns=df.columns[0])
y=df[df.columns[0]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#3.特征工程
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)     #归一化后dataframe数组变为numpy数组

#4.定义模型并训练
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#5.模型评估
print(model.score(x_test,y_test))   #计算分类的准确率
print(model.predict_proba(x_test))   #获取每个样本属于每一类的概率

#6.预测
digit=x_test[192,:].reshape(-1,784)  #predict要求传入二维数组,这里索引时会降维成一维，所以要reshape
print(model.predict(digit))     #计算预测标签
print(y_test.iloc[192])    #对比真实标签
print(model.predict_proba(digit))    #计算出现所有标签的概率

#7.画图
plt.imshow(digit.reshape(28,28),cmap='gray')
plt.show()
