import pandas as pd
from sklearn.model_selection import train_test_split       #引入数据集划分工具
from sklearn.preprocessing import StandardScaler,OneHotEncoder   #从预处理模块引入标准化器：处理数值特征；独热编码：处理分类特征
from sklearn.compose import ColumnTransformer   #引入列转换器：对不同的列做不同的预处理
from sklearn.linear_model import LogisticRegression #引入逻辑回归器

#1.读入数据
df=pd.read_csv('heart_disease.csv')
print(df.head())
df=df.dropna()

#2.划分数据集
x=df.drop(columns=df.columns[-1])
y=df['是否患有心脏病']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#3.特征工程
#数值型特征
num_features=['年龄','静息血压','胆固醇','最大心率','运动后的ST下降','主血管数量']
#类别型特征
cate_features=['胸痛类型','静息心电图结果','峰值ST段的斜率','地中海贫血']
#二元特征
binary_features=['性别','空腹血糖','运动性心绞痛']

#创建并定义一个列转换器
column_transformer=ColumnTransformer(transformers=[('num',StandardScaler(),num_features),  #对num_features这几列进行标准化缩放
                                                   ('cat',OneHotEncoder(drop='first'),cate_features),  #对cate_features这几列进行独热编码,然后删掉第一个类别对应的那一列编码，避免过度线性化
                                                   #独热编码是把分类转化为0，1组成的向量
                                                   ('binary',"passthrough",binary_features)])     #binary_features这几列不用操作

#特征转换
x_train=column_transformer.fit_transform(x_train)  #会返回一个numpy数组
x_test=column_transformer.transform(x_test)
print(x_train.shape,x_test.shape)

#4.模型定义和训练
model=LogisticRegression()
model.fit(x_train,y_train)

#5.计算准确率，评估模型
print(model.score(x_test,y_test)) #计算得分，也就是算准确率