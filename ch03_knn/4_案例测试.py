import pandas as pd
#1.引入数据
hd=pd.read_csv('heart_disease.csv')
print(hd.shape)
#处理缺失值
hd=hd.dropna(axis=0,how='any')
print(hd.shape)
print(hd.info())  #打印出每一列的信息

#2.划分训练集与测试集
from sklearn.model_selection import train_test_split
#划分特征与标签
x=hd.drop('是否患有心脏病',axis=1)  #指定删掉哪一列
y=hd['是否患有心脏病']
#划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#3.特征工程
from sklearn.compose import ColumnTransformer   #按列分别做预处理的转换器
from sklearn.preprocessing import StandardScaler,OneHotEncoder  #引入标准化缩放器和独热编码器
#数值型特征
num_features=['年龄','静息血压','胆固醇','最大心率','运动后的ST下降','主血管数量']
#类别型特征
cate_features=['胸痛类型','静息心电图结果','峰值ST段的斜率','地中海贫血']
#二元特征
binary_features=['性别','空腹血糖','运动性心绞痛']

#创建并定义一个列转换器
column_transformer=ColumnTransformer(transformers=[('num',StandardScaler(),num_features),  #对num_features这几列进行标准化缩放
                                                   ('cat',OneHotEncoder(drop='first'),cate_features),  #对cate_features这几列进行独热编码,然后删掉第一个类别对应的那一列
                                                   #独热编码是把分类转化为0，1组成的向量
                                                   ('binary',"passthrough",binary_features)])     #binary_features这几列不用操作

#特征转换
x_train=column_transformer.fit_transform(x_train)  #会返回一个numpy数组
x_test=column_transformer.transform(x_test)
print(x_train.shape,x_test.shape)

# #4.创建模型
# from sklearn.neighbors import KNeighborsClassifier  #引入knn分类器，用来做分类
# knn=KNeighborsClassifier(n_neighbors=5)  #参考附近的五个点
#
# #5.训练模型
# knn.fit(x_train,y_train)
#
# #6.模型评估，计算预测准确率
# score=knn.score(x_test,y_test) #计算预测准确率
# print(score)
#
# # #7.保存模型
# import joblib
# joblib.dump(knn,'knn.joblib')  #要保存的模型以及保存位置
# # 加载模型，对新数据进行预测
# knn_loaded=joblib.load('knn.joblib')
# y_pred=knn_loaded.predict(x_test[[10]])  #需要传入二维数据
# print(f'真实类别：{y_test[10]},测试类别：{y_pred}')


from sklearn.model_selection import GridSearchCV
#4.创建knn分类器
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()  #超参数在下面列表中选取来交叉验证

# 定义网格搜索参数列表
param_grid={'n_neighbors':[1,2,3,4,5,6,7,8,9],"weights":['uniform','distance']}  #超参数列表,告诉模型要试哪些超参数:k值与权重
#实例化一个网格搜索器对象，记住训练规则
grid_search=GridSearchCV(estimator=knn,param_grid=param_grid,cv=5) #传入要调参的模型，参数范围，指定五折交叉验证

#模型训练，进行交叉验证，选取最优参数
grid_search.fit(x_train,y_train)

#打印模型评估结果
result=pd.DataFrame(grid_search.cv_results_).to_string()
print(result)
#直接获取最佳模型与得分
print(grid_search.best_estimator_)  #打印模型
print(grid_search.best_params_)  #打印参数
print(grid_search.best_score_)  #打印得分

#使用最佳模型，进行测试评估
knn=grid_search.best_estimator_
print(knn.score(x_test,y_test))  #计算预测准确率