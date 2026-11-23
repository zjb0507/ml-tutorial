# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams['font.family'] = 'SimHei'
# x=np.arange(0,10,0.1)
# y=np.sin(x)
# y1=np.cos(x)
# plt.plot(x,y,label='sin函数',color='r',linewidth=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sin函数图与其导函数')
# plt.plot(x,y1,label='sin导函数',color='b',linewidth=2)
# plt.legend(loc='upper left')
# plt.grid(True)
# plt.show()



#1 特征工程(生成新的特征或选择有效特征)
#低方差过滤法
# import numpy as np
# a=np.random.randn(100)
# print(np.var(a))
# print(np.mean(a))
# b=np.random.normal(1,0.1,size=100)   #均值,标准差
# print(np.var(b))
# print(np.mean(b))
# x=np.vstack((a,b))
# x1=x.T
# print(x1)
# from sklearn.feature_selection import VarianceThreshold  #方差阈值,引入方差阈值过滤器
# vt=VarianceThreshold(0.01)  #实例化这个过滤器,设置方差阈值最小为0.01
# X_filtered=vt.fit_transform(x1)   #计算每一列的方差,删除方差小于等于0.01的列
# print(X_filtered)

#相关系数法     dataframe.corrwith(对象，方法)  对象只能是一维series
# #计算皮尔逊相关系数
# import pandas as pd
# df=pd.read_csv('D:\\python_test\\data\\advertising.csv')
# print(df.head())
# print(df.describe())
# x=df[['TV','Radio','Newspaper']]
# y=df['Sales']
# print(x.head())
# print(y.head())
# df=df.drop(df.columns[0],axis=1)   #在列名这一行中跨列寻找第一列，并删除
# df=df.dropna()
# x=df.drop("Sales",axis=1)
# y=df["Sales"]
# print(x)
# print(y)
# cor=df.corr(method="pearson")  #计算数组中所有列之间两两的相关系数,返回一个二维矩阵
# print(cor)
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.heatmap(cor,annot=True,cmap="coolwarm",fmt='.2f')  #要绘制的数据，是否把具体数据写在格子里，设置颜色映射，限制小数
# plt.show()
#斯皮尔曼相关系数
# import pandas as pd
# x=pd.DataFrame({'x':[5,8,10,12,15,3,7,9,14,6]})
# y=pd.Series([55,65,70,75,85,50,60,72,80,58])
# print(x.corrwith(y,method='spearman'))

#降维
#主成分分析（通过线性变化将高维数据投影到低维空间，保留方差大的特征，就是找到所有组成特征的成分（基向量），挑出方差最大的成分，重新线性组合生成新的特征数据集）
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA   #主成分分析类pca
# x=np.random.randn(1000,3)  #有三列，所以有三维特征
# pca=PCA(n_components=2)  #实例化pca对象，指定降维成2维特征
# x_fit=pca.fit_transform(x)  #计算原始数据的协方差矩阵，特征值与特征向量，找到主成分方向，然后把原始数据投影到主成分方向
# print(x_fit.shape)
# fig=plt.figure(figsize=(12,4))   #手动创建一个画布
# ax1=fig.add_subplot(121,projection='3d')  #添加子图，把画布分为一行两列，激活左边那个，开启3d坐标
# ax1.scatter(x[:,0],x[:,1],x[:,2],color='g')  #画出原始三维数据的散点图,传入三个维度的数据
# ax1.set_title('pca(3d)')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2=fig.add_subplot(122)
# ax2.scatter(x[:,0],x[:,1],color='b')  #把两个特征传到
# ax2.set_title('pca(2d)')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# plt.show()
#生成三组基础因子
# pc1=np.random.normal(0,1,1000)
# pc2=np.random.normal(0,0.5,1000)
# pc3=np.random.normal(0,0.05,1000)
# x=np.vstack([pc1+pc2,pc1-2*pc2,0.1*pc1+3*pc2-pc3]).T
# print(x.shape)
# pca=PCA(n_components=2)
# x_fit=pca.fit_transform(x)  #降维生成主成分（也就是影响最大的成分）
# fig=plt.figure(figsize=(12,4))
# ax1=fig.add_subplot(121,projection='3d')
# ax1.scatter(x[:,0],x[:,1],x[:,2],color='r')
# ax1.set_title('3D')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2=fig.add_subplot(122)
# ax2.scatter(x_fit[:,0],x_fit[:,1],color='b')
# ax2.set_title('2D')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# plt.show()



#2 模型评估与模型选择   .fit()算出并记住统计规则,.transform()按规则转换,规则记住一次后后面可只写.transform()，不用写.fit_transform()
#损失函数（衡量模型预测误差的大小，是一个非负实值函数）
#一：使用线性回归拟合出一个函数并计算其误差，也就是损失函数
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression  #从线性模型模块导入线性回归，建立线性回归模型
# from sklearn.preprocessing import PolynomialFeatures  #导入多项式特征工具，实现曲线拟合
# from sklearn.model_selection import train_test_split  #划分数据基于训练集
# from sklearn.metrics import mean_squared_error  #导入均方误差mse
# from matplotlib import rcParams
# rcParams['font.family'] = ['SimHei']
# rcParams['axes.unicode_minus']=False
# x=np.linspace(-3,3,300).reshape(300,1)
# y=np.sin(x)+np.random.uniform(-0.1,0.1,300).reshape(300,1)
# #2.1画出三个子图
# fig=plt.figure(figsize=(20,12))
# ax1=fig.add_subplot(131)
# ax1.scatter(x,y,color='y')   #画出原始数据
# ax1.set_title('原始数据')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2=fig.add_subplot(132)
# ax2.scatter(x,y,color='y')   #画出原始数据
# ax2.set_title('原始数据')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax3=fig.add_subplot(133)
# ax3.scatter(x,y,color='y')   #画出原始数据
# ax3.set_title('原始数据')
# ax3.set_xlabel('x')
# ax3.set_ylabel('y')
# #2.2区分训练集和测试集
# trainX,testX,trainY,testY=train_test_split(x,y,test_size=0.2,random_state=42)  #规定测试集大小占百分之二十,并且固定测试集
# #2.3定义线性回归模型
# model=LinearRegression()  #实例化线性回归对象
# # #一.欠拟合(直线)
# x_train1=trainX
# x_test1=testX
# #2.4训练模型
# model.fit(x_train1,trainY)  #用直线拟合出x_train1和trainY的一条直线
# #打印查看模型参数
# print(model.coef_)   #查看斜率
# print(model.intercept_)   #查看截距
# #2.5 预测结果，计算误差
# y_pred1=model.predict(x_test1)  #根据测试集来计算出预测结果
# test_loss1=mean_squared_error(testY,y_pred1)  #比较真值与预测结果间的误差
# train_loss1=mean_squared_error(trainY,model.predict(trainX))
# #画出拟合曲线，并写出训练误差与测试误差
# ax1.plot(x,model.predict(x),c='b',linewidth=2)
# ax1.text(-3,1,f'测试集误差:{test_loss1:.4f}')
# ax1.text(-3,1.1,f'训练集误差:{train_loss1:.4f}')
#
#
# #二：恰好拟合（5次多项式）
# poly5=PolynomialFeatures(degree=5)   #创建一个五次多项式特征转换器
# x_train2=poly5.fit_transform(trainX)  #把训练数据转为多项式特征
# x_test2=poly5.transform(testX)
# print(x_train2.shape)
# #2.4训练模型(之后都会记住训练时的数据格式)
# model.fit(x_train2,trainY)  #拟合出x_train2和trainY的一条曲线,这里的x_train2是五次多项式，后面使用model时需传入相同格式的数据
# #打印查看模型参数
# # print(model.coef_)   #查看斜率
# # print(model.intercept_)   #查看截距
# #2.5 预测结果，计算误差
# y_pred2=model.predict(x_test2)  #根据测试集来计算出预测结果
# test_loss2=mean_squared_error(testY,y_pred2)  #比较真值与预测结果间的误差
# train_loss2=mean_squared_error(trainY,model.predict(poly5.transform(trainX)))
# #画出拟合曲线，并写出训练误差与测试误差
# ax2.plot(x,model.predict(poly5.transform(x)),c='b',linewidth=2)
# ax2.text(-3,1,f'测试集误差:{test_loss2:.4f}')
# ax2.text(-3,1.1,f'训练集误差:{train_loss2:.4f}')
#
#
# #三：过拟合（20次多项式）
# poly20=PolynomialFeatures(degree=20)   #创建一个20次多项式特征转换器
# x_train3=poly20.fit_transform(trainX)  #把训练数据转为多项式特征
# x_test3=poly20.transform(testX)
# print(x_train3.shape)
# #2.4训练模型(之后都会记住训练时的数据格式)
# model.fit(x_train3,trainY)  #拟合出x_train3和trainY的一条曲线,这里的x_train3是20次多项式，后面使用model时需传入相同格式的数据
# #打印查看模型参数
# # print(model.coef_)   #查看斜率
# # print(model.intercept_)   #查看截距
# #2.5 预测结果，计算误差
# y_pred3=model.predict(x_test3)  #根据测试集来计算出预测结果
# test_loss3=mean_squared_error(testY,y_pred3)  #比较真值与预测结果间的误差
# train_loss3=mean_squared_error(trainY,model.predict(poly20.transform(trainX)))
# #画出拟合曲线，并写出训练误差与测试误差
# ax3.plot(x,model.predict(poly20.transform(x)),c='b',linewidth=2)
# ax3.text(-3,1,f'测试集误差:{test_loss3:.4f}')
# ax3.text(-3,1.1,f'训练集误差:{train_loss3:.4f}')
# plt.show()

#3 正则化(惩罚过大的参数，限制模型复杂度，避免过拟合,超参数决定惩罚力度)
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression,Lasso,Ridge #从线性模型模块导入线性回归，建立线性回归模型;引入拉索回归已经岭回归，用于正则化
# from sklearn.preprocessing import PolynomialFeatures  #导入多项式特征工具，实现曲线拟合
# from sklearn.model_selection import train_test_split  #划分数据基于训练集
# from sklearn.metrics import mean_squared_error  #导入均方误差mse
# from matplotlib import rcParams
# rcParams['font.family'] = ['SimHei']
# rcParams['axes.unicode_minus']=False
# x=np.linspace(-3,3,300).reshape(300,1)
# y=np.sin(x)+np.random.uniform(-0.1,0.1,300).reshape(300,1)
# #3.1画出6个子图（三个拟合图像已经其系数的直方图）
# fig=plt.figure(figsize=(20,12))
# ax1=fig.add_subplot(231)
# ax1.scatter(x,y,color='y')   #画出原始数据
# ax1.set_title('原始数据')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2=fig.add_subplot(232)
# ax2.scatter(x,y,color='y')   #画出原始数据
# ax2.set_title('原始数据')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax3=fig.add_subplot(233)
# ax3.scatter(x,y,color='y')   #画出原始数据
# ax3.set_title('原始数据')
# ax3.set_xlabel('x')
# ax3.set_ylabel('y')
# ax4=fig.add_subplot(234)
# ax5=fig.add_subplot(235)
# ax6=fig.add_subplot(236)
# #3.2区分训练集和测试集
# trainX,testX,trainY,testY=train_test_split(x,y,test_size=0.2,random_state=42)  #规定测试集大小占百分之二十,并且固定测试集
# #3.3定义线性回归模型
# model=LinearRegression()  #实例化线性回归对象
# poly20=PolynomialFeatures(degree=20)   #创建一个20次多项式特征转换器
# x_train=poly20.fit_transform(trainX)  #把训练数据转为多项式特征,fit的作用是记住规则（记下参数），transform是转换
# x_test=poly20.transform(testX)
# print(x_train.shape)
# #3.4训练模型(之后都会记住训练时的数据格式)
# model.fit(x_train,trainY)  #拟合出x_train3和trainY的一条曲线,这里的x_train3是20次多项式，后面使用model时需传入相同格式的数据
# #打印查看模型参数
# print(model.coef_)   #查看斜率
# # print(model.intercept_)   #查看截距
# #3.5 预测结果，计算误差
# y_pred=model.predict(x_test)  #根据测试集来计算出预测结果
# test_loss=mean_squared_error(testY,y_pred)  #比较真值与预测结果间的误差
# train_loss=mean_squared_error(trainY,model.predict(poly20.transform(trainX)))
# #画出拟合曲线，并写出训练误差与测试误差
# ax1.plot(x,model.predict(poly20.transform(x)),c='b',linewidth=2)
# ax1.text(-3,1,f'测试集误差:{test_loss:.4f}')
# ax1.text(-3,1.1,f'训练集误差:{train_loss:.4f}')
# ax4.bar(np.arange(21),model.coef_.reshape(21),color='b')
# ax4.set_title('系数直方图')
# ax4.set_xlabel('x')
# ax4.set_ylabel('y')

#定义lasso正则化模型（特征选择，减少特征量，避免过拟合）
# lasso=Lasso(alpha=0.01,max_iter=1000000)  #实例化正则化模型，传入超参数，决定惩罚力度,设置最大迭代次数
# poly20=PolynomialFeatures(degree=20)   #创建一个20次多项式特征转换器
# x_train=poly20.fit_transform(trainX)  #把训练数据转为多项式特征
# x_test=poly20.transform(testX)
# #4训练模型
# lasso.fit(x_train,trainY)
# #5 预测结果，计算误差
# y_pred1=lasso.predict(x_test)
# test_loss1=mean_squared_error(testY,y_pred1)
# train_loss1=mean_squared_error(trainY,lasso.predict(poly20.transform(trainX)))
# #画出拟合曲线，并写出训练误差与测试误差
# ax2.plot(x,lasso.predict(poly20.transform(x)),c='b',linewidth=2)
# ax2.text(-3,1,f'测试集误差:{test_loss1:.4f}')
# ax2.text(-3,1.1,f'训练集误差:{train_loss1:.4f}')
# ax5.bar(np.arange(21),lasso.coef_.reshape(21),color='b')
# ax5.set_title('lasso正则化系数直方图')
# ax5.set_xlabel('x')
# ax5.set_ylabel('y')

#定义Ridge正则化模型（使拟合曲线更加平滑）
# ridge=Ridge(alpha=0.1)#实例化正则化模型，传入超参数，决定惩罚力度,设置最大迭代次数
# poly20=PolynomialFeatures(degree=20)   #创建一个20次多项式特征转换器
# x_train=poly20.fit_transform(trainX)  #把训练数据转为多项式特征
# x_test=poly20.transform(testX)
# #4训练模型
# ridge.fit(x_train,trainY)
# #5 预测结果，计算误差
# y_pred2=ridge.predict(x_test)
# test_loss2=mean_squared_error(testY,y_pred2)
# train_loss2=mean_squared_error(trainY,ridge.predict(poly20.transform(trainX)))
# #画出拟合曲线，并写出训练误差与测试误差
# ax3.plot(x,ridge.predict(poly20.transform(x)),c='b',linewidth=2)
# ax3.text(-3,1,f'测试集误差:{test_loss2:.4f}')
# ax3.text(-3,1.1,f'训练集误差:{train_loss2:.4f}')
# ax6.bar(np.arange(21),ridge.coef_.reshape(21),color='b')
# ax6.set_title('Ridge正则化系数直方图')
# ax6.set_xlabel('x')
# ax6.set_ylabel('y')
# plt.show()


#4 梯度下降
# import matplotlib.pyplot as plt
# import numpy as np
# def f(x):
#     return (x**2-2)**2
# def g(x):
#     return 4*x**3-8*x
# x_list=[]
# y_list=[]
# x=1
# alpha=0.01
# for i in range(100):
#     y=f(x)
#     x_list.append(x)
#     y_list.append(y)
#     print(f'x:{x},y:{y}')
#     x-=g(x)*alpha
# print(y_list)
# x_1=np.arange(0.9,1.5,0.01)
# # plt.plot(x_1,f(x_1),c='r')
# # plt.plot(x_list,y_list,c='b')
# # plt.scatter(x_list,y_list,c='g')
#
# # 局部放大
# fig,ax=plt.subplots(1,2,figsize=(15,4))
# ax[0].plot(x_1,f(x_1),c='r')
# ax[0].plot(x_list,y_list,c='b')
# ax[0].scatter(x_list,y_list,c='g')
# x=np.arange(1.389,1.415,0.001)
# x_list2=x_list[20:]
# y_list2=y_list[20:]
# ax[1].plot(x,f(x),c='r')
# ax[1].plot(x_list2,y_list2,c='b')
# ax[1].scatter(x_list2,y_list2,c='g')
# plt.show()


#5分类评价指标
#5.1 混淆矩阵
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# #定义类别标签
# labels=['猫','狗']
# #定义数据
# y_true=['猫','猫','猫','猫','猫','猫','狗','狗','狗','狗']
# y_pred=['猫','猫','狗','猫','猫','猫','猫','猫','狗','狗']
# #得到混淆矩阵
# matrix=confusion_matrix(y_true,y_pred,labels=labels)
# print(matrix)   #矩阵标签：左边第一列标签为真实值，上面第一行标签为预测值
# print(pd.DataFrame(matrix,index=labels,columns=labels))
# sns.heatmap(matrix,annot=True,fmt='.2f')  #annot决定是否显示数字，fmt指保留几位小数
# plt.show()
# #准确率（分类正确的比例）
# from sklearn.metrics import accuracy_score
# accuracy=accuracy_score(y_true,y_pred)
# print(accuracy)  #计算准确率
# #精确率也叫查准率（预测为正例里真正例的比例）
# from sklearn.metrics import precision_score
# precision=precision_score(y_true,y_pred,pos_label='猫')  #需要指定正例标签
# print(precision)
# #召回率（实际为正类的样本中预测为正类的比例）
# from sklearn.metrics import recall_score
# recall=recall_score(y_true,y_pred,pos_label="猫")
# print(recall)
# #F1分数（精确率和召回率的调和平均）
# from sklearn.metrics import f1_score
# f1=f1_score(y_true,y_pred,pos_label="猫")
# print(f1)
#
# #5.2 评估报告（直接给出准确率，精确率，召回率，f1分数等）
# from sklearn.metrics import classification_report
# classification=classification_report(y_true,y_pred,labels=labels,target_names=['猫'])  #可指定关心的正类标签,需传入数组（列表）形式，不传入时全部算一次
# print(classification)


