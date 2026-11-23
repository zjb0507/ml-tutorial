import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import KMeans       #从聚类模块导入k均值聚类，用于自动划分k个分类（簇）
from sklearn.datasets import make_blobs    #从datasets模块（用于加载现成数据集或生成数据集）引入生成球形簇数据集函数
rcParams['font.family']=['SimHei']
rcParams['axes.unicode_minus'] = False   #关闭unicode负号


#1.生成数据
x,y=make_blobs(n_samples=300,centers=3,cluster_std=2,random_state=42)  #一共生成300个样本点，三个簇，标准差为2，返回样本特征(点的横纵坐标)和真实标签

#画出散点图
fig,ax=plt.subplots(2,figsize=(8,8))  #一次性生成画布和子图：两个子图，画布大小为（8，8）
ax[0].scatter(x[:,0],x[:,1],c='gray',s=50,label='原始数据')
ax[0].set_title('原始数据')
ax[0].legend()

#2.定义模型并聚类
model=KMeans(n_clusters=3)   #分成三个聚类
model.fit(x)

#3.获取聚类结果
centers=model.cluster_centers_   #保存各个簇的中心点坐标(第一列为横坐标，第二列为纵坐标)

#4.预测，得到所有样本点的分类标签
y_pred=model.predict(x)    #不同的点获取对应的标签

#画出聚类结果
ax[1].scatter(x[:,0],x[:,1],c=y_pred,s=50,label='分类数据')
ax[1].scatter(centers[:,0],centers[:,1],c='red',s=50,label='中心点')
ax[1].set_title('分类数据')
ax[1].legend()
plt.show()