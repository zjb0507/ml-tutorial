import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import KMeans       #从聚类模块导入k均值聚类，用于自动划分k个分类（簇）
from sklearn.datasets import make_blobs    #从datasets模块（用于加载现成数据集或生成数据集）引入生成球形簇数据集函数
import os     #导入操作系统模块，用来设置环境变量
from sklearn.metrics import silhouette_score,calinski_harabasz_score   #从模型评估模块引入轮廓系数和卡林斯基——哈拉巴斯指数
os.environ['OMP_NUM_THREADS']='2'    #设置openmp线程数为2
rcParams['font.family']=['SimHei']
rcParams['axes.unicode_minus'] = False   #关闭unicode负号


#1.生成数据
x,y=make_blobs(n_samples=300,centers=3,cluster_std=2,random_state=42)  #一共生成300个样本点，三个簇，标准差为2，返回样本特征(点的横纵坐标)和真实标签

#2.定义模型并聚类
model=KMeans(n_clusters=3)   #分成三个聚类
model.fit(x)

#3.获取聚类结果
centers=model.cluster_centers_   #保存各个簇的中心点坐标(第一列为横坐标，第二列为纵坐标)

#4.预测，得到所有样本点的分类标签
y_pred=model.predict(x)    #不同的点获取对应的标签

#画出聚类结果
plt.scatter(x[:,0],x[:,1],c=y_pred,s=50,label='分类数据')
plt.scatter(centers[:,0],centers[:,1],c='red',s=50,label='中心点')
plt.title('分类数据')
plt.legend()
plt.show()

#打印评价指标
print(model.inertia_)      #计算簇内平方和
print(silhouette_score(x,y_pred))    #计算轮廓系数，在-1到1间，越接近1，分类效果越好
print(calinski_harabasz_score(x,y_pred))     #计算ch指数，数值越大，代表簇内越紧凑，簇间分的越开