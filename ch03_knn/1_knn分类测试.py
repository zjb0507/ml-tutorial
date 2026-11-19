from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

#准备数据
x=np.array([[2,1],[3,1],[1,4],[2,6]])
y=np.array([0,1,0,1])
#定义knn分类模型
knn=KNeighborsClassifier(n_neighbors=2,weights='distance')   #选最近的2个点参考，距离为权重
#训练模型
knn.fit(x,y)
#模型预测
x_pre=np.array([[4,9]])
x_class=knn.predict(x_pre)
print(x_class)
#进行画图
rcParams['font.family'] = ['SimHei']
x0=x[y==0]  #y等于0的点
x1=x[y==1]  #y等于1的点
x0_x=x0[:,0]
x1_x=x1[:,0]
x0_y=x0[:,1]
x1_y=x1[:,1]
x_color='red'if x_class==0 else 'blue'
plt.axis('equal')  #让x，y轴等比例
plt.grid(True)
plt.scatter(x0_x,x0_y,c='r',label='标签为0')
plt.scatter(x1_x,x1_y,c='b',label='标签为1')
plt.scatter(x_pre[:,0],x_pre[:,1],c=x_color)
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.title('标签预测散点图')
plt.show()
