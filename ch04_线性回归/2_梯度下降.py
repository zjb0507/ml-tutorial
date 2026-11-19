import numpy as np

#定义损失函数
def J(beta):
    return np.sum((x@beta-y)**2)/n  #计算损失函数也就是均方误差
#定义计算梯度的函数
def gradient(beta):
    return (x.T@(x@beta-y))*2/n
#1.定义数据
#自变量，每周学习时间
x=np.array([[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]])
#因变量，数学考试成绩
y=np.array([[55],[65],[70],[75],[85],[50],[60],[72],[80],[58]])
n=x.shape[0]
one=np.ones((n,1),dtype=int)
#2.处理数据
x=np.hstack((one,x))
print(x.shape)
#3.初始化参数以及超参数
alpha=0.01
beta=np.array([[1],[1]])
iter=10000
#定义列表，保留参数变化轨迹
beta0=[]
beta1=[]
#重复迭代，算出损失函数近似最小时拟合函数的参数向量（分量从上到下为x从小到大次数的系数）
while iter>0:
    beta0.append(beta[0,0])
    beta1.append(beta[1,0])
    J1=J(beta)
    beta=beta-alpha*gradient(beta)
    J2=J(beta)
    if np.abs(J1-J2)<1e-20:
        break
print(beta)
print(J(beta))
# 绘图
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.figure(figsize=(20,10))
rcParams['font.family'] = 'SimHei'
plt.scatter(beta0,beta1,label='参数系数',c='r',)
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.title('参数变化轨迹')
plt.legend()
plt.grid(True)
plt.show()

