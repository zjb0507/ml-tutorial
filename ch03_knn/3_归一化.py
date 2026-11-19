# from sklearn.preprocessing import MinMaxScaler  #引入归一化缩放器（缩放到同一范围，0~1或-1~1）
# x=[[2,1],[3,1],[1,4],[2,6]]
# #定义归一化器
# scaler=MinMaxScaler(feature_range=(-1,1))  #实例化缩放器，指定缩放范围
# x_scaled=scaler.fit_transform(x)  #fit适配方法定义的参数，transform开始转换
# print(x_scaled)

# from sklearn.preprocessing import StandardScaler #引入标准化缩放器(均值为0，标准差为1)
# import numpy as np
# x=[[2,1],[3,1],[1,4],[2,6]]
# standard=StandardScaler()   #实例化
# x_scaled=standard.fit_transform(x)   #把x按标准化规则进行特征转换
# print(x_scaled)
# X=np.array(x_scaled)
# print(np.shape(X))
# print(np.mean(X))
# print(np.std(X))
# print((X-np.mean(X))/np.std(X))
