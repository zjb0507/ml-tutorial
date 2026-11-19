from sklearn.neighbors import KNeighborsRegressor  #引入回归器,拟合一个函数
x=[[2,1],[3,1],[1,4],[2,6]]
y=[0.2,0.3,0,1]
#创建knn回归模型
knn=KNeighborsRegressor(n_neighbors=2,weights='distance')
knn.fit(x,y)
x_test=[[2,8],[6,1],[1,2],[4,2]]
y_pre=knn.predict(x_test)
print(y_pre)
