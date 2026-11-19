from sklearn.datasets import make_classification  #引入自动生成分类数据集的函数
from sklearn.metrics import classification_report #用于生成分类评估报告
from sklearn.model_selection import train_test_split #划分数据集
from sklearn.linear_model import LogisticRegression #引入逻辑回归分类模型
# 1. 生成数据（x相当于特征，y相当于结果（标签））
X,Y=make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42) #指定数据行数，列数，分类数，已经固定随机种子，返回特殊矩阵和每一行的标签
print(X)
print(Y)  #一个一维数组
#划分数据集
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.3,random_state=42)
#定义分类模型
model=LogisticRegression()
#模型训练
model.fit(X_train,Y_train)
#预测
Y_pred=model.predict(X_test)  #Y_pred和Y_test用来进行分类评估
#生成评估报告
classification=classification_report(Y_test,Y_pred)
print(classification)
#获取预测正类的概率值(会给出两列概率值，分别是负类和正类）
prb=model.predict_proba(X_test)
print(prb[:,1])
#计算auc值（roc曲线下的面积）
from sklearn.metrics import roc_auc_score
roc_auc=roc_auc_score(Y_test,prb[:,1])  #给出正确的标签以及模型预测正类的概率
print(roc_auc)
