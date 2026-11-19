import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('heart_disease.csv')
df=df.dropna()
x=df.drop(columns=df.columns[-1])
y=df[df.columns[-1]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
#数值型特征
num_features=['年龄','静息血压','胆固醇','最大心率','运动后的ST下降','主血管数量']
#类别型特征
cate_features=['胸痛类型','静息心电图结果','峰值ST段的斜率','地中海贫血']
#二元特征
binary_features=['性别','空腹血糖','运动性心绞痛']
column=ColumnTransformer([
    ('数值',StandardScaler(),num_features),
    ('分类',OneHotEncoder(drop='first'),cate_features),
    ('二元','passthrough',binary_features)
])
x_train=column.fit_transform(x_train)
x_test=column.transform(x_test)
print(x_train.shape)
