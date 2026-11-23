# 感知机：找一个超平面，将线性可分的两类样本分开
# 实现与门
# def And(x1,x2):      #传入输入节点信号
#     w1,w2,theta=0.5,0.5,0.7   #定义两个权重和阈值
#     if w1*x1+w2*x2<=theta:    #对感知机输出神经元进行条件判断
#         return 0
#     else:
#         return 1

#另一种实现形式
import numpy as np
def And(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    sum1=w@x+b     #一维矩阵的矩阵乘法等同于求内积
    if sum1<=0:
        return 0
    else:
        return 1
#测试
print(And(0,0))
print(And(0,1))
print(And(1,0))      #与门有0就0
print(And(1,1),'\n')      #与门全1就1

#与非门
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    sum1=w@x+b     #一维矩阵的矩阵乘法等同于求内积
    if sum1<=0:
        return 0
    else:
        return 1
#测试
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))      #与非门有0就1
print(NAND(1,1),'\n')      #与门全1就0


#或门
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.4
    sum1=w@x+b     #一维矩阵的矩阵乘法等同于求内积
    if sum1<=0:
        return 0
    else:
        return 1
#测试
print(OR(0,0))      #或门全0就0
print(OR(0,1))
print(OR(1,0))
print(OR(1,1),'\n')      #或门有1就1

#多层感知机
#异或门（与非门和或门相与）
def NHUO(x1,x2):
    x=np.array([NAND(x1,x2),OR(x1,x2)])
    w=np.array([0.5,0.5])
    b=-0.7
    mj=w@x+b
    if mj<=0:
        return 0
    else:
        return 1
#测试
print(NHUO(0,0))
print(NHUO(0,1))
print(NHUO(1,0))     #不同就1
print(NHUO(1,1),'\n')    #相同就0