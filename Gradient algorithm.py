import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import alpha

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=400

print("线性回归一元梯度下降法============================")
x=np.array([1,2,3,4,5],dtype=float)
y=np.array([2,4,6,8,10],dtype=float)
w,b=0.0,0.0#模型参数，初始为零
lr=0.01#学习率阿尔法
epochs=4600#迭代次数为4600次
m=len(x)#样本大小，即数据个数
best_loss=np.inf#初始定义一个无限大的loss,便于后续找到最小的loss
for i in range(epochs):
    y_pre=w*x+b#模型函数
    error=y_pre-y#误差
    loss=np.mean(error**2)

    #Jwb=1/(2*m)*np.sum(np.square(error))#代价函数

    if loss<best_loss:
        best_loss=loss#找到最小的loss
        best_w=w#最佳的w
        best_b=b#最佳的b

    tmp_w=w-lr*((1/m)*np.sum(error*x))#梯度下降,同时更新
    tmp_b=b-lr*((1/m)*np.sum(error))

    w=tmp_w
    b=tmp_b

    if i%200==0:
        print(f"epoch {i}: loss={loss:.4},w={w:.6f},b={b:.6f}")
print(f"最优：loss={best_loss:.8f},best_w={best_w:.4f},best_b={best_b:.4f}")
print(f"模型为：y={best_w:.4f}x+{best_b:.4f}")

#图像
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.plot(x,y,"ro",label="离散点",alpha=0.8)
plt.plot(x,y_pre,"g--",label=f"y_pre={best_w:.3}x+{best_b:.3}",alpha=0.8)
plt.legend()
plt.show()


print("线性回归二元梯度下降法============================")

def binary_linear_regression_gd(X,y,lr=0.01,epochs=6000):#X是一个二元数组
    w1,w2=0.0,0.0
    b=0.0
    lr=0.01
    n=X.shape[0]#取出元组第一个数，即样本个数，如对于(100,2)形状的元组，有100行2列，即100个样本
    best_loss=np.inf#初始定义一个无限大的损失值

    for i in range(epochs):
        y_pre=w1*X[:,0]+w2*X[:,1]+b#X是一个二元数组，第一列(下标为0)为x1，第二列为x2
        error=y_pre-y#计算误差
        #计算梯度参数
        tmp_w1=w1-lr*((2/n)*np.dot(error,X[:,0]))#np.dot(a,b)用来将a，b俩个素组对应元素相乘后相加，用来高效计算权重的梯度
        tmp_w2=w2-lr*((2/n)*np.dot(error,X[:,1]))
        tmp_b=b-lr*((2/n)*np.sum(error))
        loss = np.mean(error ** 2)  # 计算平方误差
        #同步更新参数
        w1=tmp_w1
        w2=tmp_w2
        b = tmp_b

        if loss<best_loss:#最佳数值
            best_loss=loss
            best_w1=w1
            best_b=b
            best_w2=w2

        if i%200==0:#每迭代200次打印一次
            print(f"epoch为：{i},loss={loss:.4},w={w1:.6},w2={w2:.6f},b={b:.6}")

    return best_loss,best_w1,best_w2,best_b

#测试数据
x1=np.random.uniform(low=0,high=10,size=100)#从0到10的随机100个不重复的数
x2=np.random.uniform(low=0,high=10,size=100)
X=np.column_stack((x1,x2))#把多个一维数组拼接起来，由于是二元模型，因此需要二维x的数组。把x1，x2元组分别竖起来后拼接成二维数组

y=3*x1+2*x2+5+np.random.normal(0,1,100)#真实模型，其中np.random.normal(0,1,100)为可能的噪声，即误差
print("真实模型大约为：y=3x1+2x2+5")

loss,w1,w2,b=binary_linear_regression_gd(X,y,lr=0.01,epochs=6000)#调用线性二元梯度下降函数

print("*"*60)
print(f"最优结果为：best_loss={loss:.6f},best_w1={w1:.4f},best_w2={w2:.4f},best_b={b:.4f}")
#3D图像
ax=plt.subplot(1,1,1,projection='3d')
X,Y=np.meshgrid(np.linspace(x1.min(),x1.max(),50),np.linspace(x2.min(),x2.max(),50))#x1，x2是100个随机散点，不是网格坐标，meshgrid应该用等间距坐标
Z=w1*X+w2*Y+b

ax.set_title(f"二元模型:Z={w1:.4f}x1+{w2:.4f}x2+{b:.4f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.scatter(x1,x2,y,color="red",label="data")

ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,label="surface")
plt.tight_layout()
plt.legend()
plt.show()

