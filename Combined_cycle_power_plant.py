#通过本地文件导入数据集进行测试
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 400


datapath=r"C:\Users\neko\PycharmProjects\PythonProject\dataset\combined_cycle_power_plant.csv"
fd=pd.read_csv(datapath).sample(1000,random_state=42)#取样1000行，每次取样结果固定,本次取样代码是42，之后若想复现，令其值为42即可
fd.info()

fd=fd.rename(columns={#重命名列名
    'AT':'环境温度',
    'V':'排气真空度',
    'AP':'环境压强',
    'RH':'相对湿度',
    'PE':'净每小时发电量',
})
print(f"统计数据：{fd.describe()}")#包括一些最大值，最小值，平均值等
fd.isnull().any()#判断是否有缺失
fd.dropna()#处理缺失行
fd.drop_duplicates()#去掉重复值
print(fd.head())
#m=fd.shape[0]#获取到样本数
target_name=fd.columns[-1]
#决策树线性回归模型
X=fd.drop(columns=[target_name])
#错误写法：X=fd.values,这样会把列名丢带，后面需手动写一遍
y=fd[target_name]
#错误写法：y=fd.columns，y变成了五个列名，而不是PE
#还可以这么做：
#X = fd.iloc[:, :-1].values      # 相当于 diabetes.data：除最后一列外全部作为特征
#y = fd.iloc[:, -1].values       # 相当于 diabetes.target：最后一列作为目标
#print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

reg=DecisionTreeRegressor(#创建决策树回归模型
    criterion='squared_error',
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=5,
    random_state=42
)
reg.fit(X_train,y_train)#训练模型
y_pred=reg.predict(X_test)

print("特征重要性：")
for name,importance in zip(X.columns,reg.feature_importances_):
    print(f"{name} : {importance*100:.3}%")#查看各特征对于该模型的影响度

plt.title("电厂循环发电量")
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.scatter(y_test,y_pred,label="离散点")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r-',label="基准线",alpha=0.8)
plt.show()
#可视化决策树
plt.figure(figsize=(18, 10))
plot_tree(reg,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          fontsize=9,
          max_depth=3)  # 只画前3层，方便查看
plt.title("Decision Tree Regressor - 前3层", fontsize=14)

#绘制特征影响比重图
#imp=reg.feature_importances_#获取重要性
#print(imp)
plt.figure(figsize=(18, 10))
plt.title("特征影响分布图")
plt.xlabel("特征")
plt.ylabel("比重")
plt.tight_layout()
plt.legend()
plt.bar(X.columns,reg.feature_importances_,align='center')
plt.plot(X.columns,reg.feature_importances_,'ro')
plt.show()

