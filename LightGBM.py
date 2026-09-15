import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=400

X,y=make_regression(n_samples=500,n_features=5,noise=80,random_state=42)#noise为噪声，噪声越大，MAE，谁都不好预测

X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=0.4,random_state=42)#60%用来训练，40%先不管
x_val,x_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=42)#把那40%再用来继续划分：20%验证+20%测试，防止“背答案 ”

#model=LGBMRegressor(random_state=42)
#让模型自己决定建多少树
model=LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=255,random_state=42)#看过拟合曲线，因此故意把模型建的很大(255个叶子)
#model=LGBMRegressor(n_estimators=3000,learning_rate=0.05,random_state=42)#n_estimators为树的最大数量上限，learning_rate为每棵树的影响率，越小越稳也越需要更多的树
model.fit(
    X_train,y_train,#用训练集来学习
    eval_set=[(X_train,y_train),(x_val,y_val)],#同时记录训练集和验证集的误差
    #eval_set=[(x_val,y_val)],#额外告诉用验证集来考核
    eval_metric="mae",#考核指标：mae
    callbacks=[lgb.early_stopping(50,verbose=False)]#验证集连续50轮没变好的话就停下来
    #也就是说，每训练一轮就让验证集"考"一次，连续 50 轮都没有进步就自动停止建树，并把成绩最好的那一轮记为最佳模型。
)#早停必须要有验证集可以看，就是fit(里eval_set那行，如果没有就没有东西可以监督，它俩配套使用)

# ---- 侦查：先看清结构 ----
hist = model.evals_result_
print("评估集名字:", list(hist.keys()))
for name, metrics in hist.items():
    print(f"  {name} 的指标:", list(metrics.keys()))

# ---- 画图 ----
def curve(hist, i):
    name = list(hist.keys())[i]
    metric = list(hist[name].keys())[0]
    return hist[name][metric]

plt.plot(curve(hist, 0), label="训练集")
plt.plot(curve(hist, 1), label="验证集")
plt.xlabel("树的数量")
plt.ylabel("MAE")
plt.legend()
plt.show()

pred=model.predict(x_test)

print("自动停在第", model.best_iteration_, "棵树")#在建了多少颗树是验证误差最小，图像最低点
print("测试集 MAE:", mean_absolute_error(y_test, pred))#最后使用测试集
#print(f"测试集MAE：{mean_absolute_error(y_test,pred)}")

#绘制曲线
