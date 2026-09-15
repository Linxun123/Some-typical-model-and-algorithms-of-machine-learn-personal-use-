import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.datasets import make_regression#回归任务数据集
from sklearn.datasets import load_breast_cancer#分类任务数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,roc_auc_score,accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=400
#回归任务
X,y=make_regression(n_samples=500,n_features=5,noise=80,random_state=42)#noise为噪声，噪声越大，MAE，谁都不好预测

X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=0.4,random_state=42)#60%用来训练，40%先不管
x_val,x_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=42)#把那40%再用来继续划分：20%验证+20%测试，防止“背答案”

#model=LGBMRegressor(random_state=42)
#让模型自己决定建多少树
model=LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    min_child_samples=20
)#看过拟合曲线，因此故意把模型建的很大(255个叶子)，但本数据集下最多只有15个叶子，因此不起作用，可以自己更改数据集样本数，或者限制：min_child_samples=1
#model=LGBMRegressor(n_estimators=3000,learning_rate=0.05,random_state=42)#n_estimators为树的最大数量上限，learning_rate为每棵树的影响率，越小越稳也越需要更多的树
model.fit(
    X_train,y_train,#用训练集来学习
    eval_set=[(X_train,y_train),(x_val,y_val)],#同时记录训练集和验证集的误差，因此可以画出俩条曲线
    #eval_set=[(x_val,y_val)],#额外告诉用验证集来考核
    eval_metric="mae",#考核指标：mae
    callbacks=[lgb.early_stopping(50,verbose=False)]#验证集连续50轮没变好的话就停下来
    #也就是说，每训练一轮就让验证集"考"一次，连续 50 轮都没有进步就自动停止建树，并把成绩最好的那一轮记为最佳模型。
)#早停必须要有验证集可以看，就是fit(里eval_X,eval_y那行，如果没有就没有东西可以监督，它俩配套使用)

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

plt.title("真实值 vs 预测值")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r-',alpha=0.3,label='基准线')
plt.xlabel("真实值")
plt.ylabel("预测值")
pred=model.predict(x_test)
plt.scatter(y_test,pred)
plt.legend()
plt.show()

print("自动停在第", model.best_iteration_, "棵树")#在建了多少颗树是验证误差最小，图像最低点
print("测试集 MAE:", mean_absolute_error(y_test, pred))#最后使用测试集
#print(f"测试集MAE：{mean_absolute_error(y_test,pred)}")

#尝试四种叶子数量，当然其他俩个参数学习率和叶子最少样本数也可以如此操作进行对比
for nl in [7,31,255,550]:
    m=LGBMRegressor(
        n_estimators=300,
        random_state=42,
        learning_rate=0.1,
        min_child_samples=1,#由于本数据集只有300个样本，默认最小叶子样本数为20，因此最多分300/20=15个结点就不再分，故num_leaves大于15的数值时都不再起作用，因此这里将其设为1
        verbose=-1,#关闭自带的info，防止刷屏
        num_leaves=nl,#改变叶子数
    ).fit(X_train,y_train)

    train_mea=mean_absolute_error(y_train,m.predict(X_train))#训练集误差
    val_mea=mean_absolute_error(y_val,m.predict(x_val))#验证集误差

    print(f"num_leaves:{nl}训练集误差：{train_mea}--验证集误差:{val_mea}")
print("over")

#分类任务
Xc,yc=load_breast_cancer(return_X_y=True)#569个样本，判断肿瘤是恶性还是良性
#依旧三段切
Xc_train,Xc_tmp,y_train,y_tmp=train_test_split(Xc,yc,test_size=0.4,random_state=42)
Xc_val,Xc_test,yc_val,yc_test=train_test_split(Xc_tmp,y_tmp,test_size=0.5,random_state=42)
#实例化模型
model=LGBMClassifier(
    n_estimators=300,
    random_state=42,
    learning_rate=0.05,
    min_child_samples=20,#此处默认改会20，因为上例中是为了探索不同叶子数的影响，这里只是为了分类的任务
    verbose=-1,
    num_leaves=31,
)
#训练模型
model.fit(
    Xc_train,y_train,
    eval_set=[(Xc_val,yc_val)],#用验证集监控
    eval_metric="auc",#分类常用auc
    callbacks=[lgb.early_stopping(50,verbose=False)]
)

proba=model.predict_proba(Xc_test)[:,1]#良性肿瘤的概率
#print(proba)
label=model.predict(Xc_test)#直接给出类标签，0=恶行，1=良性
#print(label)
print("良性肿瘤AUC:", roc_auc_score(yc_test, proba))
print("恶行肿瘤AUC:", roc_auc_score(yc_test, model.predict_proba(Xc_test)[:,0]))
print("Accuracy:", accuracy_score(yc_test, label))
