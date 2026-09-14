import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report,roc_auc_score

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=400
#导入数据库
iris = load_iris()

feature_names=iris.feature_names
data = iris.data
target = iris.target
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.2,random_state=7,stratify=target)
#实例化随机森林分类模型
rf = RandomForestClassifier(
    criterion='entropy',
    random_state=0,
    n_estimators=64,#64颗决策树
    max_depth=5,
    min_samples_split=10,#少于10不分裂
)
#训练模型
dtree=rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
#计算指标
rf_roc_auc=roc_auc_score(y_test,dtree.predict_proba(X_test),multi_class='ovr')
print("随机森林 AUC= ",rf_roc_auc)
print(classification_report(y_test,y_pred,target_names=iris.target_names))

for name,importance in zip(iris.feature_names,dtree.feature_importances_):
    print(name,importance)
