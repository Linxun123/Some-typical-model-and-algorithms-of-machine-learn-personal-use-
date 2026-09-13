# ====================== 分类决策树完整示例 ======================
from sklearn.datasets import load_iris#导入自带的鸢尾花数据集
from sklearn.datasets import load_wine#导入自带的葡萄酒数据集
from sklearn.model_selection import train_test_split#划分数据集
from sklearn.tree import DecisionTreeClassifier, plot_tree#plot_tree决策树可视化
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
#sepal:萼片,petal:花瓣
# 1. 加载数据
#iris = load_iris()#鸢尾花数据集
wine=load_wine()
#X = iris.data
X=wine.data
#y = iris.target
y=wine.target
#feature_names = iris.feature_names
#target_names = iris.target_names
feature_names=wine.feature_names
target_names=wine.target_names

# 查看数据基本情况
print("特征名称:", feature_names)
print("类别名称:", target_names)
print("数据形状:", X.shape)
print("\n前5行数据:")
print(pd.DataFrame(X, columns=feature_names).head())#把特征数据X转成列名为columns的表格，并显示前五行
#.head()默认显示前五行，可以通过.head(10)显示前10行

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
#train_test_split(将数据集随机划分为训练集和测试集)
#test_size=0.3指测试集占比为30%；random_state=42随机种子，保证每次划分结果一样
#stratify=y 分类任务强烈加上，按照标签分层抽样，按照y=iris.target分类
# 3. 创建并训练模型
clf = DecisionTreeClassifier(#初始化一个决策树
    criterion="gini",          # 默认使用基尼指数而非熵，因为基尼指数的计算只有加法和乘法，熵需要计算对数，计算成本更低，且而且形状排序几乎一样，基尼峰值为0.5，而熵为1
    max_depth=4,               # 限制深度，防止过拟合
    min_samples_split=5,#防过拟合参数，如果某个节点的样本数小于该数值则该节点变为叶子节点，不再往下分
    min_samples_leaf=3,#叶子节点中必须包含的最小样本数，如果分裂后子节点样本数少于该值则不再往下分
    random_state=42#随机种子，保证每次划分结果一致
)
clf.fit(X_train, y_train)#clf.fit(比如传入俩个参数，特征和标签，因此是监督学习)

# 4. 预测
y_pred = clf.predict(X_test)#用训练好的模型加上测试集进行预测

# 5. 评估
print("\n===== 模型评估 =====")
print("准确率:", accuracy_score(y_test, y_pred))#用来计算分类准确率的函数，需要传入测试标签和预测到的标签；分类任务时才使用，而回归任务用MSE
print("\n分类报告:")
#召回率指属于该类别的样本中，有多少被预测出来的比例，support为该类别在测试集中的真实样本数
print(classification_report(y_test, y_pred, target_names=target_names))#一次性输出每个类别的精确率，召回率，F1分数以及整体的宏平均和加权平均
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 6. 特征重要性
print("\n特征重要性:")#查看每个特征对模型的贡献程度，数值越大，标明该特征在构建决策树时更重要
for name, importance in zip(feature_names, clf.feature_importances_):#zip(列表1，列表2)会返回一个配对后的迭代器
    print(f"{name}: {importance:.4f}")#取出每一对的值

#zip()函数，把多个列表按位置配对

# 7. 可视化决策树
plt.figure(figsize=(16, 10))
plot_tree(clf,#模型名
          feature_names=feature_names,#特征名
          class_names=target_names,#类别名
          filled=True,#填充颜色
          rounded=True,#圆角
          fontsize=10)
          #max_depth=5,显示五层)
plt.title("Decision Tree Classifier (Iris)", fontsize=14)
plt.show()