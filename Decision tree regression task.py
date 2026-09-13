# ====================== 回归决策树完整示例 ======================
from sklearn.datasets import load_diabetes#导入自带的糖尿病数据集
from sklearn.datasets import fetch_california_housing#导入加州房价数据集
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']=400

# 1. 加载数据
#diabetes = load_diabetes()#糖尿病数据集
fch=fetch_california_housing()#加州房价数据集
#X = diabetes.data
X=fch.data
#y = diabetes.target
y=fch.target
#feature_names = diabetes.feature_names#特征名
feature_names=fch.feature_names

print("特征名称:", feature_names)
print("数据形状:", X.shape)
print("目标变量范围: [{:.1f}, {:.1f}]".format(y.min(), y.max()))
print("\n前5行数据:")
print(pd.DataFrame(X, columns=feature_names).head())#默认打印前五行

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42#测试集占20%，随机种子，保证每次划分结果一致，方便复现
)

# 3. 创建并训练模型
reg = DecisionTreeRegressor(
    criterion="squared_error",  # 均方误差
    max_depth=5,
    min_samples_split=10,#若某节点样本数小于10则不再划分，直接成为叶子节点
    min_samples_leaf=5,#叶子节点中样本数最少为5
    random_state=42#随机种子，每次划分结果一致，便于复现
)
reg.fit(X_train, y_train)#.fit()开始训练函数

# 4. 预测
y_pred = reg.predict(X_test)#.prefict()开始预测函数

# 5. 评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== 模型评估 =====")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"R² 分数: {r2:.4f}")

# 6. 特征重要性
print("\n特征重要性:")
for name, importance in zip(feature_names, reg.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 7. 可视化决策树（前几层）
plt.figure(figsize=(18, 10))
plot_tree(reg,
          feature_names=feature_names,
          filled=True,
          rounded=True,
          fontsize=9,
          max_depth=3)  # 只画前3层，方便查看
plt.title("Decision Tree Regressor (Diabetes) - 前3层", fontsize=14)
plt.show()

# 8. 真实值 vs 预测值 对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)#分别为x轴的起点和终点，y轴的起点和终点，线宽lw为2
#相当于y=x，该线为基准，点在上方时预测值偏大，在下方时预测值偏小，点上时预测准确
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值")
plt.grid(True, alpha=0.3)
plt.show()