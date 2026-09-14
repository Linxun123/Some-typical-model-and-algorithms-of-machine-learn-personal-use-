import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

diabetes = load_diabetes()
X=diabetes.data
y=diabetes.target
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#多模型创建
models = {
    '决策树': DecisionTreeRegressor(criterion='squared_error', min_samples_split=10,
                                     min_samples_leaf=5, max_depth=5, random_state=42),
    '随机森林': RandomForestRegressor(criterion='squared_error', min_samples_split=10,
                                       min_samples_leaf=5, max_depth=5, random_state=42),
    '线性回归(参照)': LinearRegression(),
    '预测均值(基线)': DummyRegressor(strategy='mean'),
}

# 1：单次划分，两个模型都算全指标
print(f"{'模型':<14}{'训练R2':>9}{'测试R2':>9}{'RMSE':>9}{'MAE':>9}")
for name, m in models.items():
    m.fit(X_train, y_train)
    p_tr, p_te = m.predict(X_train), m.predict(X_test)
    print(f"{name:<14}{r2_score(y_train, p_tr):>9.4f}{r2_score(y_test, p_te):>9.4f}"
          f"{mean_squared_error(y_test, p_te) ** 0.5:>9.2f}{mean_absolute_error(y_test, p_te):>9.2f}")
#>,<分别表示右对齐和左对齐占用多少多少格

# 2：同一套 5 折交叉验证，比均值±标准差
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}

print(f"\n{'模型(5折CV)':<16}{'R2':>17}{'RMSE':>17}{'MAE':>17}{'训练R2':>9}")
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    print(f"{name:<16}{r['test_r2'].mean():>10.3f}±{r['test_r2'].std():.3f}"
          f"{-r['test_rmse'].mean():>10.2f}±{r['test_rmse'].std():.2f}"
          f"{-r['test_mae'].mean():>10.2f}±{r['test_mae'].std():.2f}"
          f"{r['train_r2'].mean():>9.3f}")

# 3：并排画预测图
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, name in zip(axes, ['决策树', '随机森林']):
    m = models[name].fit(X_train, y_train)
    pred = m.predict(X_test)
    ax.scatter(y_test, pred, s=18, alpha=0.6, label=f'{name}  R2={r2_score(y_test, pred):.3f}')
    ax.plot([y.min(), y,max()], [y.min(), y,max()], 'r--', lw=1.5, label='理想线')
    ax.set_title(f'真实值 vs 预测值（{name}）')
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.grid(True, alpha=0.3);
    ax.legend()

fig.tight_layout();
plt.show()

# 4：残差分布对比
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, name in zip(axes, ['决策树', '随机森林']):
    m = models[name].fit(X_train, y_train)
    ax.hist(y_test - m.predict(X_test), bins=20, alpha=0.75)
    ax.axvline(0, color='red', lw=1)
    ax.set_title(f'{name} 的残差分布');
    ax.set_xlabel('残差')
    ax.grid(True, alpha=0.3)

fig.tight_layout();
plt.show()