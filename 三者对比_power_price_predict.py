# ============ 山东 PMOS 电价预测：决策树 / 随机森林 / LightGBM 完整对比 ============
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = r"C:\Users\neko\Documents\New project\electricity-price-models\electricity_price_forecast3.0\data\shandong_pmos_hourly.xlsx"
TARGET = "日前电价"          # 换成 "实时电价" 就是另一个预测任务
EXOG = ["竞价空间预测值", "风电总加预测值", "光伏总加预测值", "直调负荷预测值",
        "核电总加预测值", "联络线受电负荷预测值"]   # ★ 只用「预测值」，用「实际值」等于偷看答案

# ================= 1. 读数据 =================
df = pd.read_excel(DATA_PATH)
df["时刻"] = pd.to_datetime(df["时刻"])
df = df.sort_values("时刻").set_index("时刻")
print("数据形状:", df.shape, " 时间范围:", df.index.min(), "~", df.index.max())

# ================= 2. 造特征（三个模型共用，保证对比公平）=================
X = df[EXOG].copy()
X["净负荷"] = df["直调负荷预测值"] - df["新能源总加预测值"]     # 电力市场最关键的供需指标
X["小时"] = df.index.hour
X["星期"] = df.index.dayofweek
X["月份"] = df.index.month
for lag in [1, 2, 3, 24, 48, 168]:                            # 历史电价：上1小时/前天/昨天/上周
    X[f"lag{lag}"] = df[TARGET].shift(lag)
X["近24h均值"] = df[TARGET].shift(1).rolling(24).mean()         # shift(1) 保证不偷看当前时刻
X[TARGET] = df[TARGET]
X = X.dropna()                                                # 前 168 小时没有 lag，去掉
print("特征数:", len(X.columns) - 1, " 样本数:", len(X))

# ================= 3. 按时间切分（时间序列绝不能随机切）=================
train = X.loc[:"2025-06-30"]
val = X.loc["2025-07-01":"2025-12-31"]
test = X.loc["2026-01-01":]
cols = [c for c in X.columns if c != TARGET]
print(f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")

# ================= 4. 训练三个模型 =================
models = {
    "决策树": DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42),
    "随机森林": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1),
}

preds = {}
for name, m in models.items():
    m.fit(train[cols], train[TARGET])              # 这两个模型没有早停，直接训练
    preds[name] = m.predict(test[cols])            # 存下测试集预测，后面画图直接用
    print(f"{name:<6} MAE={mean_absolute_error(test[TARGET], preds[name]):7.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(test[TARGET], preds[name])):7.2f}")

# LightGBM 需要验证集做早停，单独训练
lgbm = LGBMRegressor(n_estimators=5000, learning_rate=0.03, num_leaves=63,
                     min_child_samples=50, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=1, verbose=-1, random_state=42)
lgbm.fit(train[cols], train[TARGET],
         eval_set=[(val[cols], val[TARGET])], eval_metric="mae",
         callbacks=[lgb.early_stopping(100, verbose=False)])
models["LightGBM"] = lgbm
preds["LightGBM"] = lgbm.predict(test[cols])
print(f"{'LightGBM':<6} MAE={mean_absolute_error(test[TARGET], preds['LightGBM']):7.2f}  "
      f"RMSE={np.sqrt(mean_squared_error(test[TARGET], preds['LightGBM'])):7.2f}")
print(f"{'基线':<6} MAE={mean_absolute_error(test[TARGET], test['lag24']):7.2f}  "
      f"（基线 = 拿昨天的同一时刻当预测）")
print("LightGBM 最佳轮数:", lgbm.best_iteration_)

# ================= 5. 画图：三条曲线 vs 实际 =================
N = 168                                        # 最后 7 天
idx = test.index[-N:]
plt.figure(figsize=(14, 4.5))
plt.plot(idx, test[TARGET].iloc[-N:], color="black", lw=1.6, label="实际")
plt.plot(idx, preds["决策树"][-N:], lw=1.2, ls="--", color="tab:red", label="决策树")
plt.plot(idx, preds["随机森林"][-N:], lw=1.2, color="tab:green", label="随机森林")
plt.plot(idx, preds["LightGBM"][-N:], lw=1.2, color="tab:blue", label="LightGBM")
plt.title("测试集最后 7 天：实际 vs 三种模型")
plt.ylabel("电价（元/MWh）")
plt.legend()
plt.tight_layout()
plt.savefig("电价预测_三模型_7天.png", dpi=150)
plt.show()

N2 = 72                                        # 最后 3 天，放大看细节
idx2 = test.index[-N2:]
plt.figure(figsize=(14, 4.5))
plt.plot(idx2, test[TARGET].iloc[-N2:], color="black", lw=1.8, label="实际")
plt.plot(idx2, preds["决策树"][-N2:], lw=1.6, ls="--", color="tab:red", label="决策树")
plt.plot(idx2, preds["随机森林"][-N2:], lw=1.4, color="tab:green", label="随机森林")
plt.plot(idx2, preds["LightGBM"][-N2:], lw=1.4, color="tab:blue", label="LightGBM")
plt.title("测试集最后 3 天（放大版）")
plt.ylabel("电价（元/MWh）")
plt.legend()
plt.tight_layout()
plt.savefig("电价预测_三模型_3天.png", dpi=150)
plt.show()

# ================= 6. 递归预测未来 24 小时（三个模型共用一套逻辑）=================
def forecast_24h(model):
    """从数据末尾往后推 24 小时；每步的预测值会喂给下一步当 lag 特征"""
    hist = df[TARGET].copy()                  # 已发生的电价历史
    out = {}
    for ts in pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h"):
        # ⚠️ 外生变量的未来值这里用「昨天同一时刻」占位
        #    真实使用时请替换成官方发布的次日预测值
        e = df.loc[ts - pd.Timedelta(hours=24), EXOG + ["新能源总加预测值"]]
        row = {c: e[c] for c in EXOG}
        row["净负荷"] = e["直调负荷预测值"] - e["新能源总加预测值"]
        row["小时"], row["星期"], row["月份"] = ts.hour, ts.dayofweek, ts.month
        for lag in [1, 2, 3, 24, 48, 168]:
            row[f"lag{lag}"] = hist.iloc[-lag]
        row["近24h均值"] = hist.iloc[-24:].mean()

        p = float(model.predict(pd.DataFrame([row])[cols])[0])
        out[ts] = hist.loc[ts] = p            # 预测值写入历史，供下一步使用
    return pd.Series(out)


result = pd.DataFrame({name: forecast_24h(m) for name, m in models.items()})
print("\n未来 24 小时预测：")
print(result.round(1).to_string())