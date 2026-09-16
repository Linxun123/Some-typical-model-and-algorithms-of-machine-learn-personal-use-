import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 400


# ==================== 1. 读数据（与 DNN 一致） ====================
data_path = r"C:\Users\neko\PycharmProjects\PythonProject\dataset\shandong_pmos_hourly.xlsx"   # ← 改成你的路径
df = pd.read_excel(data_path)

df.columns = [c.lower().strip() for c in df.columns]
print("列名:", df.columns.tolist())

price_col = None
for col in ["price", "clearing price (cny/mwh)", "clearing_price"]:
    if col in df.columns:
        price_col = col
        break
if price_col is None:
    price_col = df.columns[1]

time_col = None
for col in ["date", "datetime", "time", "timestamp"]:
    if col in df.columns:
        time_col = col
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

print(f"价格列: {price_col}, 长度: {len(df)}")

# ==================== 2. 特征（与加强版 DNN 完全一致） ====================
for lag in [1, 2, 4, 96, 192, 288, 672]:
    df[f"lag_{lag}"] = df[price_col].shift(lag)

s = df[price_col].shift(1)
df["roll_mean_96"] = s.rolling(96, min_periods=24).mean()
df["roll_std_96"]  = s.rolling(96, min_periods=24).std()
df["roll_min_96"]  = s.rolling(96, min_periods=24).min()
df["roll_max_96"]  = s.rolling(96, min_periods=24).max()
df["diff_vs_roll"] = df["lag_1"] - df["roll_mean_96"]
df["is_neg_lag1"]  = (df["lag_1"] < 0).astype(np.float32)

if time_col:
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
else:
    df["hour"] = (df.index % 96) // 4
    df["dayofweek"] = (df.index // 96) % 7
    df["month"] = 1

df["period"] = pd.cut(df["hour"], bins=[-1, 7, 16, 24], labels=[0, 1, 2]).astype(int)
df = df.dropna().reset_index(drop=True)

feature_cols = (
    [f"lag_{lag}" for lag in [1, 2, 4, 96, 192, 288, 672]]
    + ["roll_mean_96", "roll_std_96", "roll_min_96", "roll_max_96", "diff_vs_roll", "is_neg_lag1"]
    + ["hour", "dayofweek", "month", "period"]
)

X = df[feature_cols]
y = df[price_col].values

# ==================== 3. 按时间划分（80% 训练，后 20% 测试） ====================
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 训练内部再按时间切 10% 做验证（早停用）
val_idx = int(len(X_train) * 0.9)
X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

print(f"训练: {len(X_tr)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

# ==================== 4. LightGBM 训练 ====================
params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="mae",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)

# ==================== 5. 预测与评估 ====================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n【LightGBM】整体 MAE = {mae:.2f}")

for name, mask in [
    ("负电价 (<0)", y_test < 0),
    ("低价 (0-200)", (y_test >= 0) & (y_test < 200)),
    ("中价 (200-500)", (y_test >= 200) & (y_test < 500)),
    ("尖峰 (>=500)", y_test >= 500),
]:
    if mask.any():
        m = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"{name} MAE = {m:.2f}  (n={mask.sum()})")
    else:
        print(f"{name}: 无样本")

# ==================== 6. 特征重要性 ====================
imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n特征重要性 Top10:")
print(imp.head(10).to_string(index=False))

# ==================== 7. 画图 ====================
plt.figure(figsize=(14, 5))
plt.plot(y_test[:96*7], label="真实", alpha=0.85)
plt.plot(y_pred[:96*7], label="LightGBM", alpha=0.85)
plt.title(f"LightGBM 测试集前7天 (MAE={mae:.2f})")
plt.xlabel("时间点"); plt.ylabel("价格")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.barh(imp.head(12)["feature"][::-1], imp.head(12)["importance"][::-1])
plt.title("LightGBM 特征重要性 Top12")
plt.tight_layout(); plt.show()