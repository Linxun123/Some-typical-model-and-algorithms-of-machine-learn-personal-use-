import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=400
plt.rcParams['figure.constrained_layout.use'] = True

datapath = r"C:\Users\neko\PycharmProjects\PythonProject\dataset\tianchi.csv"

df =pd.read_csv(datapath)

print("当前列名：",df.columns.tolist())
# ========== 2. 简单处理时间（根据你实际列名调整） ==========
# 情况A：如果有 'Date' 列
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])#时间格式统一
    df = df.sort_values('Date').reset_index(drop=True)
    price_col = 'Price'
    time_col = 'Date'

# 情况B：如果有 'day' 和 'time' 列（天池原始常见情况）
elif 'day' in df.columns and 'time' in df.columns:
    df['datetime'] = pd.to_datetime(df['day'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    df = df.sort_values('datetime').reset_index(drop=True)
    price_col = [c for c in df.columns if 'price' in c.lower() or 'Price' in c][0]
    time_col = 'datetime'

# 情况C：其他情况，请告诉我列名
else:
    print("其他列名")
    price_col = df.columns[1]  # 临时假设第二列是价格
    time_col = df.columns[0]

print(f"使用的时间列: {time_col}")
print(f"使用的价格列: {price_col}")
print(f"数据总长度: {len(df)}")

# ========== 3. 最简单的预测方法：用“昨天同一时刻”的价格预测今天 ==========
# 一天有96个点（15分钟一个）
df['pred_naive'] = df[price_col].shift(96)# 往前推96个点 = 昨天同一时刻

# 去掉最前面没有昨天数据的部分
df_valid = df.dropna(subset=['pred_naive']).copy()#复制一份

# 计算误差
mae = mean_squared_error(df_valid[price_col], df_valid['pred_naive'])#均方误差
print(f"\n最简单基线（用昨天价格预测今天）的 MAE = {mae:.2f}")

# ========== 4. 画对比图（最后7天） ==========
plt.figure(figsize=(14, 5))
plt.plot(df_valid[time_col].values[-96*7:], df_valid[price_col].values[-96*7:], label='true price', alpha=0.8)
plt.plot(df_valid[time_col].values[-96*7:], df_valid['pred_naive'].values[-96*7:], label='pre(yesterday)', alpha=0.8)
plt.title(f"最后7天：真实 vs 简单预测 (MAE={mae:.2f})")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.grid(True)#辅助线网格
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#=================================================
#df['Date']=pd.to_datetime(df['Date'])

#print("数据形状(行数 列数)：",df.shape)
#print("\n列名:")
#print(df.columns.tolist())
#print("\n前五行:")
#print(df.head())
#print("\n数据类型:")
#print(df.dtypes)
#print("\n缺失值统计:")
#print(df.isnull().sum())

#plt.figure(figsize=(12,4))
#plt.plot(df['Date'].values[:96*7],df['Price'].values[:96*7])
#plt.title("前七天出清价格走势")
#plt.xlabel("time point(15分钟一个点)")
#plt.ylabel("price(cmy/mwh)")
#plt.grid(True)
#plt.show()