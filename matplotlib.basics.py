#画图
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error#均方误差

plt.rcParams['font.sans-serif']=['SimHei']#字体为黑体
plt.rcParams['axes.unicode_minus']=False#解决'-'号显示为方块的问题
plt.rcParams['figure.dpi']=400#图片清晰度
plt.rcParams['font.size']#字体大小
plt.rcParams['figure.constrained_layout.use'] = True#自动调整格式
#plt.rcParams['text.usetexs']=True#打印数学公式
#简单例子
#data=[1,2,3,4,5]
#plt.title("折线图")
#plt.xlabel("人数")
#plt.ylabel("年龄均值")
#stock1=[1,2,3,4,5]
#stock2=[2,4,6,8,10]
#plt.plot(data,stock1,"r.-",label="折线1")#后面俩个为设置折线的样式
#plt.plot(data,stock2,"go--",label="折线2")#r.-代表红色，小圆点，直线/go--代表绿色，大圆点，虚线/k+-.代表黑色，加号，点状线etc
#依次为(颜色，点，线)
#plt.legend()#为图标添加图例，即标注每条线/图形代表着什么的小方框
#plt.grid()#添加辅助线网格
#plt.xlim(2,3)#设置x轴的显示范围
#plt.ylim(2,8)#设置y轴的显示范围
#plt.show()

print("========================================")

fd=pd.read_csv("tianchi.csv")
fd.info()
fd.describe()
time_col=fd.columns[0]
price_col=fd.columns[1]
print("时间列为：",time_col)
print("价格列为：",price_col)

plt.figure(figsize=(14,5))
fd['pred_naive']=fd[price_col].shift(96)
fd_valid=fd.dropna(subset=['pred_naive']).copy()
fd['Date']=pd.to_datetime(fd['Date'])
fd = fd.sort_values('Date').reset_index(drop=True)#按Date列进行排序，然后重置索引，防止乱跳
fd=fd.dropna()

mae=mean_squared_error(fd_valid[price_col],fd_valid['pred_naive'])
print(f"均方误差为：{mae:.2f}")

plt.figure(figsize=(14,5))
plt.title("天池数据折线图")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
#plt.grid(True)
plt.plot(fd_valid[time_col].values[-96*7:],fd_valid[price_col].values[-96*7:],label='True price',alpha=0.7)
plt.plot(fd_valid[time_col].values[-96*7:],fd_valid['pred_naive'].values[-96*7:],label='Pre price',alpha=0.7)
plt.tight_layout()#自动调整子图参数，留出合适间距
plt.xticks(rotation=45)
plt.show()