#对数据的处理和分析
import pandas as pd
print("Series为创建一维数组==================================")
# 从列表创建Series数组 不指定标签 标签默认为角标
s = pd.Series([1,True,3.14,"Hello"])#标签为左边所显示的东西，默认从0开始
print(s)

# 指定标签
s = pd.Series([1,2,3,4], index=['a','b','c','d'])#标签为index
print(s)

# 从字典创建Series数组
dic = {'a':10, 'b':20, 'c':30, 'd':40}
s = pd.Series(dic)
print(s)

print("DataFrame为创建二维数组=========================================")
dict={'name':['alice','dady','honda'],'age':[21,22,23]}
s=pd.DataFrame(dict)#通过字典
print(s)

data=[['alice',21],['honda',22],['dady',23]]
cols=['name','age']
s=pd.DataFrame(data,columns=cols)#通过二维数组创建，指定标签时需要俩个参数，数据和标签名(一维数组)
print(s)
s=pd.DataFrame(data)#未指定标签
print(s)

print("读取和写入数据=========================================================")

#读取excel文件时需要安装三方库 pip install openpyxl
#读取一个已经存在的csv文件
datapath=r"C:\Users\neko\PycharmProjects\PythonProject\.venv\tianchi.csv"
df=pd.read_csv(datapath)
print(df)#输出文件内容
#写入csv文件
data=[['alice',21],['honda',22],['dady',23]]
cols=['name','age']
df=pd.DataFrame(data,columns=cols)
df.to_csv("test.csv",index=False)

print("数据查看与信息获取=====================================================")

df=pd.read_csv("test.csv")
df.info()#获取df的基本信息

rows,cols=df.shape#df.shape是pandas返回的一个元组，把df的行列数分别赋值给rows和cols
print(rows,cols)#3行2列
#.....

print("数据处理与清洗========================================================")

#处理缺失值
df=pd.read_csv("tianchi.csv")
print(df)
print("缺失值处理后：")
df1=df.dropna()#删除包含缺失值行
print(df1)
print("填充缺失值：")
df2=df.fillna({'Price':df['Price'].mean(),'Exogenous 1':df['Exogenous 1'].mean()})#处理结果
print(df2)

#处理重复值
df3=df.drop_duplicates()
print(df3)