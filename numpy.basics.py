#对数组或矩阵的处理
import numpy as np

lst=[1,2,3,4,5]
print(lst)

arr=np.array([1,2,3,4,5])
print(arr)

arr=np.array([[1,2,3],[4,5,6],[7,8,9]])#二维3x3矩阵
print(arr)

arr=np.zeros((3,4))#创建3x4大小的全零矩阵
print(arr)

arr=np.ones((3,4))#全1矩阵
print(arr)

arr=np.full((3,4),5)#全5矩阵
print(arr)

arr=np.arange(0,10,2)#从零开始d=2的等差数列，一直到10
print(arr)

arr=np.linspace(0,10,5)#从0-10范围内等间隔取五个数
print(arr)

#arr=np.random.rand(2,10).reshape((4,5))#随机矩阵，0-10的范围随机20个数，形状为4x5
#print(arr)
np.savetxt('test.txt',arr)#保持数据为text.txt
arr=np.loadtxt('test.txt')#读取矩阵
print(arr)
print("数组属性==================================")

print(arr.shape)#数组的形状
print(arr.size)#数组的大小(多少个元素)
print(arr.dtype)#数组的类型(int/float...)

print("数组的切片==================================")

arr=np.array([1,2,3,4,5])
print(arr[0])#从左往右第零个元素
print(arr[-1])#从右往左第零个元素
print(arr[1:3])#下标从1-2的元素
#二维数组
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr[0][0])#i=0,j=0的元素,下表从零开始的矩阵
print(arr[2][1])#i=3,j=2的元素
print(arr[-1][-1])
print(arr[1,:])#第二行所有元素
print(arr[:,1])#第二列所有元素

print("数组的运算+++++++++++++++++++++++++++++++++++")

arr1=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2=np.array([[1,2,3],[4,5,6],[7,8,9]])
#元素级操作
print("加法\n")
print(arr1+arr2)
print("减法\n")
print(arr1-arr2)
print("乘法\n")
print(arr1*arr2)#对位直接相乘，并不是矩阵的乘法
print("除法\n")
print(arr1/arr2)
print("数组整体加法\n")
print(arr1+3)
print("数组整体乘法\n")
print(arr1*3)
#矩阵乘法
arr3=np.array([[1,2,3],[4,5,6]])#2x3的矩阵
arr4=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])#3x4的矩阵
arr5=np.dot(arr3,arr4)#矩阵的乘法，得到2x4的矩阵
print(arr5)

print("数组的数学和统计函数==================================")

arr=np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])#3x4的矩阵
print("计算元素和")
print(np.sum(arr))
print("计算平均值")
print(np.mean(arr))
print("计算最大/最小值")
print(np.max(arr))
print(np.min(arr))
print("计算标准差")
print(np.std(arr))
print("计算第二行的平均值")
print(np.mean(arr[1,:]))
print("计算第二列的平均值")
print(np.mean(arr[:,1]))

print("数组的变形与拼接=================================")

arr=np.array([1,2,3,4,5,6,7,8])
print(arr)
arr1=arr.reshape((4,2))#将数组arr变换成4x2的矩阵
print(arr1)
#arr2=arr.reshape((2,2,2))#2x2x2
#print(arr2)

#拼接
arr1=np.array([[1,2],[3,4]])
arr2=np.array([[5,6],[7,8]])
arr3=np.hstack((arr1,arr2))#水平拼接
print(arr3)
arr4=np.vstack((arr1,arr2))#竖直拼接
print(arr4)