import argparse
import numpy as np

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Compare two .npy files.')
parser.add_argument('file_float', type=str, help='Path to the first .npy file')
parser.add_argument('file_int8', type=str, help='Path to the second .npy file')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for data2')

# 解析命令行参数
args = parser.parse_args()

# 从命令行参数获取文件名和系数
file1 = args.file_float
file2 = args.file_int8
scale = args.scale

# 加载数据
data1 = np.load(file1)
data2 = np.load(file2) / scale  # 将data2乘以系数

# print(data1)
# print(data2)

# 计算差异
diff = data1 - data2

# 计算均方误差（MSE）
mse = np.mean(np.square(diff))

# 打印均方误差
print('MSE:', mse)
