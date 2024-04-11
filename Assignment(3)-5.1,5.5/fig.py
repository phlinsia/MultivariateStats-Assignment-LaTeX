import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from matplotlib import rcParams

# Set the global font to be Garamond, size 10 (or any other size you want)
rcParams['font.family'] = 'Garamond'
rcParams['font.size'] = 10

alpha = 0.05
n_values = np.arange(2, 121, 1)
p_values = np.arange(1, 61, 1)

# Create a meshgrid for n and p values
n, p = np.meshgrid(n_values, p_values)

# Calculate the function values
z = ((n-1)*p)/(n-p)*f.ppf(1-alpha, p, n-p)

# Create a scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(n, p, z)

ax.set_xlabel('n')
ax.set_ylabel('p')
ax.set_zlabel('Value')

sns.set_style("darkgrid")

plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from matplotlib import rcParams
# from matplotlib.lines import Line2D

# # 设置全局字体为SimSun
# rcParams['font.family'] = 'SimSun'

# # 定义给定的变量
# x_bar = np.array([0.564, 0.603])
# mu = np.array([0.55, 0.60])
# S = np.array([[0.0144, 0.0117], [0.0117, 0.0146]])
# S_inv = np.array([[203.018, -163.391], [-163.391, 200.228]])

# # 计算S的特征值和特征向量
# eigvals, eigvecs = np.linalg.eig(S)

# # 计算椭圆的旋转角度
# angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

# # 计算椭圆的宽度和高度
# width, height = 2 * np.sqrt(eigvals)

# # 创建新的图形
# fig, ax = plt.subplots()

# # 绘制椭圆
# ellipse = Ellipse(xy=x_bar, width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2, zorder=4)
# ax.add_patch(ellipse)

# # 绘制椭圆的中心
# ax.scatter(*x_bar, color='red')

# # 绘制假设的均值向量
# ax.scatter(*mu, color='blue', label='假设的均值向量 ({}, {})'.format(*mu))

# # 设置图形的限制
# ax.set_xlim(0.4, 0.7)
# ax.set_ylim(0.4, 0.8)

# # 设置图形的标签
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')

# # 添加图例
# ax.legend()

# plt.show()