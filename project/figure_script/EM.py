import numpy as np
import matplotlib.pyplot as plt

def get_respons(data, mu, sigma, weights):
    # 计算每个数据点属于每个高斯分布的概率（责任值）
    respons = np.zeros((len(data), len(mu)))
    for k in range(len(mu)):
        for i in range(len(data)):
            # 计算每个数据点在第k个高斯分布下的概率
            respons[i, k] = weights[k] * gaussian(data[i], mu[k], sigma[k]) 
    # 归一化概率，得到每个数据点对应于每个高斯分布的责任值
    return respons / np.sum(respons, axis=1, keepdims=True)


def update_param(data, responsibilities):
    # 使用责任值重新估计高斯分布的均值、方差和数据点的混合权重
    num_clusters = responsibilities.shape[1]
    num_points = responsibilities.shape[0]
    new_mu =  np.zeros(num_clusters)
    new_sigma = np.zeros(num_clusters)
    new_weights = np.zeros(num_clusters)

    for k in range(num_clusters):
        total_respons = np.sum(responsibilities[:, k])
        # 更新均值
        new_mu[k] = np.dot(responsibilities[:, k], data) / total_respons
        # 更新标准差
        new_sigma[k] = np.sqrt(np.dot(responsibilities[:, k], (data - new_mu[k]) ** 2) / total_respons)
        # 更新混合权重
        new_weights[k] = total_respons / num_points

    return new_mu, new_sigma, new_weights

# 计算高斯分布的概率密度函数
def gaussian(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def EM_algorithm(data, num_clusters, max_iterations):
    # 初始化参数
    mu = np.random.rand(num_clusters)
    sigma = np.ones(num_clusters)
    weights = np.ones(num_clusters) / num_clusters

    for _ in range(max_iterations):
        # E步：计算责任值
        responsibilities = get_respons(data, mu, sigma, weights)

        # M步：利用责任值更新参数
        mu, sigma, weights = update_param(data, responsibilities)

    return mu, sigma, weights

# 运行EM算法
def run_EM(data, num_clusters=5, max_iterations=300):
    mu, sigma, weights = EM_algorithm(data, num_clusters, max_iterations)

    # 打印并写入文件
    with open('em_output.txt', 'a') as file:
        file.write(f"Estimated means: {mu}\n")
        file.write(f"Estimated standard deviations: {sigma}\n")
        file.write(f"Estimated weights: {weights}\n")

    # 命令行同步输出
    print("Estimated means:", mu)
    print("Estimated standard deviations:", sigma)
    print("Estimated weights:", weights)
    return mu, sigma, weights

# 绘制每个簇的高斯分布曲线和数据分布直方图
def plot_gauss_hist(data, mu, sigma, num_clusters=5):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Garamond'
    plt.hist(data, bins=30, density=True, alpha=0.5, color='silver', edgecolor='black', label='Data Distribution')

    # 指定颜色列表
    colors = ['lightgreen', 'turquoise', 'cadetblue', 'steelblue', 'midnightblue']

    x = np.linspace(np.min(data), np.max(data), 500)
    for k in range(num_clusters):
        y = gaussian(x, mu[k], sigma[k])
        plt.plot(x, y, color=colors[k], label=f'Cluster {k+1}')

    plt.title('EM Algorithm for Gaussian Mixture Model', fontsize=18)
    plt.xlabel('Randomly Generated Data Points', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.legend()
    plt.savefig('EM_Gaussian_Mixture_Model.pdf', format='pdf')
    plt.show()

# 生成测试数据，从我的学号最后两位03开始
def get_data():
    data = np.concatenate([np.random.normal(3, 1, 200), np.random.normal(4, 1, 200), np.random.normal(5, 1, 200), np.random.normal(6, 1, 200), np.random.normal(7, 1, 200)])
    return data

if __name__ == "__main__":
    # 示例使用
    data = get_data()
    mu, sigma, weights = run_EM(data)
    plot_gauss_hist(data, mu, sigma)