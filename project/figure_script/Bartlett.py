from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

# 绘制 Q-Q 图
def plot_qq(data, p, g, df):
    plt.rcParams['font.family'] = 'Garamond'
    # Step1: 排序，计算概率
    nums = len(data)
    probability = (np.arange(1, nums + 1) - 0.5) / nums
    sorted_data = sorted(data)

    # Step2: 计算分位数对应的值
    normal_data = chi2.ppf(probability, df)

    # Step3: 画图
    plt.scatter(sorted_data, normal_data, s=3, color='midnightblue')  # 调整点的大小为10，颜色为darkblue
    plt.title(f'Q-Q Plot For X^2 Sample p={p}, g={g}', fontsize=18)

    # 添加拟合直线
    z = np.polyfit(sorted_data, normal_data, 1)
    pp = np.poly1d(z)
    plt.plot(sorted_data, pp(sorted_data), color='crimson', linewidth=0.7)  # 拟合直线颜色为crimson

    plt.savefig(f"qq_plot_p={p}_g={g}.pdf", format='pdf')
    plt.clf()

# 绘制直方图和卡方分布的概率密度函数
def plot_hist_chi(plot_data, p, g):
    # 画直方图
    plt.rcParams['font.family'] = 'Garamond'
    plt.hist(plot_data, bins=30, density=True, alpha=0.6, color='lightgray', edgecolor='black', label='Data Distribution')

    # 生成卡方分布的概率密度函数
    x = np.linspace(min(plot_data), max(plot_data), 10000)
    df = p * (g - 1)
    pdf = chi2.pdf(x, df=df)

    # 叠加卡方分布的概率密度函数
    plt.plot(x, pdf, color='darkblue', label='Chi-Square PDF')

    # 设置坐标轴标签和标题
    plt.xlabel('Value', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.title(f'Histogram and Chi-Square PDF p={p}, g={g}', fontsize=18)
    plt.legend()

    # 保存图像
    plt.savefig(f"hist_plot_p={p}_g={g}.pdf", format="pdf")

    # 清除缓存，以便于绘制新的图像
    plt.clf()

# 统计分布采样选择函数
def sample_dist(size, dist=None):
    if dist == 'N':  # 正态分布
        return np.random.normal(loc=0, scale=0.1, size=size)
    elif dist == 'U':  # 均匀分布
        return np.random.uniform(low=-1, high=1, size=size)
    elif dist == 'P':  # 泊松分布
        return np.random.poisson(lam=5, size=size)
    elif dist == 'E':  # 指数分布
        return np.random.exponential(scale=5, size=size)
    elif dist == 'T':  # 标准t分布
        return np.random.standard_t(df=3, size=size)
    elif dist == 'X2':  # 卡方分布
        return np.random.chisquare(df=7, size=size)
    else:
        raise ValueError('Unknown distribution')


# 计算 Wilks' Lambda
def calc_wilks_lambda(data, p, g, N):
    group_means = np.mean(data, axis=2)  # group_means: [g, p]
    overall_means = np.mean(data, axis=(0, 2))  # overall_means: [p]

    # 计算组均值与总体均值之间的差异
    group_mean_diff = (group_means - overall_means[None, :])[:, :, None]  # group_mean_diff: [g, p, 1]
    group_mean_diff_transpose = (group_means - overall_means[None, :])[:, None, :]  # group_mean_diff_transpose: [g, 1, p]

    # 计算组间散布矩阵 B
    B = (group_mean_diff @ group_mean_diff_transpose).sum(axis=0) * N

    # 计算观测值与对应组均值之间的差异
    within_group_diff = (data - group_means[:, :, None]).transpose(0, 2, 1)[:, :, :, None]  # within_group_diff: [g, N, p, 1]
    within_group_diff_transpose = (data - group_means[:, :, None]).transpose(0, 2, 1)[:, :, None, :]  # within_group_diff_transpose: [g, N, 1, p]

    # 组内散布矩阵 W
    W = (within_group_diff @ within_group_diff_transpose).sum(axis=(0, 1))

    # 计算 Wilks' Lambda
    lambda_val = np.linalg.det(W) / np.linalg.det(B + W)

    return -(g * N - 1 - (p + g) / 2) * np.log(lambda_val)


# 迭代函数
def perform_iter(P, G, N, Step, pdf_list):
    for p in P:
        for g in G:
            plot_data = []
            for step in range(Step):
                all_data = []

                name_list = np.random.choice(pdf_list, size=p, replace=True)
                for index in range(g):
                    data_list = [sample_dist(N, name_list[idx])[None, :] for idx in range(p)]
                    data = np.concatenate(data_list)
                    all_data.append(data[None, :])

                data = np.concatenate(all_data)  # data: [g, p, N]

                result = calc_wilks_lambda(data, p, g, N)
                plot_data.append(result)

            plot_hist_chi(plot_data, p, g)
            plot_qq(plot_data, p, g, df=p * (g - 1))

if __name__ == '__main__':
    # 超参数
    # G=[3]
    # P=[4]
    G = [3, 5, 7]
    P = [3, 10, 20, 30]
    N = 1000
    Step = 2000
    pdf_list = ['N', 'U', 'P', 'E', 'T', 'X2']  # 可选的分布

    # 调用函数进行迭代
    perform_iter(P, G, N, Step, pdf_list)

