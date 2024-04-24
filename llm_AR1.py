import numpy as np
from scipy.stats import norm
import time


def ar1_generator(T, mu = 0.5, phi = 0.975, x_sigma_square = 0.02, y_sigma_square = 2, seed = 123):
    """
    AR1的生成器
    Args:
        N: particle numbers
        T: 期数
        mu, phi, x_sigma_square, y_sigma_square: 模型参数
    """
    np.random.seed(seed)
    eta = np.sqrt(x_sigma_square) * np.random.randn(T)
    epsilon = np.sqrt(y_sigma_square) * np.random.randn(T)
    states = np.zeros((T, ))
    for t in range(T):
        if t == 0:
            states[t] = np.random.randn() * np.sqrt(x_sigma_square / (1 - phi ** 2))
        else:
            states[t] = mu + phi * (states[t - 1] - mu) + eta[t]
    observations = states + epsilon
    return observations

def stratif_sampling_selection(weights, particles, seed = 567):
    """
    分层+改进的重抽样
    Args:
        weights: 标准化后的权重
        particles: 粒子
    Returns:
        resampling后的新一期particle
    """

    N = len(weights)
    pi = (np.append([0],weights) + np.append(weights, [0]))/2 ## 计算概率lambda

    # =============================================================================== #
    # We also keep the uniforms associated with the stratified bootstrap method fixed #
    # =============================================================================== #
    np.random.seed(seed)
    u = (np.random.uniform(size = 1) + np.arange(N))/ N  ## 生成均匀分布随机变量
    quantitle = np.cumsum(pi)  ## 概率lambda累计求和得到每个区间的端点
    r = np.searchsorted(a = quantitle, v = u, side = 'left')  ## 判断生成的均匀分布随机变量落入哪一个区间
    u_new = (u - (quantitle[r] - pi[r])) / pi[r]
    # new_quantitle = np.insert(quantitle, obj = 0, values = 0) ## 扩充区间端点，把0给包括进来
    # u_new = (u - new_quantitle[r]) / pi[r]  ## 生成新的均匀分布随机变量
    u_new[r == N] = 0  ## 当均匀分布u落入[quantitile[-1], 1]区间时，particle等于最大的particle，这里先赋值为0，后面能统一计算
    new_particles = np.append(particles, particles[0]) ## 当落入[0, quantitiles[0]]区间时，particle等于最小的particle，这里先在原先的particle的最后插入该值，后面能统一计算
    ##注意当r = N的时候,u_new = 0，因此 new_particle就等于原先最大的particle; r = 0 时 (new_particles[0] - new_particles[-1]) = 0，因此new_particle等于原先最小的particle
    new_particles = (new_particles[r] - new_particles[r - 1]) * u_new + new_particles[r - 1] 

    return new_particles

    # ======================================== #
    #               论文中的伪代码               #
    # ======================================== #
    # pi = np.zeros(N + 1)
    # pi[0] = weights[0]/2
    # pi[N] = weights[-1]/2
    # for i in range(1, N):
    #     pi[i] = (weights[i]+weights[i-1])/2
    # pi = np.cumsum(pi)
    # np.random.uniform(low = np.append([0], pi[: -1]), high = pi, size = N)

    # r = np.zeros(N)
    # u_new = np.zeros(N)
    # s = 0
    # j = 1

    # u0 = np.random.uniform(size = 1)
    # u = [(u0 + i) / N for i in range(N)]
    # for i in range(N + 1):
    #     s = s + pi[i]
    #     while (j <= N and u[j-1] <= s):
    #         r[j-1]=i
    #         u_new[j-1] = (u[j-1]-(s-pi[i]))/pi[i]
    #         j += 1
    # r = r.astype(int)

    # x_new = np.zeros(N)
    # for k in range(N):
    #     if r[k] == 0:
    #         x_new[k] = particles[0]
    #     elif r[k] == N:
    #         x_new[k] = particles[-1]
    #     else:
    #         x_new[k] = (particles[r[k]] - particles[r[k]-1]) * u_new[k] + particles[r[k]-1]

    


def CSIR_particle_filters(observations, particles, mu = 0.5, phi = 0.975, x_sigma_square = 0.02, y_sigma_square = 2, transition_seed = 123, stratified_seed = 456):
    """
    《Particle filters for continuous likelihood evaluation and maximisation》
    Args:
        observations: 观测数据
        particles: 粒子
        mu, phi, x_sigma_quare, y_sigma_quare: ar1模型参数
        seed: 随机种子
    Returns:
        likelihood: 极大似然估计
    """
    T = len(observations)
    N = len(particles)
    y_sigma = np.sqrt(y_sigma_square) 


    # ====================================================================================================== #
    # We fix the random numbers (or equivalently the random number seeds) used in step 1 of Algorithm: SIR.  #
    # This fixes the innovations we propagate through the state equation.                                    #
    # ====================================================================================================== #
    np.random.seed(transition_seed)
    eta = np.sqrt(x_sigma_square) * np.random.randn(N)
    likelihoods = np.zeros(T) ## 保存最大似然
    sampling_selection_time = 0  ## resampling time
    total_sort_time = 0 ## sorting time
    likelihoods[0] = norm.pdf(observations[0], loc = 0, scale = np.sqrt(x_sigma_square / (1 - phi ** 2) + y_sigma_square)) ## 第1期
    for t in range(1, T):
        particles = mu + phi * (particles - mu) + eta  ## transition

        sort_start = time.time()
        particles = np.sort(particles)  ## 排序
        sort_end = time.time()
        total_sort_time += (sort_end - sort_start)

        weights = norm.pdf(observations[t], loc = particles, scale = y_sigma)  ## 计算权重
        likelihoods[t] = np.mean(weights)  ## 计算最大似然
        normalised_weights = weights / sum(weights)  ## 标准化权重
      
        if t == T - 1:
            break
        
        start_time = time.time()
        particles = stratif_sampling_selection(normalised_weights, particles, seed = stratified_seed)
        end_time = time.time()
        sampling_selection_time += (end_time - start_time)
        
    # print("sorting time:{}s".format(round(total_sort_time, 2)))
    # print("resampling time:{}s".format(round(sampling_selection_time, 2)))
    return likelihoods