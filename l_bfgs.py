#coding:UTF-8
 
import numpy as np
from typing import *
from collections import deque

def l_bfgs(fun: Callable[[np.ndarray], float], gfun: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, max_iter: int = 100) -> List[float]:
    """
    L-BFGS算法
    :param fun: 目标函数
    :param gfun: 目标函数的梯度
    :param x0: 初始点
    :param max_iter: 最大迭代次数
    :return: 迭代过程中的目标函数值
    """
    # 超参
    m = 10000 # 存储向量序列的最大长度
    rho = 0.45 # 线搜索参数
    sigma = 0.4 # 线搜索参数

    # sks向量序列存储\delta x，yks向量序列存储\delta \nabla f, rhok存储\rho, 
    k = 0
    sks = deque(maxlen=m)
    yks = deque(maxlen=m)
    rhoks = deque(maxlen=m)

    result_x0 = []
    result_func = []
    while (k < max_iter):
        dk = gfun(x0)
        delta = max(0, k - m)
        L = min(k, m)
        vk1s = np.zeros(L)
        vk2s = np.zeros(L)
        for i in range(L - 1, -1, -1): # 后向循环
            j = i + delta
            vk1s[i] = rhoks[j - delta] * sks[j - delta].T @ dk
            dk -= vk1s[i] * yks[j - delta]
        for i in range(L): # 前向循环
            j = i + delta
            vk2s[j] = rhoks[j - delta] * yks[j - delta].T @ dk
            dk += (vk1s[i] - vk2s[i]) * sks[j - delta]
        dk *= -1

        gk = gfun(x0)
        alpha = 0
        alphak = 0
        while (alpha < 2000): # 非精确线搜索
            newf = fun(x0 + rho ** alpha * dk)
            oldf = fun(x0)
            if (newf < oldf + sigma * (rho ** alpha) * (gk.T @ dk)[0, 0]):
                alphak = alpha
                break
            alpha = alpha + 1
        
        # 参数更新
        x = x0 + rho ** alphak * dk
        sk = x - x0
        yk = gfun(x) - gk
        x0 = x
        sks.append(sk)
        yks.append(yk)
        rhoks.append(1 / yk.T @ sk)

        # 停止条件
        if (yk.T @ sk < 1e-10 or (yk + gk).T @ (yk + gk) < 1e-10):
            break
        
        k = k + 1
        result_x0.append(x0)
        result_func.append(fun(x0))
    
    return result_x0, result_func