import torch
import numpy as np
from scipy.linalg import sqrtm


def toFIDMetric(act_values1: torch.Tensor, act_values2: torch.Tensor) -> float:
    act_values1_np = act_values1.cpu().numpy()
    act_values2_np = act_values2.cpu().numpy()

    # 计算均值和协方差矩阵
    print('start calculate cov1')
    mu1, sigma1 = act_values1_np.mean(axis=0), np.cov(act_values1_np, rowvar=False)
    print('start calculate cov2')
    mu2, sigma2 = act_values2_np.mean(axis=0), np.cov(act_values2_np, rowvar=False)

    # 计算均值差的平方和
    print('start calculate diff^2')
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # 计算协方差矩阵乘积的平方根
    print('start calculate sqrtm')
    covmean = sqrtm(sigma1.dot(sigma2))

    # 如果结果中有复数，则只保留实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 FID
    print('start calculate fid')
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return float(fid)
    return 0.0
