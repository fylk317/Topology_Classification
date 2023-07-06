# -*- coding: utf-8 -*-
# In[]
import cmath
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
import math
from math import *
import scipy as sp
from scipy.linalg import expm, sinm, cosm
import random
#from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import polyfit, poly1d
#import matplotlib
#from matplotlib import cm
#from scipy.optimize import curve_fit 
import numpy as np
from sklearn.model_selection import train_test_split
from mpmath import *
import decimal
import torch
from torch.utils.data import Dataset, DataLoader

def get_data(L, LL, N, device):

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress = True)
    np.set_printoptions(precision= 8)

    decimal.getcontext().prec = 6
    mp.dps=10

    #import matplotlib
    #from matplotlib.font_manager import FontProperties
    #from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
    #import matplotlib.ticker as ticker

    #font = FontProperties()
    #font.set_family('Times New Roman')
    #font.set_size(15)
    #plt.rcParams['xtick.labelsize'] = 15
    #plt.rcParams['ytick.labelsize'] = 15

    def hx(axn,bxn,n,k):
        return axn*np.cos(n*k) + bxn*np.sin(n*k)


    def hy(ayn,byn,n,k):
        return ayn*np.cos(n*k) + byn*np.sin(n*k)


    # 创建形状为 (L, N) 的二维数组，其中的元素随机取自 -1 到 1
    axn = 2 * np.random.rand(L, N) - 1
    bxn = 2 * np.random.rand(L, N) - 1
    ayn = 2 * np.random.rand(L, N) - 1
    byn = 2 * np.random.rand(L, N) - 1

    k_range = np.linspace(0, 2*np.pi,LL,endpoint=True)

    # 初始化HX
    HX = np.zeros((L, LL))
    HY = np.zeros((L, LL))
    # 计算HX中的每个值
    for i in range(L):
        for j, k in enumerate(k_range):
            for n in range(5):  # 对n进行求和，范围是[0, 4]
                HX[i, j] += hx(axn[i,n], bxn[i,n], n, k)
                HY[i, j] += hy(ayn[i,n], byn[i,n], n, k)
    # print(r'HX',HX)
    # print(r'HY',HY)

    # 使用column_stack将HX和HY堆叠起来，形成一个新的数组
    input_data = np.column_stack((HX.flatten(), HY.flatten()))

    # 将input_data数组重新分割为L组，每组包含LL个点的数据
    input_data = input_data.reshape(L, LL, 2)

    # print(input_data)
    for i in range(L):
        for j in range(LL):
            norm = np.sqrt((input_data[i,j,0])**2+(input_data[i,j,1])**2)
            input_data[i,j,0] = input_data[i,j,0]/norm
            input_data[i,j,1] = input_data[i,j,1]/norm

    # print(r'final input',input_data)
    # print(r'first',input_data[0,:,:])

    U = np.zeros((L, LL, 1),dtype=complex)  # 初始化新的数组

    for i in range(L):
        for j in range(LL):
            u = input_data[i,j,0] + 1j*input_data[i,j,1]
            # theta = np.angle(u)
            U[i, j, 0] = u#theta
        
    # print(r'angle',U)

    W = []
    for i in range(L):
        w = 0
        for j in range(LL-1):
            w = w - np.angle((U[i,j+1,0]/U[i,j,0]))/np.pi/2
        W.append(round(np.abs(w)))
    W = np.array(W)


    input_data_train, input_data_test, W_train, W_test= train_test_split(input_data, W, test_size=0.3, shuffle=True)  

    return [torch.FloatTensor(input_data_train).to(device),torch.from_numpy(W_train).to(device)], [torch.FloatTensor(input_data_test).to(device), torch.from_numpy(W_test).to(device)]
#print(input_data_train.shape,W_train.shape)

def get_batch(data, i , batch_size):
    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(data[0]) - i)
    input = data[0][ i:i + batch_len ]
    target = data[1][ i:i + batch_len ]
    return input, target