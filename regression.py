# -*- coding: utf-8 -*-
"""
Regression analysis based on RF, PLS, and SVM
Function1: Input a photo, output 
@author: Zhipeng Yin
"""
# 数据导入
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ESTIMATOR = 0
MAX_DEPTH = 0

# 计算真实值与预测值偏差


def RMSER2(true_y, predict_y):
    # 格式统一为1d nparray
    predict_y = predict_y.ravel()
    n = len(true_y)
    residue2 = np.sum((true_y-predict_y)**2)
    rmse = np.sqrt(residue2/n)
    r2 = np.corrcoef(predict_y, true_y)[0, 1]
    return rmse, r2

# 根据斜率截距计算y值


def Fx(x, slope, intercept):
    y = slope*x + intercept
    return y

# 查找目标值最近的数据点


def find_closest(array_A, target):
    # array_A must be sorted, target is an array or float
    idx = array_A.searchsorted(target)
    idx = np.clip(idx, 1, len(array_A)-1)
    left = array_A[idx-1]
    right = array_A[idx]
    idx -= target - left < right - target
    # 返回索引
    return idx


# RF预测性能
def RFcalibration(estimators, max_depth, x_train, y_train, x_test, y_test):
    regr_rf = RandomForestRegressor(
        n_estimators=estimators, max_depth=max_depth, random_state=1)
    rf = regr_rf.fit(x_train, y_train)
    # 校正线本身的x y线性程度
    predicted_y_train = rf.predict(x_train)
    predicted_y_test = rf.predict(x_test)
    # 以预测y值为y,deg=1线性拟合np.polyfit(x,y,1) z=[slope, intercept]
    z = np.polyfit(y_train, predicted_y_train, 1)
    predicted_y_train = np.array(predicted_y_train).reshape(-1, 1)
    predicted_y_test = np.array(predicted_y_test).reshape(-1, 1)
    RMSE_train, R2_train = RMSER2(y_train, predicted_y_train)
    RMSE_test, R2_test = RMSER2(y_test, predicted_y_test)
    score = rf.score(x_test, y_test)
    return predicted_y_train, predicted_y_test,\
        RMSE_train, R2_train, RMSE_test, R2_test,\
        z, score
# RF预测函数
# RF调参


def RFoptimization(x_train, y_train, x_test, y_test):
    global ESTIMATOR
    global MAX_DEPTH

    estimators = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    max_depth = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    scores = []
    for es in estimators:
        for de in max_depth:
            predicted_y_train, predicted_y_test,\
                RMSE_train, R2_train, RMSE_test, R2_test,\
                z, score = RFcalibration(
                    es, de, x_train, y_train, x_test, y_test)
            scores.append([es, de, score])
    scores = np.array(scores)
    # 按升序排列
    scores = scores[np.argsort(scores[:, 2]), :]
    # 取末尾最大值
    estimator_op = scores[-1, 0]
    depth_op = scores[-1, 1]

    ESTIMATOR = int(estimator_op)
    MAX_DEPTH = int(depth_op)


# SVM预测性能
def SVMcalibration(kernel, C, gamma, x_train, y_train, x_test, y_test):
    if kernel == 'poly':
        vr_lin = SVR(kernel='poly', C=C, gamma='auto')
    else:
        vr_lin = SVR(kernel=kernel, C=C, gamma=gamma)
    svmr = vr_lin.fit(x_train, y_train)
    # 校正线本身的x y线性程度
    predicted_y_train = svmr.predict(x_train)
    predicted_y_test = svmr.predict(x_test)
    z = np.polyfit(y_train, predicted_y_train, 1)
    predicted_y_train = np.array(predicted_y_train).reshape(-1, 1)
    predicted_y_test = np.array(predicted_y_test).reshape(-1, 1)
    RMSE_train, R2_train = RMSER2(y_train, predicted_y_train)
    RMSE_test, R2_test = RMSER2(y_test, predicted_y_test)
    score = svmr.score(x_test, y_test)
    return predicted_y_train, predicted_y_test,\
        RMSE_train, R2_train, RMSE_test, R2_test,\
        z, score
# SVM预测函数
# SVM调参


def SVMoptimization(x_train, y_train, x_test, y_test):
    kernel = ["linear", "poly", "rbf"]  # "sigmoid"存在零除错误，尚不清楚原因
    C = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    scores = []
    for ke in kernel:
        for c in C:
            for ga in gamma:
                predicted_y_train, predicted_y_test,\
                    RMSE_train, R2_train, RMSE_test, R2_test,\
                    z, score = SVMcalibration(
                        ke, c, ga, X_train, Y_train, X_test, Y_test)
                scores.append([ke, c, ga, score])
    scores = np.array(scores)
    # 按升序排列
    scores = scores[np.argsort(scores[:, 3]), :]
    # 取末尾最大值
    kernel_op = scores[-1, 0]
    C_op = float(scores[-1, 1])
    gamma_op = float(scores[-1, 2])
    return kernel_op, C_op, gamma_op


# PLS预测性能
def PLScalibration(n_comp, x_train, y_train, x_test, y_test):
    pls = PLSRegression(n_components=n_comp)
    pls = pls.fit(x_train, y_train)
    # 校正线本身的x y线性程度
    X_train_r, Y_train_r = pls.transform(x_train, y_train)
    X_test_r, Y_test_r = pls.transform(x_test, y_test)
    predicted_y_train = pls.predict(x_train)
    predicted_y_test = pls.predict(x_test)
    z = np.polyfit(y_train, predicted_y_train, 1)
    predicted_y_train = np.array(predicted_y_train).reshape(-1, 1)
    predicted_y_test = np.array(predicted_y_test).reshape(-1, 1)
    # y_train=np.array(y_train).reshape(-1,1)
    # y_test=np.array(y_test).reshape(-1,1)
    RMSE_train, R2_train = RMSER2(y_train, predicted_y_train)
    RMSE_test, R2_test = RMSER2(y_test, predicted_y_test)
    return X_train_r, Y_train_r, predicted_y_train,\
        X_test_r, Y_test_r, predicted_y_test,\
        RMSE_train, R2_train, RMSE_test, R2_test,\
        z,\
        pls.x_loadings_
# PLS预测函数
# PLS调参


def PLSoptimization(x_train, y_train, x_test, y_test):
    # 主成分数<=变量个数
    n_comp = list(range(1, x_train.shape[1]+1))
    scores = []
    for comp_i in n_comp:
        X_train_r, Y_train_r, predicted_y_train,\
            X_test_r, Y_test_r, predicted_y_test,\
            RMSE_train, R2_train, RMSE_test, R2_test,\
            z,\
            x_loadings_ = PLScalibration(
                comp_i, x_train, y_train, x_test, y_test)
        scores.append([comp_i, score, R2_train, RMSE_train])
    scores = np.array(scores)
    RMSE_deviat = scores[0, 3]-scores[-1, 3]
    # 取RMSE下降曲线的elbow,此处采用90%RMSE偏差处的点
    target_RMSE = scores[0, 3]-0.9*RMSE_deviat
    target_id = find_closest(np.ravel(scores[:, 3]), target_RMSE)
    comp = scores[target_id, 0]
    return comp


# ANN预测性能
def ANNcalibration(activation, size, alpha, x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    regr = MLPRegressor(random_state=1, max_iter=10000, activation=activation,
                        hidden_layer_sizes=size, alpha=alpha).fit(x_train, y_train)
    # 校正线本身的x y线性程度
    predicted_y_train = regr.predict(x_train)
    predicted_y_test = regr.predict(x_test)
    z = np.polyfit(y_train, predicted_y_train, 1)
    predicted_y_train = np.array(predicted_y_train).reshape(-1, 1)
    predicted_y_test = np.array(predicted_y_test).reshape(-1, 1)
    RMSE_train, R2_train = RMSER2(y_train, predicted_y_train)
    RMSE_test, R2_test = RMSER2(y_test, predicted_y_test)
    score = regr.score(x_test, y_test)
    return predicted_y_train, predicted_y_test,\
        RMSE_train, R2_train, RMSE_test, R2_test,\
        z, score
# ANN预测

# ANN调参


def ANNoptimization(x_train, y_train, x_test, y_test):
    print('test1')
    # , 'relu', 'logistic', 'tanh' 在10000步内无法收敛，可能由于数据量太少
    activation = ['identity']
    size = [5, 10, 20, 40, 50, 80, 100, [10, 2], [50, 2], [100, 2], [100, 10]]
    alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    scores = []
    for ac in activation:
        for si in size:
            for al in alpha:
                predicted_y_train, predicted_y_test,\
                    RMSE_train, R2_train, RMSE_test, R2_test,\
                    z, score = ANNcalibration(
                        ac, si, al, X_train, Y_train, X_test, Y_test)
                scores.append([ac, si, al, score])
    scores = np.array(scores, dtype=object)
    # 按升序排列
    scores = scores[np.argsort(scores[:, 3]), :]
    # 取末尾最大值
    activation_op = scores[-1, 0]
    size_op = scores[-1, 1]
    alpha_op = float(scores[-1, 2])
    return activation_op, size_op, alpha_op
