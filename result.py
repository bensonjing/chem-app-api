# Input: user provided image information
# Output: predicted concentration

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


import userData
import regression


# get user data
data, variable = userData.read_data("photo/info.txt")
X_train, X_test, Y_train, Y_test = train_test_split(
    data, variable, test_size=0.2, shuffle=True, random_state=1)
work_X = X_train[1, :].reshape(-1, 3)


def RFprediction(estimators, max_depth, x_train, y_train, work_x):
    regr_rf = RandomForestRegressor(
        n_estimators=estimators, max_depth=max_depth, random_state=1)
    rf = regr_rf.fit(x_train, y_train)
    work_y = rf.predict(work_x)
    work_y = np.array(work_y).reshape(-1, 1)
    return work_y


def SVMprediction(kernel, C, gamma, x_train, y_train, work_x):
    vr_lin = SVR(kernel=kernel, C=C, gamma=gamma)
    svmr = vr_lin.fit(x_train, y_train)
    work_y = svmr.predict(work_x)
    work_y = np.array(work_y).reshape(-1, 1)
    return work_y


def PLSprediction(n_comp, x_train, y_train, work_x):
    pls = PLSRegression(n_components=n_comp)
    pls = pls.fit(x_train, y_train)
    work_y = pls.predict(work_x)
    return work_y


def ANNprediction(activation, size, alpha, x_train, y_train, work_x):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    work_x = scaler.transform(work_x)
    regr = MLPRegressor(random_state=1, max_iter=10000, activation=activation,
                        hidden_layer_sizes=size, alpha=alpha).fit(x_train, y_train)
    work_y = regr.predict(work_x)
    return work_y


# estimators, max_depth = regression.RFoptimization(
#     X_train, Y_train, X_test, Y_test)
# kernel, C, gamma = regression.SVMoptimization(X_train, Y_train, X_test, Y_test)
# n_comp = 3
# activation, size, alpha = regression.ANNoptimization(
#     X_train, Y_train, X_test, Y_test)


output_rf = RFprediction(1, 1, X_train, Y_train, work_X)
# output_svm = SVMprediction(kernel, C, gamma, X_train, Y_train, work_X)
# output_pls = PLSprediction(n_comp, X_train, Y_train, work_X)
# output_ann = ANNprediction(activation, size, alpha, X_train, Y_train, work_X)

print(output_rf[0][0])
