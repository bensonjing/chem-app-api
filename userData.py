# -*- coding: utf-8 -*-
"""
@author: Zhipeng Yin
Task: The regression analysis of DGT detecting antimony species.
The input data should be photo that has been cut into regular shape.
The output is a calibration curve and a function that can predict antimony concentration.

Primary protocol: A database is integrated in the app. End-users do not have to 
produce the calibration curve by themselves. They take a photo and the result should
be popped out.

Second protocol: Allow end-users to produce their own cal. curve. They should upload a
photo and then input a number.
"""

import pandas as pd
import numpy as np
from PIL import Image


img = Image.open('photo/0.jpg')


def read_data(datainfo):
    info = pd.read_table(datainfo, sep="\t", header="infer")
    filename = info["file_name"]
    count = 0
    for file in filename:
        count = count+1
        # read single photo
        file = "photo/"+file
        img_tp = Image.open(file)
        arr_tp = np.array(img_tp)
        # average the pixels for RGB
        R_p = arr_tp[:, :, 0]
        R_p = np.sum(R_p)/(R_p.shape[0]*R_p.shape[1])
        G_p = arr_tp[:, :, 1]
        G_p = np.sum(G_p)/(G_p.shape[0]*G_p.shape[1])
        B_p = arr_tp[:, :, 2]
        B_p = np.sum(B_p)/(B_p.shape[0]*B_p.shape[1])
        # d_temp=pd.read_table(file,sep=datasep,header=datahead,names=colname,\
        #                  decimal=datadecimal,dtype=np.float64)
        d_temp = np.array([R_p, G_p, B_p]).reshape(1, -1)
        if count == 1:
            data = d_temp
            # variable=info["Concentration"][count-1]
        else:
            # print(file) 检查读取的每个文件
            data = np.vstack((data, d_temp))
    variable = info["Concentration"]
    return data, variable
