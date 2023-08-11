# -*- coding: utf-8 -*-
"""
Created on 2023/7/28 13:55 
@Author: Wu Kaixuan
@File  : dataset_split.py 
@Desc  : dataset_split 
"""

import os
import random
import pandas as pd

data = pd.read_csv("captions.txt",sep=",")
images = data.image.unique()
# 从images里随机采样1000个作为测试集
test_images = random.sample(list(images), 500)
print(test_images)

#把test_images从data里取出来，并删除
test_data = data[data.image.isin(test_images)]
train_data = data[~data.image.isin(test_images)]

test_data.to_csv("test_captions.csv", index=False)
train_data.to_csv("train_captions.csv", index=False)
