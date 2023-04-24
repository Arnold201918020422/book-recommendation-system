# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 17:24
# @Author  : alpha
# @File    : ucf.py
# @Software: PyCharm
# @desc    :
import collections
import math
import os
import pickle
import sys

import django
from tqdm import tqdm

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd + "../")
# 找到根目录（与工程名一样的文件夹）下的settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AmBookRecomm.settings')
django.setup()

from user import models

# ret = models.Music.objects.all()
# print(len(ret))
# ret = models.User.objects.all()
# print(len(ret))
records = models.Rate.objects.all()

train = collections.defaultdict(dict)

for rec in tqdm(records):
    user = rec.user.id
    item = rec.book.id
    score = rec.mark
    train[user][item] = score
print(train)

similarity = "cosine"
similarity = "iif"
# 得到每个item被哪些user评价过
item_user = dict()
for user, items in train.items():
    for item in items:
        item_user.setdefault(item, set())
        item_user[item].add(user)
print("item->users: ", len(item_user))

userSimMatrix = collections.defaultdict(dict)
# 建立用户物品交集矩阵W, 其中C[u][v]代表的含义是用户u和用户v之间共同喜欢的物品数
for item, users in item_user.items():
    for u in users:
        userSimMatrix.setdefault(u, collections.defaultdict(int))
        for v in users:
            if u == v:
                continue
            if similarity == "cosine":
                userSimMatrix[u][v] += 1  # 将用户u和用户v共同喜欢的物品数量加一
            elif similarity == "iif":
                userSimMatrix[u][v] += 1. / math.log(1 + len(users))

# print("userSimMatrix:", userSimMatrix)
# 建立用户相似度矩阵
for u, related_user in userSimMatrix.items():
    # 相似度公式为 |N[u]∩N[v]|/sqrt(N[u]||N[v])
    for v, cuv in related_user.items():
        nu = len(train[u])
        nv = len(train[v])
        userSimMatrix[u][v] = cuv / math.sqrt(nu * nv)
print("F userSimMatrix:", userSimMatrix)
print("DONE")

with open("data/ucf.pkl", mode='wb') as f:
    pickle.dump(userSimMatrix, f)
