# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 15:45
# @Author  : alpha
# @File    : crontab.py.py
# @Software: PyCharm
# @desc    :
import collections
import gzip
import json
import os
import random
import sys

import django

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd + "../")
# 找到根目录（与工程名一样的文件夹）下的settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AmBookRecomm.settings')
django.setup()
from user.models import *

review_path = "D://data//amazon//reviews_Books_5.json.gz"


def parse_common(jsongz_path, asinTop=2000, reviewTop=3000):
    asins = []
    reviewers = []
    count = 0
    with gzip.open(jsongz_path, 'rt', encoding='utf-8') as zipfile:
        for line in zipfile:
            # print(line)
            rjson = json.loads(line)
            asin = rjson['asin']
            reviewerID = rjson['reviewerID']
            asins.append(asin)
            reviewers.append(reviewerID)
            count += 1
    asinsCounter = collections.Counter(asins)
    reviewerCounter = collections.Counter(reviewers)
    topAsins = asinsCounter.most_common(n=asinTop)
    topReviewers = reviewerCounter.most_common(n=reviewTop)
    return topAsins, topReviewers


def random_phone():
    res = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    return res


def random_book_id(num=5):
    book_nums = Book.objects.all().order_by('?').values('id')[:num]
    return [book['id'] for book in book_nums]


def random_mark():
    return random.randint(1, 5)


def init_user():
    topAsins, topReviewers = parse_common(review_path)

    for rev, cnt in topReviewers:
        user_name = rev
        try:
            user, created = User.objects.get_or_create(username=user_name,
                                                       name=user_name,
                                                       defaults={'password': user_name, "phone": random_phone(),
                                                                 "address": user_name,
                                                                 "email": user_name + '@163.com'})
            for book_id in random_book_id():
                Rate.objects.get_or_create(user=user, book_id=book_id, defaults={"mark": random_mark()})
        except Exception as e:
            raise e


if __name__ == '__main__':
    init_user()
