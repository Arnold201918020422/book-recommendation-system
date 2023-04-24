# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 15:45
# @Author  : alpha
# @File    : crontab.py.py
# @Software: PyCharm
# @desc    :
import os
import sys

import django
import pandas as pd

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd + "../")
# 找到根目录（与工程名一样的文件夹）下的settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AmBookRecomm.settings')
django.setup()


def init_book():
    """
    获取调度方案
    :return:
    """
    from user import models

    ret = models.Book.objects.all()
    # models.Music.objects.all().delete()
    print("opt ====>", ret)

    mdf = pd.read_csv("data/am_img_books.csv", encoding='utf8')
    # print(mdf.columns)
    # print(mdf.head())
    for i, row in mdf.iterrows():
        # print(row)
        desc = eval(row['description'])[0] if len(row['description']) > 3 else ""
        imag = eval(row['imageURL'])[0] if len(row['imageURL']) > 3 else ""
        book, bct = models.Book.objects.get_or_create(title=row['title'],
                                                      asin=row['asin'],
                                                      brand=row['brand'],
                                                      years=row['date'],
                                                      imageURL=imag,
                                                      description=desc,
                                                      # defaults={'artist': artist_name, "pic": img_url, 'album': album_name,'lyric': lyric, 'years': publish_time}
                                                      )
        cates = eval(row['category'])
        for cate in cates:
            tag_str = str(cate).replace('&amp;', '&')
            tag, cct = models.Category.objects.get_or_create(name=tag_str)
            # 将多对多的关系添加到book中
            book.tags.add(tag)
            # 保存book的多对多关系
            book.save()


if __name__ == '__main__':
    init_book()
