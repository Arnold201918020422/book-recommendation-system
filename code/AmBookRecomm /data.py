# -*- coding: utf-8 -*-
# @Time    : 2023/3/18 23:04
# @Author  : alpha
# @File    : data.py
# @Software: PyCharm
# @desc    :
import gzip
import json

import pandas as pd
from stqdm import stqdm

meta_path = "D://data//amazon//meta_Books.json.gz"


def parse_jsongz(jsongz_path):
    json_array = []
    count = 0
    with gzip.open(jsongz_path, 'rt', encoding='utf-8') as zipfile:
        for line in stqdm(zipfile):
            # print(line)
            rjson = json.loads(line)
            json_array.append(rjson)
            count += 1
    return pd.json_normalize(json_array)


def hasImagBooks():
    book_meta_df = parse_jsongz(meta_path)
    # book_meta_df = book_meta_df.dropna(subset=["imageURL"])
    imgBook_meta_df = book_meta_df[book_meta_df["imageURL"].map(lambda x: len(str(x)) > 3)]
    imgBook_meta_df.to_csv("data/am_img_books.csv", encoding='utf8')


if __name__ == '__main__':
    hasImagBooks()
