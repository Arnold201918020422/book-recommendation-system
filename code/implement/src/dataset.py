import torch
import torch.utils.data as data
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
import random
from copy import deepcopy
import numpy as np

DATA_DIR = os.path.join("/Users/chengfujia/Downloads/10_DeepCrossNetwork/data/mix_data.csv")


class DataSetV2(data.Dataset):
    def __init__(self, data_dir, mode='train', ratio=0.8):
        super(DataSetV2, self).__init__()
        labelencoder = LabelEncoder()
        self.data_dir = data_dir
        self.data = pd.read_csv(DATA_DIR)
        self.data.dropna(inplace=True)
        self.data = self.data[['user_id', 'age', 'book_author', 'year_of_publication', 'publisher', 'rating']]

        # min and max ratings will be used to normalize the ratings
        self.min_rating = min(self.data["rating"])
        self.max_rating = max(self.data["rating"])

        # 将字符串特征转换成数字特征
        self.data.iloc[:, 2] = labelencoder.fit_transform(self.data.iloc[:, 2])
        self.data.iloc[:, 4] = labelencoder.fit_transform(self.data.iloc[:, 4])

        self.sparse_feature = deepcopy(self.data.iloc[:, :-1]).values

        self.data.iloc[:, 0] = KBinsDiscretizer(n_bins=200, encode="ordinal", strategy="uniform").fit_transform(
            self.data.iloc[:, 0].values.reshape(-1, 1))
        self.data.iloc[:, 2] = KBinsDiscretizer(n_bins=200, encode="ordinal", strategy="uniform").fit_transform(
            self.data.iloc[:, 2].values.reshape(-1, 1))
        self.data.iloc[:, 4] = KBinsDiscretizer(n_bins=200, encode="ordinal", strategy="uniform").fit_transform(
            self.data.iloc[:, 4].values.reshape(-1, 1))

        self.dense_feature = OneHotEncoder().fit_transform(self.data.iloc[:, :-1]).toarray()

        self.dataset = [(self.sparse_feature[i], self.dense_feature[i], (self.data.iloc[:, -1].values[i] - self.min_rating) / (self.max_rating - self.min_rating)) for i in range(self.sparse_feature.shape[0])]
        random.shuffle(self.dataset)
        if mode == 'train':
            self.dataset = self.dataset[:int(ratio * len(self.dataset))]
        elif mode == 'valid':
            self.dataset = self.dataset[int(ratio * len(self.dataset)):]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return torch.from_numpy(self.dataset[item][0]).to(torch.int64), \
               torch.from_numpy(self.dataset[item][1]), \
               torch.tensor(self.dataset[item][2], dtype=torch.float32)


if __name__ == '__main__':
    data = DataSetV2(DATA_DIR)
    print('res')