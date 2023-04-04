import numpy as np
import pandas as pd
import datetime


from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

import torchkeras

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n\n')



from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

dfdata = pd.read_csv("train_1m.txt", sep="\t", header=None)
dfdata.columns = ["label"] + ["I" + str(x) for x in range(1, 14)] + [
    "C" + str(x) for x in range(14, 40)]

cat_cols = [x for x in dfdata.columns if x.startswith('C')]
num_cols = [x for x in dfdata.columns if x.startswith('I')]
num_pipe = Pipeline(steps=[('impute', SimpleImputer()), ('quantile', QuantileTransformer())])

for col in cat_cols:
    dfdata[col] = LabelEncoder().fit_transform(dfdata[col])

dfdata[num_cols] = num_pipe.fit_transform(dfdata[num_cols])

categories = [dfdata[col].max() + 1 for col in cat_cols]



# DataFrame转换成torch数据集Dataset, 特征分割成X_num,X_cat方式
class DfDataset(Dataset):
    def __init__(self, df,
                 label_col,
                 num_features,
                 cat_features,
                 categories,
                 is_training=True):

        self.X_num = torch.tensor(df[num_features].values).float() if num_features else None
        self.X_cat = torch.tensor(df[cat_features].values).long() if cat_features else None
        self.Y = torch.tensor(df[label_col].values).float()
        self.categories = categories
        self.is_training = is_training

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        if self.is_training:
            return ((self.X_num[index], self.X_cat[index]), self.Y[index])
        else:
            return (self.X_num[index], self.X_cat[index])

    def get_categories(self):
        return self.categories


dftrain_val, dftest = train_test_split(dfdata, test_size=0.2)
dftrain, dfval = train_test_split(dftrain_val, test_size=0.2)

ds_train = DfDataset(dftrain, label_col="label", num_features=num_cols, cat_features=cat_cols,
                     categories=categories, is_training=True)

ds_val = DfDataset(dfval, label_col="label", num_features=num_cols, cat_features=cat_cols,
                   categories=categories, is_training=True)

ds_test = DfDataset(dftest, label_col="label", num_features=num_cols, cat_features=cat_cols,
                    categories=categories, is_training=True)
dl_train = DataLoader(ds_train, batch_size=2048, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=2048, shuffle=False)
dl_test = DataLoader(ds_test, batch_size=2048, shuffle=False)

for features, labels in dl_train:
    break