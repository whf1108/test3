
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 以movielens数据为例，取200条样例数据进行流程演示
# user_id,movie_id,rating,timestamp,title,genres,gender,age,occupation,zip
data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
SEQ_LEN = 50
negsample = 0

# 1. 首先对于数据中的特征进行ID化编码，然后使用 `gen_date_set` and `gen_model_input`来生成带有用户历史行为序列的特征数据

features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}
for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1   # 有n个类别的话--1,2,3,4...n

    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id"]].drop_duplicates('movie_id')
user_profile.set_index("user_id", inplace=True)
user_item_list = data.groupby("user_id")['movie_id'].apply(list)


#
data.sort_values("timestamp", inplace=True)
item_ids = data['movie_id'].unique()

import random
data.sort_values("timestamp", inplace=True)
item_ids = data['movie_id'].unique()
import numpy as np
train_set = []
test_set = []
for reviewerID, hist in tqdm(data.groupby('user_id')):
    pos_list = hist['movie_id'].tolist()
    rating_list = hist['rating'].tolist()

    if negsample > 0:
        candidate_set = list(set(item_ids) - set(pos_list))
        neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]),rating_list[i]))
            for negi in range(negsample):
                train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
        else:
            test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

random.shuffle(train_set)
random.shuffle(test_set)

train_uid = np.array([line[0] for line in train_set])
train_seq = [line[1] for line in train_set]
train_iid = np.array([line[2] for line in train_set])
train_label = np.array([line[3] for line in train_set])
train_hist_len = np.array([line[4] for line in train_set])

# train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
train_model_input = {"user_id": train_uid, "movie_id": train_iid,
                     "hist_len": train_hist_len}

for key in ["gender", "age", "occupation", "zip"]:
    print(user_profile.loc[train_model_input['user_id']][key].values)
    train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

print(train_model_input)