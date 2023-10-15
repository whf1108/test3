

"""from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss"""

import pandas as pd
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
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
    data[feature] = lbe.fit_transform(data[feature]) + 1   # 应该不需要再加1了吧？？
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id"]].drop_duplicates('movie_id')
user_profile.set_index("user_id", inplace=True)  # 设置user索引

user_item_list = data.groupby("user_id")['movie_id'].apply(list)
train_set, test_set = gen_data_set(data, negsample)  # [(1,[movie_id 序列],1个movie_Id,正样本1，movie序列长度,评分),(2,...)]

train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)  # {‘use_id’:array[],'gen':array[],'movid_序列'：[[],[],]}  label:array []
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

# 2. 配置一下模型定义需要的特征列，主要是特征名和embedding词表的大小

embedding_dim = 16

user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                        SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                        SparseFeat("age", feature_max_idx['age'], embedding_dim),
                        SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                        SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

# 3. 定义一个YoutubeDNN模型，分别传入用户侧特征列表`user_feature_columns`和物品侧特征列表`item_feature_columns`。然后配置优化器和损失函数，开始进行训练。

K.set_learning_phase(True)

model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, 16))
# model = MIND(user_feature_columns,item_feature_columns,dynamic_k=True,p=1,k_max=2,num_sampled=5,user_dnn_hidden_units=(64,16),init_std=0.001)

model.compile(optimizer="adagrad", loss=sampledsoftmaxloss)  # "binary_crossentropy")

history = model.fit(train_model_input, train_label,  # train_label,
                    batch_size=256, epochs=1, verbose=1, validation_split=0.0, )

# 4. 训练完整后，由于在实际使用时，我们需要根据当前的用户特征实时产生用户侧向量，并对物品侧向量构建索引进行近似最近邻查找。这里由于是离线模拟，所以我们导出所有待测试用户的表示向量，和所有物品的表示向量。

test_user_model_input = test_model_input
all_item_model_input = {"movie_id": item_profile['movie_id'].values, "movie_idx": item_profile['movie_id'].values}

# 以下两行是deepmatch中的通用使用方法，分别获得用户向量模型和物品向量模型
user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
# 输入对应的数据拿到对应的向量
user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
# user_embs = user_embs[:, i, :]  i in [0,k_max) if MIND
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print(user_embs.shape)
print(item_embs.shape)

# 5. [可选的]如果有安装faiss库的同学，可以体验以下将上一步导出的物品向量构建索引，然后用用户向量来进行ANN查找并评估效果

test_true_label = {line[0]: [line[2]] for line in test_set}
import numpy as np
import faiss
from tqdm import tqdm
from deepmatch.utils import recall_N

index = faiss.IndexFlatIP(embedding_dim)
# faiss.normalize_L2(item_embs)
index.add(item_embs)
# faiss.normalize_L2(user_embs)
D, I = index.search(user_embs, 50)
s = []
hit = 0
for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    try:
        pred = [item_profile['movie_id'].values[x] for x in I[i]]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=50)
        s.append(recall_score)
        if test_true_label[uid] in pred:
            hit += 1
    except:
        print(i)
print("recall", np.mean(s))
print("hr", hit / len(test_user_model_input['user_id']))
