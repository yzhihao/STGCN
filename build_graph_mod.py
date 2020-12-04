import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MinMaxScaler
import pickle

from biterm.btm import oBTM
from biterm.utility import vec_to_biterms, topic_summuary
import Biterm_sampler



datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'weibo']
# build corpus
dataset = 'mr'

# dataset = "weibo"

word_embeddings_dim = 100
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)

doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1



new_word_freq = sorted(word_freq.items(),key=lambda item:item[1], reverse=True)[:5000]

vocab = [x[0] for x in new_word_freq]
vocab_size = len(vocab)
shuffle_doc_words_list_ = []
for doc_word in shuffle_doc_words_list:
    tmp_list = doc_word.split(" ")
    tmp = list(set(tmp_list).intersection(set(vocab)))
    tmp_str = " ".join(tmp)
    shuffle_doc_words_list_.append(tmp_str)

shuffle_doc_words_list = shuffle_doc_words_list_

# vocab = list(word_set)
# vocab_size = len(vocab)


word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i


#====get shuffle_doc_words_list_index
shuffle_doc_words_list_index=[]
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    doc_words_index=[]
    for word in words:
        doc_words_index.append(word_id_map[word])
    shuffle_doc_words_list_index.append(doc_words_index)

#====get shuffle_doc_words_list_index end


vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()


def cnt_freq(word_list):
    tmp_list = []
    doc_word_freq = []
    word_freq_dct = {}
    for word in word_list:
        if word not in tmp_list:
            word_freq_dct[word_id_map[word]] = 1
        else:
            word_freq_dct[word_id_map[word]] += 1
        tmp_list.append(word)
    for i,item in enumerate(word_freq_dct):
        tmp_str = str(item) + ":" + str(word_freq_dct[item])
        doc_word_freq.append(tmp_str)
    return doc_word_freq


'''
build mr corpus to nvdm
'''

# vocab dict
f = open('data/mr/' + 'vocab.pkl', 'wb')
pkl.dump(word_id_map,f)
f.close()


# training data
doc_word_ids = []
for doc_meta,doc in zip(shuffle_doc_name_list,shuffle_doc_words_list):
    tmp_list = []
    # if doc_meta.split('\t')[1] == 'test':
    #     continue
    label = doc_meta.split('\t')[2]
    # tmp_list.append(label)
    word_list = doc.split()
    for item in word_list:
        tmp_list.append(word_id_map[item])
    doc_word_ids.append(np.array(tmp_list))
    print(tmp_list)

f = open('data/mr/' + 'train' + '.pkl', 'wb')
pkl.dump(doc_word_ids, f)
f.close()

# testing data
doc_word_ids_te = []
for doc_meta,doc in zip(shuffle_doc_name_list,shuffle_doc_words_list):
    tmp_list = []
    if doc_meta.split('\t')[1] == 'train':
        continue
    label = doc_meta.split('\t')[2]
    # tmp_list.append(label)
    word_list = doc.split()
    for item in word_list:
        tmp_list.append(word_id_map[item])
    doc_word_ids_te.append(np.array(tmp_list))
    print(tmp_list)


'''
Word definitions begin
'''

definitions = []


'''
Word definitions end
'''
topic_size = 30

# BTM
data_for_btm = [x.split() for x in shuffle_doc_words_list]
doc_topic, topic_word = Biterm_sampler.fit_transform(data_for_btm, iter=1, num_topics=topic_size)

print("123"+str(doc_topic.shape))

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector


'''
for i, topic_dist in enumerate(topic_vec):
     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
'''

#=========add topic features
topic_vectors = np.random.uniform(-0.01, 0.01,
                                 (topic_size, word_embeddings_dim))

row_topx = []
col_topx = []
data_topx = []
for i in range(topic_size):
    for j in range(word_embeddings_dim):
        row_topx.append(i)
        col_topx.append(j)
        data_topx.append(topic_vectors[i][j])

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
topx = sp.csr_matrix((data_topx, (row_topx, col_topx)),
                   shape=(topic_size, word_embeddings_dim))

topy = []
for i in range(topic_size):
    one_hot = [0 for l in range(len(label_list))]
    topy.append(one_hot)
topy = np.array(topy)

#=====


row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []
#
# def compute_sim(v1, v2):
#     outputs = 1.0 - cosine(vector_i, vector_j)
#     return outputs

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    # similarity = 0.0
    # pmi = (1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j/(num_window * num_window))
    # if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
    #     print('Vector i is:')
    #     print(vector_i)
    #     print('--------------')
    #     vector_i = np.array(word_vector_map[vocab[i]])
    #     vector_j = np.array(word_vector_map[vocab[j]])
    #     # similarity = 1.0 - cosine(vector_i, vector_j)
    #     similarity = compute_sim(vector_i, vector_j)
    #     pmi *= similarity
    #
    # pmi = log(pmi+similarity)
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)


# word vector cosine similarity as weights

# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size


print("3333"+str(doc_topic.shape))
ret=0
for i in range(len(shuffle_doc_words_list)):
    for j in range(topic_size):#50*10662=
        #doc_topic_weight = doc_topic[i][j]
        #row.append(i)
        #if doc_topic[i][j]<0.005:
        #    ret += 1
        #    continue
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        #print(doc_topic.shape)
        col.append(j + node_size)
        weight.append(doc_topic[i][j])


print("doc_topic"+str(ret))

ret=0
for i in range(topic_size):#50*18764=938,200
    for j in range(len(word_id_map)):
        if topic_word[i][j]<0.000001:
            ret+=1
            continue
        row.append(i + node_size)
        col.append(j + train_size)
        weight.append(topic_word[i][j])
print("topic_word"+str(ret))

node_size += topic_size

adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))
print(adj.shape)
print(allx.shape)


# print(doc_vec)
# print(word_vectors)

# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.topx".format(dataset), 'wb')
pkl.dump(topx, f)
f.close()

f = open("data/ind.{}.topy".format(dataset), 'wb')
pkl.dump(topy, f)
f.close()


f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()
