import numpy as np
import tensorflow as tf
import pandas as pd
#
seed_value=0
#
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)



data_path = ""
BATCH_SIZE = 64
SEQ_LENGTH = 56
EMB_DIM = 200 # embedding dimension
HIDDEN_DIM = 256 # hidden state dimension of lstm cell
corpus = "mr"
# emb_dict_file='data/{}/{}_word_vectors.txt'.format(corpus, corpus)
# sen_emb_file='data/{}/{}_doc_vectors.txt'.format(corpus, corpus)
# data_file ="data/{}/data.txt".format(corpus)
# data_label_file = "data/{}/data_label.txt".format(corpus)

emb_dict_file='mr_word_vectors.txt'
sen_emb_file='mr_doc_vectors.txt'
data_file ="data.txt"
data_label_file = "data_label.txt"

datasets = {"mr":7108, "weibo":35614, 'stackoverflow':17991, "biomeical":18000}
num_classes = 2
#negative_file = data_path + "sst_neg_sentences_id.txt"
STEPS = 10000
TRAIN_SIZE = datasets[corpus]
graph_path = "./implementation_2_graph"

class Dataloader():
    def __init__(self, batch_size, max_length = 56):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.max_length = max_length

    def load_data(self, data_file,data_label_file,vocab_dict,sen_list):
        # Load  train data
        examples = []
        label_examples = []
        with open(data_file)as fin:
            for line in fin.readlines()[:TRAIN_SIZE]:
                line = line.strip()
                line = line.split()
                parse_line=[]
                for x in line:
                    if x not in vocab_dict:
                        parse_line.append(vocab_dict['<pad>'])
                    else:
                        parse_line.append(vocab_dict[x])
                examples.append(parse_line)

        tmp_labels = []
        labels_cnt = 0
        with open(data_label_file) as f:
            for line in f.readlines()[:TRAIN_SIZE]:
                line = line.strip()
                line = line.strip("\n")
                if line not in tmp_labels:
                    labels_cnt += 1
                tmp_labels.append(line)
        tmp = pd.get_dummies(tmp_labels)
        label_examples = np.array(tmp)


        self.sentences = np.array(examples)
        self.labels= np.array(label_examples)
        self.sen_emb = np.array(sen_list[:TRAIN_SIZE])


        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sen_emb = self.sen_emb[:self.num_batch * self.batch_size]

        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)


        self.sen_emb_batches = np.split(self.sen_emb, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

        # Load  test data
        examples_test = []
        label_examples_test = []
        with open(data_file)as fin:
            for line in fin.readlines()[TRAIN_SIZE:]:
                line = line.strip()
                line = line.split()
                parse_line=[]
                for x in line:
                    if x not in vocab_dict:
                        parse_line.append(vocab_dict['<pad>'])
                    else:
                        parse_line.append(vocab_dict[x])

                examples_test.append(parse_line)

        tmp_labels = []
        labels_cnt = 0
        with open(data_label_file) as f:
            for line in f.readlines()[TRAIN_SIZE:]:
                line = line.strip()
                line = line.strip("\n")
                if line not in tmp_labels:
                    labels_cnt += 1
                tmp_labels.append(line)
        tmp = pd.get_dummies(tmp_labels)
        label_examples_test = np.array(tmp)
        # with open(data_label_file)as fin:
        #     for line in fin.readlines()[TRAIN_SIZE:]:
        #         line = line.strip()
        #         if int(line) == 1:
        #             label_examples_test.append([0, 1])
        #         else:
        #             label_examples_test.append([1, 0])

        self.sentences_test = np.array(examples_test)
        self.labels_test = np.array(label_examples_test)
        self.sen_emb_test = np.array(sen_list[TRAIN_SIZE:])



        self.num_batch_test = int(len(self.labels_test) / self.batch_size)

        self.sen_emb_test = self.sen_emb_test[:self.num_batch_test * self.batch_size]

        self.sentences_test = self.sentences_test[:self.num_batch_test * self.batch_size]
        self.labels_test = self.labels_test[:self.num_batch_test * self.batch_size]

        self.sentences_test_batches = np.split(self.sentences_test, self.num_batch_test, 0)
        self.labels_test_batches = np.split(self.labels_test, self.num_batch_test, 0)
        self.sen_emb_test_batches = np.split(self.sen_emb_test, self.num_batch_test, 0)

        self.pointer_test = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer],self.sen_emb_batches[self.pointer]
        self.pointer = (self.pointer + 1) % (self.num_batch)
        return ret

    def test_batch(self):#Preserve part of dataset for testing
        ret = self.sentences_test_batches[self.pointer_test], self.labels_test_batches[self.pointer_test],self.sen_emb_test_batches[self.pointer_test]
        self.pointer_test = (self.pointer_test + 1) % (self.num_batch_test)
     #   endtrue=False
      #  if self.pointer_test==0:
      #      endtrue=True
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Detection:
    def __init__(self, sequence_length, batch_size, vocab_size, emb_dim, hidden_dim=128, output_keep_prob=0.7):
        self.num_emb = vocab_size  # vocab size
        self.batch_size = batch_size  # batch size
        self.emb_dim = emb_dim  # dimision of embedding
        self.hidden_dim = hidden_dim  # hidden size
        self.sequence_length = sequence_length  # sequence length
        self.output_dim = num_classes
        self.output_keep_prob = output_keep_prob  # to prevent overfit
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("placeholder"):
            self.x = tf.placeholder(shape=[self.batch_size, self.sequence_length,self.emb_dim], dtype=tf.float32)



            self.x_sen = tf.placeholder(shape=[self.batch_size,self.emb_dim], dtype=tf.float32)
            self.targets = tf.placeholder(shape=[self.batch_size, self.output_dim], dtype=tf.int64)

        with tf.variable_scope("embedding"):
            #self.g_embeddings = tf.Variable(tf.random_uniform([self.num_emb, self.emb_dim], -1.0, 1.0), name="W_text")
            self.inputs = self.x  # seq_length x batch_size x emb_dim
        with tf.variable_scope("rnn"):
            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=False)  # single lstm unit
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.output_keep_prob)
            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=False)  # single lstm unit
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.output_keep_prob)

        #from tensorflow.contrib import rnn
        #rnn.static_bidirectional_rnn()

        with tf.variable_scope('attention'):
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, self.inputs,
                                                                        dtype=tf.float32)  # [64,56,256]*2
            self.W_attention = tf.get_variable(shape=[hidden_dim*2,hidden_dim*2],
                                           initializer=tf.random_normal_initializer(stddev=0.1),
                                           name='W_attention')
            self.b_attention = tf.get_variable(shape=[hidden_dim*2],name='b_attention')
            self.context_vector = tf.get_variable("what_is_the_informative_word", 
                                                  shape=[hidden_dim * 2,],
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
            # [batch_size*sequence_length, hidden_size*2]

            self.outputs = tf.concat([self.outputs[0],self.outputs[1]], axis=-1)
            #self.outputs = tf.reshape(self.outputs, shape=[-1, self.sequence_length, self.hidden_dim*2])  # [128,56,256]
            hidden_state = tf.reshape(self.outputs,[-1,hidden_dim*2])
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state,self.W_attention) + self.b_attention)
            hidden_representation = tf.reshape(hidden_representation, shape=[-1,sequence_length,hidden_dim * 2])
            # 计算相似度
            hidden_state_context_similiarity = tf.multiply(hidden_representation,self.context_vector)
            attention_logits = tf.reduce_sum(hidden_state_context_similiarity,axis=2)
            # 为了防止softmax溢出，所以用logits减去max，再进行softmax
            attention_logits_max = tf.reduce_max(attention_logits, axis=1,keep_dims=True)
            p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
            p_attention_expanded = tf.expand_dims(p_attention, axis=2)
            # 加权求和得到表示句子的向量
            sentence_representation = tf.multiply(p_attention_expanded,self.outputs)
            sentence_representation = tf.reduce_sum(sentence_representation,axis=1)

        with tf.name_scope('dropout'):
            # dropout防止过拟合
            sentence_representation = tf.concat([sentence_representation, self.x_sen], 1)  # [64,456]和句子向量拼接
            self.rnn_drop = tf.nn.dropout(sentence_representation, keep_prob=0.7)
            #self.rnn_drop = sentence_representation

        with tf.name_scope('output'):
            tf.layers.dense(self.outputs, self.output_dim, name="logits")

            W = tf.get_variable(shape=[hidden_dim * 2+EMB_DIM, self.output_dim],initializer=tf.contrib.layers.xavier_initializer(),name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.output_dim]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # 将dropout的输出乘以w再加b
            self.logits = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="logits")
            self.prob = tf.nn.softmax(self.logits, name="softmax_output")

        '''
        with tf.variable_scope("output"):
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, self.inputs, dtype=tf.float32)#[64,56,256]*2
            self.outputs = tf.reshape(self.outputs, shape=[-1, self.sequence_length, self.hidden_dim])#[128,56,256]
            self.outputs = tf.transpose(self.outputs, perm=[1, 0, 2])  #[56,128,256]
            self.outputs = tf.reduce_mean(self.outputs, 0)#[128,256]
            self.outputs = self.outputs[:self.batch_size] + self.outputs[self.batch_size:]#[64,256]
            self.outputs=tf.concat([self.outputs, self.x_sen], 1)#[64,456]

            #self.outputs=self.x_sen
            self.logits = tf.layers.dense(self.outputs, self.output_dim, name="logits")
            self.prob = tf.nn.softmax(self.logits, name="softmax_output")
        '''

        with tf.variable_scope("train"):
            l2_reg_lambda=0.1
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits))+ l2_reg_lambda * l2_loss
            tvars = tf.trainable_variables()
            max_grad_norm = 5
            # We clip the gradients to prevent explosion
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            gradients = list(zip(grads, tvars))
            self.train_op = tf.train.AdamOptimizer(0.001).apply_gradients(gradients)

            # 不加att 0.001+256 hid：0.78435；0.0005+256 hid：0.7832  ;0.005+256 hid：0.7732
            # 加att 0.001+256 hid：0.7818
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.equal(tf.argmax(self.targets, axis=1), tf.argmax(self.prob, axis=1))

    def train(self, sess, x_batch, y_batch,x_sen):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={self.x: x_batch, self.targets: y_batch, self.x_sen: x_sen})
        return loss

    def predict(self, sess, x_batch):
        prob = sess.run([self.prob], feed_dict={self.x: x_batch})
        return prob

    def get_accuracy(self, sess, x_batch, y_batch,x_sen):
        accuracy = sess.run([self.accuracy], feed_dict={self.x: x_batch, self.targets: y_batch, self.x_sen: x_sen})
        return (accuracy[0].tolist().count(True) / len(x_batch))


def load_emb_data(emb_dict_file,sen_emb_file):
    word_dict = {}
    word_list = []
    with open(emb_dict_file, 'r') as f:
        lines = f.readlines()
        emb_pad=np.random.rand(EMB_DIM)
        word="<pad>"
        word_dict[word]=emb_pad
        for line in lines:
            word = line.strip().split()[0]
            word_dict[word] = np.array([float(x) for x in line.strip().split()[1:]])
            word_list.append(word)
    length = len(word_dict)

    sen_list = []
    with open(sen_emb_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sen_list.append(np.array([float(x) for x in line.strip().split()[1:]]))

    return word_dict, length, word_list,sen_list




if __name__ == "__main__":
    vocab_dict, vocab_size, vocab_list,sen_list = load_emb_data(emb_dict_file,sen_emb_file)
    dis_data_loader = Dataloader(BATCH_SIZE, SEQ_LENGTH)

    ''''''


    dis_data_loader.load_data(data_file,data_label_file,vocab_dict,sen_list)
    detection = Detection(SEQ_LENGTH, BATCH_SIZE, vocab_size, EMB_DIM, HIDDEN_DIM)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter(graph_path, sess.graph)
        # writer.close()
        accuracy_best=0.0
        loss_all=0.0
        for i in range(STEPS):
            x_batch, y_batch,x_sen= dis_data_loader.next_batch()
            loss = detection.train(sess, x_batch, y_batch,x_sen)
            loss_all=loss_all+loss
            if (i % 100 == 0):
                accuracy_list=[]
                for j in range(dis_data_loader.num_batch_test):
                    test_x_batch, test_y_batch,x_test_sen = dis_data_loader.test_batch()
                    accuracy_list.append(detection.get_accuracy(sess, test_x_batch, test_y_batch,x_test_sen))
                accuracy=np.mean(np.array(accuracy_list))
                print("%d, loss:%f,loss_all:%f, accuracy:%f" % (i, loss,loss_all, float(accuracy)))
                if accuracy_best<accuracy:
                    accuracy_best=accuracy
                print("%d, best_accuracy:%f" % (i, accuracy_best))
                print("=======================================")
                loss_all = 0.0
