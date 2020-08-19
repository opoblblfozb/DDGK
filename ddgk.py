import tensorflow as tf # tensorflow 1._ required
import multiprocessing
import networkx as nx
import random
from tensorflow.contrib import training as contrib_training
import os
import tqdm
import collections

'''
input data requirement

dictionary : {graphid : neworkxgraph}

graphid 0, ... ,i

networkxgraph:
→ node: labeled 0,...,N integer 
'''

class ddgk:
    def __init__(self, graphdata,  score_window=10,
                num_sources=None, embedding_size=4, num_dnn_layers=4, 
                learning_rate=0.01, train_num_epochs=1000, score_num_epochs=1000,
                node_label_loss_coefficient=0, incident_label_loss_coefficient=0,
                num_node_labels=0, num_edge_labels=0):

        self.graphdata = graphdata # dictionary: {graphid : networkxgraph} nodeid should be labeld 0 ,,, M
        self.workingdir = os.getcwd()
        self.hparams = contrib_training.HParams(
                            embedding_size=embedding_size,
                            num_dnn_layers=num_dnn_layers,
                            learning_rate=learning_rate,
                            train_num_epochs=train_num_epochs,
                            score_num_epochs=score_num_epochs,
                            node_label_loss_coefficient=node_label_loss_coefficient,
                            incident_label_loss_coefficient=incident_label_loss_coefficient,
                            num_node_labels=num_node_labels,
                            num_edge_labels=num_edge_labels,
                            score_window=score_window)
        if num_sources is None:
            self.sources = graphdata
        else:
            self.sources = dict(random.sample(graphdata.items(), num_sources))
    
    def ckpt(self, k):
        ### return checkpoint prefix
        return os.path.join(self.workingdir, str(k), 'ckpt')

    def Embed(self):
        scores = collections.defaultdict(dict)

        with tqdm.tqdm(total=len(self.sources)) as pbar:
            tqdm.tqdm.write('Encoding {} source graphs...'.format(len(self.sources)))

            def encode(i):
              os.mkdir(os.path.dirname(self.ckpt(i)))
              self.MakeEncoder(self.sources[i], self.ckpt(i), self.hparams)
              pbar.update(1)

            # Alogoritmの2-9をやっているところ
            pool = multiprocessing.pool.ThreadPool(32)
            pool.map(encode, self.sources.keys())

              
        with tqdm.tqdm(total=len(self.graphdata) * len(self.sources)) as pbar:
            tqdm.tqdm.write('Scoring {} target graphs...'.format(len(self.graphdata)))
            
            def score(i):
              ### j番目のソースグラフそれぞれについて，予測誤差の計算．
              for j, source in self.sources.items():
                scores[i][j] = self.Measure(source, self.graphdata[i], self.ckpt(j), self.hparams)[-1]
                pbar.update(1)

            ### Algorithm 10-23
            pool = multiprocessing.pool.ThreadPool(32)
            pool.map(score, self.graphdata.keys())

        return scores

    def AdjMatrixLoss(self, logits, labels):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        #### 全ての要素を足し合わせて，平均を取る操作
        return tf.reduce_mean(losses)  # Report loss per edge

    def MakeEncoder(self, source, ckpt_prefix, hparams):
        #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.reset_default_graph()

        g = tf.Graph()
        session = tf.Session(graph=g)

        with g.as_default(), session.as_default():
            ##################### 定義
            ######### データを整形
            A = nx.adjacency_matrix(source, weight=None)

            x = tf.one_hot(
                list(source.nodes()), source.number_of_nodes(), dtype=tf.float64)
            y = tf.convert_to_tensor(A.todense(), dtype=tf.float64)

            ########### モデルを定義,確率値出力直前までは，すべて全結合
            layer = tf.layers.dense(x, hparams.embedding_size, use_bias=False)

            for _ in range(hparams.num_dnn_layers):
              layer = tf.layers.dense(
                  layer, hparams.embedding_size * 4, activation=tf.nn.tanh)

            logits = tf.layers.dense(
                layer, source.number_of_nodes(), activation=tf.nn.tanh)

            ############# ロス定義
            loss = self.AdjMatrixLoss(logits, y)

            ############# BackPropagation定義
            train_op = contrib_training.create_train_op(
                loss,
                tf.train.AdamOptimizer(hparams.learning_rate),
                summarize_gradients=False)

            ###################### 実行
            session.run(tf.global_variables_initializer())

            for _ in range(hparams.train_num_epochs):
              session.run(train_op)

            ##################### 保存
            tf.train.Saver(tf.trainable_variables()).save(session, ckpt_prefix)

    def Measure(self, source, target, ckpt_prefix, hparams):
        #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.reset_default_graph()

        g = tf.Graph()
        session = tf.Session(graph=g)

        with g.as_default(), session.as_default():
            ########################## モデル定義
            A = nx.adjacency_matrix(target, weight=None)

            x = tf.one_hot(
                list(target.nodes()), target.number_of_nodes(), dtype=tf.float64)
            y = tf.convert_to_tensor(A.todense(), dtype=tf.float64)

            with tf.variable_scope('attention'):
              attention = tf.layers.dense(x, source.number_of_nodes(), use_bias=False)
              source_node_prob = tf.nn.softmax(attention) 

            layer = tf.layers.dense(
                source_node_prob, hparams.embedding_size, use_bias=False)
            for _ in range(hparams.num_dnn_layers):
              layer = tf.layers.dense(
                  layer, hparams.embedding_size * 4, activation=tf.nn.tanh)
            logits = tf.layers.dense(
                layer, source.number_of_nodes(), activation=tf.nn.tanh)

            with tf.variable_scope('attention_reverse'):
              attention_reverse = tf.layers.dense(logits, target.number_of_nodes())
              # target_neighbors_pred = tf.nn.sigmoid(attention_reverse)
              # target_neighbors_prob = ProbFromCounts(target_neighbors_pred) # ラベルロスの計算に使用

            ########################### ロス定義
            loss = self.AdjMatrixLoss(attention_reverse, y)

            ########################### 訓練定義
            ## attention以外のパラメタはrestore
            vars_to_restore = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='(?!attention)') 
            ## attention, reverse_attentionのパラメタはtrain
            vars_to_train = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='attention') 

            train_op = contrib_training.create_train_op(
                loss,
                tf.train.AdamOptimizer(hparams.learning_rate),
                variables_to_train=vars_to_train,
                summarize_gradients=False)

            ############################ 実行
            session.run(tf.global_variables_initializer())

            tf.train.Saver(vars_to_restore).restore(session, ckpt_prefix) 
            losses = []

            for _ in range(hparams.score_num_epochs):
              losses.append(session.run([train_op, loss])[1])

            return losses[-hparams.score_window:]

