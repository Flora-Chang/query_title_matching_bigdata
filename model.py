# encoding: utf-8
import tensorflow as tf
from util import FLAGS

class Model(object):
    def __init__(self, max_query_word, max_title_word,word_vec_initializer, batch_size, filter_size,
                 vocab_size, embedding_size, learning_rate, keep_prob):
        self.word_vec_initializer = word_vec_initializer
        self.BATCH_SIZE = batch_size
        self.VOCAB_SIZE = vocab_size
        self.EMBEDDING_SIZE = embedding_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.MAX_QUERY_WORD_SIZE = max_query_word # TODO max_query_word => max_query_word_size
        #self.MAX_QUERY_WORD_SIZE = max_query_word
        self.MAX_TITLE_WORD_SIZE = max_title_word # TODO  max_title_word => max_title_word_size
        self.FILTER_SIZE = filter_size
        self.local_output = None
        self.distrib_output = None
        #self.train_title_num = FLAGS.train_title_num
        self.pairwise = FLAGS.pairwise
        self._input_layer()
        self.TITLE_NUM = self.title_num[0]
        self.optimizer(self.train_query, self.train_title)
        self.eval(self.train_query, self.train_title)
        self.predict_online(self.predict_query, self.predict_title)

        self.merged_summary_op = tf.summary.merge([self.sm_loss_op])

    def _input_layer(self):
        with tf.variable_scope('Train_Inputs'):
            self.train_query = tf.placeholder(dtype=tf.int32, shape=(None, self.MAX_QUERY_WORD_SIZE), name='query')
            self.train_title = tf.placeholder(dtype=tf.int32, shape=(None, self.MAX_TITLE_WORD_SIZE), name='title')
            self.train_labels = tf.placeholder(dtype=tf.int32, shape=(None, ), name='labels')

        with tf.variable_scope('Predict_Inputs'):
            self.predict_query = tf.placeholder(dtype=tf.int32, shape=(None, self.MAX_QUERY_WORD_SIZE), name='query')
            self.predict_title = tf.placeholder(dtype=tf.int32, shape=(None, self.MAX_TITLE_WORD_SIZE), name='title')
            self.predict_labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.title_num = tf.placeholder(dtype=tf.int32, shape=(1,),  name='title_num')

    def _embed_layer(self, query, title):
        with tf.variable_scope('Embedding_layer'), tf.device("/cpu:0"):
            self.embedding_matrix = tf.get_variable(name= "embedding",
                                                    shape=[self.VOCAB_SIZE, self.EMBEDDING_SIZE],
                                                    dtype=tf.float32,
                                                    trainable=False)
            '''
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    initializer=self.word_vec_initializer,
                                                    dtype=tf.float32,
                                                    trainable=False)
            '''
            #self.sm_emx_op = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)
            embedding_query = tf.nn.embedding_lookup(self.embedding_matrix, query)
            embedding_title = tf.nn.embedding_lookup(self.embedding_matrix, title)
            return embedding_query, embedding_title

    def distrib_model(self, query, title, is_training=True, reuse=False,is_online_computing=False):
        with tf.variable_scope('Distrib_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            embedding_query, embedding_title = self._embed_layer(query=query, title=title)
            with tf.variable_scope('distrib_query'):
                query = tf.reshape(embedding_query,
                                   [-1, self.MAX_QUERY_WORD_SIZE, self.EMBEDDING_SIZE, 1])  # [?, max_query_word(=15), self.embedding_size,1]
                conv1 = tf.layers.conv2d(inputs=query, filters=self.FILTER_SIZE,
                                         kernel_size=[3, self.EMBEDDING_SIZE],
                                         activation=tf.nn.tanh, name="conv_query")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.MAX_QUERY_WORD_SIZE - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_query")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.FILTER_SIZE])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.FILTER_SIZE, activation=tf.nn.tanh, name="fc_query")
                self.distrib_query = dense1  # [?, self.filter_size]

            with tf.variable_scope('distrib_title'):
                title = tf.reshape(embedding_title, [-1, self.MAX_TITLE_WORD_SIZE, self.EMBEDDING_SIZE, 1])
                conv1 = tf.layers.conv2d(inputs=title, filters=self.FILTER_SIZE,
                                         kernel_size=[3, self.EMBEDDING_SIZE],
                                         activation=tf.nn.tanh, name="conv_title")  # [?,15-3+1,1, self.filter_size]
                pooling_size = self.MAX_TITLE_WORD_SIZE - 3 + 1
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[pooling_size, 1],
                                                strides=[1, 1], name="pooling_title")  # [?, 1,1 self.filter_size]
                pool1 = tf.reshape(pool1, [-1, self.FILTER_SIZE])  # [?, self.filter_size]
                dense1 = tf.layers.dense(inputs=pool1, units=self.FILTER_SIZE, activation=tf.nn.tanh, name="fc_title")
                self.distrib_title = dense1  # [?, self.filter_size]

                shape_ = tf.shape(self.distrib_title)
                print("shape: ", shape_)
            if is_online_computing == True:
                self.distrib_query = tf.tile(self.distrib_query, [self.TITLE_NUM, 1])

            distrib = tf.multiply(self.distrib_query, self.distrib_title) #[?, self.filter_size]
            distrib = tf.reshape(distrib, [-1, self.FILTER_SIZE]) #[?, self.dims2]

            fuly1 = tf.layers.dense(inputs=distrib, units=self.FILTER_SIZE, activation=tf.nn.tanh)
            drop = tf.layers.dropout(inputs=fuly1, rate=self.keep_prob, training=is_training)  # extra add
            fuly2 = tf.layers.dense(inputs=drop, units=self.FILTER_SIZE, activation=tf.nn.tanh)

            self.distrib_output = fuly2
            print("distrib_output:",self.distrib_output)

            with tf.variable_scope("distrib_match"):
                if is_online_computing == True:
                    embedding_query = tf.tile(embedding_query, [self.TITLE_NUM, 1, 1])

                embedding_title = tf.transpose(embedding_title, perm=[0, 2, 1])
                match_matrix = tf.matmul(embedding_query, embedding_title)
                print("embedding query", embedding_query)
                print("embedding title",embedding_title)
                print("match_matrix", match_matrix)
                conv = tf.layers.conv1d(inputs=match_matrix, filters=self.FILTER_SIZE, kernel_size=[1],
                                        activation=tf.nn.tanh)  # [?,max_query_word,1,self.filter_size]
                conv = tf.reshape(conv,
                                  [-1, self.FILTER_SIZE * self.MAX_QUERY_WORD_SIZE])  # [?,max_query_word*self.filter_size]
                dense1 = tf.layers.dense(inputs=conv, units=self.FILTER_SIZE,
                                         activation=tf.nn.tanh)  # [?, self.filter_size]
                dropout = tf.layers.dropout(inputs=dense1, rate=self.keep_prob, training=is_training) #extra add
                dense2 = tf.layers.dense(inputs=dropout, units=self.FILTER_SIZE, activation=tf.nn.tanh)
                self.match_output = dense2

            return self.distrib_output, self.match_output

    def ensemble_model(self, query, title, is_training=True, reuse=False, is_online_computing = False):
        with tf.variable_scope('emsemble_model'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            distrib_representation, match_representation = self.distrib_model(is_training=is_training, query=query,title=title,\
                                                                              reuse=reuse, is_online_computing = is_online_computing)
            self.model_output = tf.concat([distrib_representation, match_representation], axis=-1)
            fuly = tf.layers.dense(inputs=self.model_output, units=self.FILTER_SIZE, activation=tf.nn.tanh)
            if self.pairwise == False:
                fuly1 = tf.layers.dense(inputs=fuly, units=2, activation=tf.nn.tanh)
            else:
                fuly1 = tf.layers.dense(inputs=fuly, units=1, activation=tf.nn.sigmoid)
        output = fuly1
        return output

    def optimizer(self, query, title):
        self.score = self.ensemble_model(query=query, title=title, is_training=True, reuse=False, is_online_computing=False)  # [batch_size, 1]
        #self.score = tf.squeeze(self.score, -1, name="squeeze")  # [batch_size]
        print("score:",self.score)
        if self.pairwise==False:
            self.predictions = tf.argmax(self.score, 1, name="predictions")
            print("predicitons:", self.predictions)
            labels = tf.one_hot(self.train_labels, depth=2)
            self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.score)
            self.loss = tf.reduce_mean(self.losses)
            self.score = tf.cast(tf.nn.softmax(self.score)[:, 1], dtype=tf.float32, name = "score")
            #self.loss = tf.reduce_mean(tf.square(self.score - self.labels))
            print("pointwise score: ", self.score)
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)
        else:
            #self.score = tf.cast(tf.nn.softmax(self.score)[:, 1], dtype=tf.float32, name="score")
            self.predictions = self.score
            self.score = tf.reshape(self.score, [-1, 2])
            print("pairwise score: ", self.score)
            self.score_pos = self.score[: ,0]
            self.score_neg = self.score[: ,1]
            #self.score_pos = tf.squeeze(self.score[:, 0], -1, name="squeeze_pos")  # [batch_size]
            #self.score_neg = tf.squeeze(self.score[:, 1], -1, name="squeeze_neg")  # [batch_size]
            self.sub = tf.subtract(self.score_pos, self.score_neg, name="pos_sub_neg")

            self.losses = tf.maximum(0.0, tf.subtract(FLAGS.margin, tf.subtract(self.score_pos, self.score_neg)))
            self.loss = tf.reduce_mean(self.losses)
            # self.loss = tf.reduce_mean(tf.log(1.0 + tf.exp(- 2.0 * self.sub)))
            self.sm_loss_op = tf.summary.scalar('Loss', self.loss)


        with tf.name_scope("optimizer"):
            self.optimize_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.90, beta2=0.999,epsilon=1e-08).minimize(self.loss)

    def eval(self, query, title):
        self.predict_score =  self.ensemble_model(
            query=query,
            title=title,
            is_training=False,
            reuse=True,
            is_online_computing=False)
        #self.predict_score = tf.reshape(self.predict_score, [-1, self.title_num])
        print("eval prediciton score : ", self.predict_score)
        if self.pairwise == False:
            self.eval_predictions = tf.argmax(self.predict_score, 1, name="predictions")
            print("eval_predictions: ", self.eval_predictions)
            self.eval_score = tf.cast(tf.nn.softmax(self.predict_score)[:, 1], dtype=tf.float32, name="eval_score")
            print("pointwise eval score: ", self.eval_score)
        else:
            self.eval_score = tf.reshape(self.predict_score, [-1, 2])
            print("pairwise eval score: ", self.eval_score)
        print("eval_score: ", self.eval_score)

    def predict_online(self, query, title):
        #query = tf.tile(tf.expand_dims(query, 1), [1, self.title_num, 1])
        #query = tf.reshape(query, [-1, self.max_query_word])
        #title = tf.reshape(title, [-1, self.max_title_word])
        self.predict_score =  self.ensemble_model(
            query=query,
            title=title,
            is_training=False,
            reuse=True,
            is_online_computing=True)
        self.predict_score = tf.reshape(self.predict_score, [-1], name="output")
        #self.predict_score = tf.reshape(self.predict_score, [-1, self.title_num])
        #self.predict_score = tf.cast(tf.nn.softmax(self.predict_score)[:, 1], dtype=tf.float32, name="predict_score")
        print("predict_score: ", self.predict_score)

