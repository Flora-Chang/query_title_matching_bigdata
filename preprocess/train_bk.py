# encoding: utf-8
#!/usr/bin/env python
import os
import time
import sys
import numpy as np
import tensorflow as tf
from util import FLAGS
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector, LoadClickData
from test import test_pointwise, test_pairwise

vocab_dict = get_vocab_dict()
#word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
#print("vocab_size: ",vocab_size)
#print("word_vector: ", len(word_vectors))
if FLAGS.pairwise==False:
    training_set = LoadTrainData(vocab_dict,
                             data_path=FLAGS.human_label_training_set,
                             query_len_threshold=FLAGS.query_len_threshold,
                             title_len_threshold=FLAGS.title_len_threshold,
                             batch_size=FLAGS.batch_size)
else:
    training_set = LoadClickData(vocab_dict,
                                 data_path=FLAGS.click_training_set,
                                 query_len_threshold=FLAGS.query_len_threshold,
                                 title_len_threshold=FLAGS.title_len_threshold,
                                 shuffle=True,
                                 batch_size=FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.GPU_rate
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "{}_margin{}_lr{}_bz{}_filter{}".format(FLAGS.flag,
                                            FLAGS.margin,
                                            FLAGS.learning_rate,
                                            FLAGS.batch_size,
                                            FLAGS.filter_size)

    model = Model(max_query_word=FLAGS.query_len_threshold,
                  max_title_word=FLAGS.title_len_threshold,
                  word_vec_initializer=None,
                  batch_size=FLAGS.batch_size,
                  vocab_size=vocab_size,
                  embedding_size=FLAGS.embedding_dim,
                  learning_rate=FLAGS.learning_rate,
                  filter_size=FLAGS.filter_size,
                  keep_prob=FLAGS.keep_prob)

    model_dir = "../models/" + model_name
    best_model_dir = "../models/" + model_name + "-best"
    model_path = os.path.join(model_dir, "model.ckpt")
    best_model_path = os.path.join(best_model_dir, "model.ckpt")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(best_model_dir):
        os.mkdir(best_model_dir)
    #step = tf.get_variable("step", dtype=tf.int32, shape = [1], initializer=tf.constant_initializer(0),trainable=False)
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_epochs = FLAGS.num_epochs
    sys.stderr.write('\nLoading word embeddings from tensor file...\n')
    saver = tf.train.Saver({'embedding': model.embedding_matrix})
    saver.restore(sess, FLAGS.vectors_path)
    sys.stderr.write('\nLoading word embeddings finished !\n')
    saver = tf.train.Saver(tf.global_variables())
    # x 为输入tensor, keep_prob为dropout的prob tensor
    reload = FLAGS.reload
    if reload == True:
        saver.restore(sess, best_model_path)
    score_before = 0
    step = 0
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for batch_data in training_set.next_batch():
            queries, titles, labels,_ = batch_data
            #print("len data: ", len(queries), len(titles), len(labels))
            feed_dict = {model.train_query: queries,
                         model.train_title: titles,
                         model.train_labels: labels}
            _, loss, predictions, scores, summary = \
                sess.run([model.optimize_op, model.loss, model.predictions, model.score, model.merged_summary_op],feed_dict)

            if step % FLAGS.validation_steps == 0:
                if FLAGS.pairwise == False:
                    print("label", labels[0:10])
                    print("predictions: ", predictions[0:10])
                    print("scores: ", scores[:10])
                    print(step, " - loss:", loss)

                    train_set = LoadTestData(vocab_dict, FLAGS.human_label_train_set,
                                             query_len_threshold=FLAGS.query_len_threshold,
                                             title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
                    print("On training set:\n")
                    acc_train, pre_train, recall_train = test_pointwise(sess, model, train_set, filename=None)
                    dev_set = LoadTestData(vocab_dict, FLAGS.human_label_dev_set, query_len_threshold=FLAGS.query_len_threshold,
                                           title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
                    print("On validation set:\n")
                    acc_test, pre_test, recall_test = test_pointwise(sess, model, dev_set, filename=None)
                    score_new = (0.7 * pre_test + 0.3 * recall_test)
                else:
                    #print("predictions: ", predictions[0:10])
                    print("scores: ", scores[:10])
                    print(step, " - loss:", loss)
                    train_set = LoadClickData(vocab_dict, FLAGS.click_train_set,
                                             query_len_threshold=FLAGS.query_len_threshold,
                                             title_len_threshold=FLAGS.title_len_threshold, shuffle=False, batch_size=FLAGS.batch_size)
                    print("On training set:\n")
                    acc_train = test_pairwise(sess, model, train_set, filename=None)
                    dev_set = LoadClickData(vocab_dict, FLAGS.click_dev_set, query_len_threshold=FLAGS.query_len_threshold,
                                           title_len_threshold=FLAGS.title_len_threshold, shuffle=False, batch_size=FLAGS.batch_size)
                    print("On validation set:\n")
                    acc_test = test_pairwise(sess, model, dev_set, filename=None)
                    score_new = acc_test
                saver.save(sess, model_path)
                print("score_new: ", score_new, "score_before: ", score_before)
                if score_new > score_before:
                    score_before = score_new
                    saver.save(sess, best_model_path)


            step += 1

    coord.request_stop()
    coord.join(threads)
