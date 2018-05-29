# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from test import test_pairwise, test_pointwise
from util import FLAGS
vocab_dict = get_vocab_dict()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    # 加载结构，模型参数和变量
    print("importing model...")
    signature_key = 'serving_default'
    input_query = 'input_query'
    input_title = 'input_title'
    input_title_num = 'input_title_num'
    #input_feature_local = 'input_feature_local'
    output_score = 'output_score'

    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.save_dir + "saved_model_online")
    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    query_tensor_name = signature[signature_key].inputs[input_query].name
    title_tensor_name = signature[signature_key].inputs[input_title].name
    title_num_tensor_name = signature[signature_key].inputs[input_title_num].name
    #feature_local_tensor_name = signature[signature_key].inputs[input_feature_local].name
    score_tensor_name = signature[signature_key].outputs[output_score].name

    # 获取tensor 并inference
    query = sess.graph.get_tensor_by_name(query_tensor_name)
    title = sess.graph.get_tensor_by_name(title_tensor_name)
    input_title_num = sess.graph.get_tensor_by_name(title_num_tensor_name)
    #feature_local = sess.graph.get_tensor_by_name(feature_local_tensor_name)
    score = sess.graph.get_tensor_by_name(score_tensor_name)

    all_queries = [[4199975, 679, 4199939, 4199977, 34225, 947, 573, 534, 34, 4200000, 828, 1028, 0, 0, 0, 0, 0, 0 , 0, 0],
                   [1,9149,26,1,1,25,1,3403,9149,151,97,42,268,9149,1062,37,71,571,0,0],
                   [3836,21,417,984,1922,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    all_titles = [[4199939, 4199977, 34225, 947, 573, 534, 81, 435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [9149,151,97,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [45,2,3836,265,85,167,984,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    for q, t in zip(all_queries, all_titles):
        queries = [q]
        titles = [t for i in range(20)]
        input_title_nums = len(titles)
        scores = sess.run(score, feed_dict={query: queries,
                                            title: titles,
                                            input_title_num:input_title_nums})
        print("score: ", scores)