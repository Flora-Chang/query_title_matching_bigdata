# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from test import test_pointwise, test_pairwise
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
    input_title_num = "input_title_num"
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
    title_num = sess.graph.get_tensor_by_name(title_num_tensor_name)
    #feature_local = sess.graph.get_tensor_by_name(feature_local_tensor_name)
    score = sess.graph.get_tensor_by_name(score_tensor_name)

    # _x 实际输入待inference的data
    testing_set = LoadTestData(vocab_dict, FLAGS.human_label_dev_set, query_len_threshold=FLAGS.query_len_threshold,
                               title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
    total_data = 0
    right = 0
    top = 0
    precision_bottom = 0
    recall_bottom = 0
    cnt = 0
    for batch_data in testing_set.next_batch():
        # batch_feature_local, queries, titles, labels, ranks = batch_data
        queries, titles, labels, index = batch_data
        if cnt == 0:
            print("queries: ", queries[0], queries[5], queries[27])
            print("titles: ", titles[0], titles[5], titles[27])
            print("index: ", index[0], index[5], index[27])
        scores = sess.run(score, feed_dict={query: queries,
                                            title: titles,
                                            title_num: 1})
        # print("score:", res[:10])
        # print("label:", labels[:10])
        res = list(zip(index, labels, scores))
        df = pd.DataFrame(res, columns=['index', 'label', 'score'])
        total_data += len(df)
        for i in range(0, len(df)):
            _label = df['label'][i]
            _score = df['score'][i]
            if _label == 1:
                recall_bottom += 1
            if _score >= 0.5:
                precision_bottom += 1
            if _label == 1 and _score >= 0.5:
                top += 1
            if _label == 1 and _score >= 0.5 or _label == 0 and _score < 0.5:
                right += 1
        df = df.drop('label', axis=1)
        if cnt == 0:
            df.to_csv("../predict/predict.csv", mode='a', index=False)
            cnt += 1
        else:
            df.to_csv("../predict/predict.csv", mode='a', index=False, header=False)

print("accuracy：", right / (total_data + 0.001))
print("precision:", top / (precision_bottom + 0.001))
print("recall:", top / (recall_bottom + 0.001))
print("=" * 60)
