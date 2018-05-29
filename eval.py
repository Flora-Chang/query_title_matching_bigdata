# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector, LoadClickData, termid_to_id
from test import test_pairwise, test_pointwise
from util import FLAGS


vocab_dict = get_vocab_dict()
#vocab_dict = termid_to_id()
test_flag = FLAGS.test_data_flag
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    # 此处需要根据model名字改
    model_path = os.path.join(FLAGS.save_dir, "model.ckpt.meta")
    # 加载结构，模型参数和变量
    print("importing model...")
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint( FLAGS.save_dir))
    #sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    '''
    # 根据次数输出的变量名和操作名确定下边取值的名字
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print(v.name)

    for op in sess.graph.get_operations():
        print(op.name)
    '''
    query = graph.get_tensor_by_name("Train_Inputs/query:0")
    title = graph.get_tensor_by_name("Train_Inputs/title:0")
    #feature_local = graph.get_tensor_by_name("Predict_Inputs/feature_local:0")
    label = graph.get_tensor_by_name("Train_Inputs/labels:0")
    #title_num = graph.get_tensor_by_name("Train_Inputs/title_num:0")
    # score = graph.get_tensor_by_name("squeeze:0")
    if FLAGS.pairwise == False:
        prediction = graph.get_tensor_by_name("predictions_1:0")
        score = graph.get_tensor_by_name("strided_slice_1:0")

    elif FLAGS.pairwise == True:
        score = graph.get_tensor_by_name("Reshape_1:0")

    # _x 实际输入待inference的data
    if test_flag == "human_label":
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
                #print("queries: ", queries[0], queries[5], queries[27])
                #print("titles: ", titles[0], titles[5], titles[27])
                #print("index: ", index[0], index[5], index[27])
            scores = sess.run(score, feed_dict={query: queries,
                                                title: titles})
            # print("score:", res[:10])
            # print("label:", labels[:10])
            scores = np.array(scores).reshape(-1, 1)
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
                df.to_csv(FLAGS.predict_dir, mode='a', index=False)
                cnt += 1
            else:
                df.to_csv(FLAGS.predict_dir, mode='a', index=False, header=False)

        print("accuracy：", right / (total_data + 0.001))
        print("precision:", top / (precision_bottom + 0.001))
        print("recall:", top / (recall_bottom + 0.001))
        print("=" * 60)

    elif test_flag == "click":
        testing_set = LoadClickData(vocab_dict, FLAGS.click_dev_set, query_len_threshold=FLAGS.query_len_threshold,
                                    title_len_threshold=FLAGS.title_len_threshold, batch_size=FLAGS.batch_size)
        total_data = 0
        right = 0
        cnt = 0
        f = open(FLAGS.predict_dir, 'w')
        for batch_data in testing_set.next_batch():
            # batch_feature_local, queries, titles, labels, ranks = batch_data
            queries, titles, labels, index = batch_data
            if cnt == 0:
                print("queries: ", queries[0], queries[5], queries[27])
                print("titles: ", titles[0], titles[5], titles[27])
                #print("index: ", index[0], index[5], index[27])
            scores = sess.run(score, feed_dict={query: queries,
                                                title: titles})
            scores = np.array(scores).reshape([-1, 2])
            scores = list(scores)
            for s in scores:
                cnt += 1
                f.write(str(s[0]) + "\t" + str(s[1]) + "\n")
                if s[0] > s[1]:
                    right += 1
        acc = right / (cnt + 0.000001)
        print("accuracy: ", acc)
        print("cnt")
        print("="*60)

