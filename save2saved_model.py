# encoding: utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model import Model
from load_data import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector, LoadClickData
from test import test_pointwise, test_pairwise
from util import FLAGS

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
    query = graph.get_tensor_by_name("Predict_Inputs/query:0")
    title = graph.get_tensor_by_name("Predict_Inputs/title:0")
    #feature_local = graph.get_tensor_by_name("Predict_Inputs/feature_local:0")
    #label = graph.get_tensor_by_name("Predict_Inputs/labels:0")
    title_num = graph.get_tensor_by_name("Predict_Inputs/title_num:0")
    # score = graph.get_tensor_by_name("squeeze:0")
    #prediction = graph.get_tensor_by_name("predictions:0")
    score = graph.get_tensor_by_name("output:0")
    print("begin saved_model...")
    
    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.save_dir + "saved_model_online")
    inputs = {'input_query': tf.saved_model.utils.build_tensor_info(query),
              'input_title': tf.saved_model.utils.build_tensor_info(title),
              'input_title_num': tf.saved_model.utils.build_tensor_info(title_num)}
    print("build_tensor_info_query", tf.saved_model.utils.build_tensor_info(query))
    print("build_tensor_info_title", tf.saved_model.utils.build_tensor_info(title))
    #print("build_tensor_info_feature_local", tf.saved_model.utils.build_tensor_info(feature_local))

    outputs = {'output_score': tf.saved_model.utils.build_tensor_info(score)}

    #signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')
    #builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature': signature})

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs,
            outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'serving_default': prediction_signature},
        legacy_init_op=legacy_init_op
    )

    builder.save()
    print("saved_model done..")
