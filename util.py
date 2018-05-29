# encoding: utf-8
import tensorflow as tf


flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("filter_size", 64, "the num of filters of CNN")
flags.DEFINE_integer("embedding_dim", 100, "words embedding size")
flags.DEFINE_float("keep_prob", 0.6, "dropout keep prob")

# change each runing
flags.DEFINE_string("flag", "click", "word/char/drmm")
flags.DEFINE_string("save_dir", "../models/test_lr0.001_bz64_filter64-best/", "save dir")
flags.DEFINE_string("predict_dir", "predict_word_1.csv", "predict result dir")


# Training / test parameters
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("title_len_threshold", 20, "threshold value of document length")
flags.DEFINE_boolean("pairwise", True, "pointwise or pairwise")
#flags.DEFINE_integer("title_num", 20, "the num of titles for each query")
flags.DEFINE_float("margin", 0.5, "learning rate")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("num_epochs",5, "number of epochs")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("pooling_size", 80, "pooling size")
flags.DEFINE_float("validation_steps", 500, "steps between validations")
flags.DEFINE_float("GPU_rate", 0.9, "steps between validations")

flags.DEFINE_string("human_label_training_set", "../data/data_qtitle10w/human_label_train.txt", "training set path")
flags.DEFINE_string("click_training_set", "../data/data_qtitle10w/click_train.txt", "human labeled training set path")
flags.DEFINE_string("human_label_train_set", "../data/data_qtitle10w/human_label_train.test.txt", "click train set path")
flags.DEFINE_string("click_train_set", "../data/data_qtitle10w/click_train.test.txt", "train set path")
flags.DEFINE_string("human_label_dev_set", "../data/data_qtitle10w/human_label_dev.txt", "dev set path")
flags.DEFINE_string("click_dev_set", "../data/data_qtitle10w/click_dev.txt", "the dev set of bigdata")
flags.DEFINE_string("test_data_flag", "human_label", "click/human_label")
flags.DEFINE_string("vocab_path", "../data/data_qtitle10w/word_dict.42000k.txt", "vocab path")
flags.DEFINE_string("vectors_path", "../data/data_qtitle10w/vectors_word.42000k", "vectors path")
flags.DEFINE_string("termid_path", "../data/data_qtitle10w/vocab.exp2hidden", "termid path")

# SavedModel
flags.DEFINE_integer("model_version", 1, "model version") 
flags.DEFINE_string("export_dir", "../models/tmp_lr0.001_bz64_filter64-best/", "export dir")
flags.DEFINE_boolean("reload", False, "whether reload from exists model")
FLAGS = flags.FLAGS

